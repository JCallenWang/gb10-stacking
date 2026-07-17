#!/usr/bin/env bash
# ============================================================================
# DGX Spark Two-Node NCCL Setup & Bandwidth Validation
#
# Usage:
#   ./spark_nccl_setup.sh --start   Run the full setup and validation flow
#   ./spark_nccl_setup.sh --clean   Remove all settings/artifacts created by
#                                   this flow (run on each node)
#   ./spark_nccl_setup.sh --help    Show this header
#
# Environment overrides:
#   THRESH_GBPS=180   Pass threshold for the 16G bandwidth test (Gbps)
#   RUN_TIMEOUT=900   Per-test timeout in seconds
#
# Run this script once on each node (both can run in parallel; the nodes
# synchronize at the connectivity stage). The bandwidth test itself is
# executed from Node A only.
#
# Technical notes:
#   - Two-node stacking requires only ONE QSFP cable for full bandwidth
#     (~200 Gbps). A second cable is used for multi-node topologies (ring,
#     or switch-based clusters), which are NOT covered by this script.
#   - Each physical QSFP port is exposed as two logical interfaces (enp1*
#     and enP2p*). Per the playbook, IP configuration uses netplan for all
#     four interfaces so the setup persists across reboots.
#   - Do NOT set NCCL_IB_HCA. NCCL automatically aggregates the available
#     RDMA devices; restricting devices manually reduces bandwidth.
#   - "GPU Direct RDMA Disabled" in NCCL logs is expected on DGX Spark
#     (unified-memory architecture) and is not an error.
#   - The 16G bandwidth test allocates ~45 GiB of unified memory per node.
#     Stop memory-intensive workloads (e.g., inference servers) first, and
#     never run multiple tests concurrently.
# ============================================================================
set -u

# ---------- Tunables (override via environment) ----------
THRESH_GBPS="${THRESH_GBPS:-180}"     # pass threshold for the 16G test (Gbps)
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"     # per-mpirun timeout (seconds)
NCCL_TAG="v2.28.9-1"                  # NCCL version pinned by the playbook
LOG_DIR="$HOME/spark_nccl_setup_logs/$(date +%Y%m%d_%H%M%S)"
SSH_OPTS=(-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o LogLevel=ERROR)

CUDA_HOME="/usr/local/cuda"
MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
NCCL_HOME="$HOME/nccl/build/"
FULL_LDPATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
NETPLAN_FILE="/etc/netplan/40-cx7.yaml"

# ---------- Helpers ----------
C_G='\033[1;32m'; C_R='\033[1;31m'; C_Y='\033[1;33m'; C_B='\033[1;36m'; C_0='\033[0m'
step()  { echo; echo -e "${C_B}========== $* ==========${C_0}"; }
ok()    { echo -e "${C_G}[PASS]${C_0} $*"; }
warn()  { echo -e "${C_Y}[NOTE]${C_0} $*"; }
fail_stop() {  # $1 = message, $2... = possible causes
    echo -e "${C_R}[FAIL]${C_0} $1"; shift
    if (( $# > 0 )); then
        echo "  Possible causes:"
        local c; for c in "$@"; do echo "    - $c"; done
    fi
    read -rp "Force continue anyway? (y/N): " ANS
    [[ "${ANS,,}" == "y" ]] || exit 1
}
confirm_enter() { read -rp "$1 (press Enter to continue): " _; }

# ============================================================================
# Cleanup mode: remove everything created by this flow
# ============================================================================
CX7_IFACES=(enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1)
DOC_IPS=(192.168.100.10 192.168.100.11 192.168.100.14 192.168.100.15 \
         192.168.200.12 192.168.200.13 192.168.200.16 192.168.200.17)
GEN_DIRS=("$HOME/nccl" "$HOME/nccl-tests" "$HOME/spark_nccl_setup_logs")

cleanup_mode() {
    step "Cleanup mode: restore this node to its initial state"
    echo "This affects the local node only; run --clean on the other node as well."
    echo
    echo "Current state:"
    [[ -f "$NETPLAN_FILE" ]] && echo "  [found] netplan config $NETPLAN_FILE" \
                             || echo "  [none ] netplan config $NETPLAN_FILE"
    local ifc ipv4 d
    for ifc in "${CX7_IFACES[@]}"; do
        ipv4=$(ip -4 -o addr show dev "$ifc" 2>/dev/null | awk '{print $4}' | paste -sd, -)
        [[ -n "$ipv4" ]] && echo "  [found] $ifc IPv4: $ipv4"
    done
    for d in "${GEN_DIRS[@]}"; do
        [[ -e "$d" ]] && echo "  [found] $d ($(du -sh "$d" 2>/dev/null | cut -f1))"
    done
    local AK="$HOME/.ssh/authorized_keys" SSH_PAT='spark-|shared-cluster-key'
    [[ -f "$AK" ]] && grep -Eq "$SSH_PAT" "$AK" 2>/dev/null \
        && echo "  [found] $(grep -Ec "$SSH_PAT" "$AK") cluster public key entr(y/ies) in authorized_keys"
    [[ -f "$HOME/.ssh/id_ed25519_shared" ]] \
        && echo "  [found] discover-sparks shared cluster key ~/.ssh/id_ed25519_shared (+.pub, +config)"
    echo
    warn "Removing CX-7 IPs will disconnect the QSFP link between the nodes."
    read -rp "Cleanup mode: all items (a) / confirm each item (i) / cancel (anything else): " MODE
    case "${MODE,,}" in a|i) ;; *) echo "Cancelled."; exit 0 ;; esac
    ask() {  # mode 'a' always proceeds; mode 'i' confirms per item
        [[ "${MODE,,}" == "a" ]] && return 0
        read -rp "  Remove: $1 ? (y/N): " R; [[ "${R,,}" == "y" ]]
    }

    # 1. netplan config (playbook Option 2 artifact)
    if [[ -f "$NETPLAN_FILE" ]] && ask "netplan config $NETPLAN_FILE (followed by netplan apply)"; then
        sudo rm -f "$NETPLAN_FILE" && sudo netplan apply 2>/dev/null
        ok "netplan config removed and reapplied"
    fi
    # 2. Runtime IPv4 on the CX-7 interfaces
    if ask "IPv4 addresses on the four CX-7 interfaces"; then
        for ifc in "${CX7_IFACES[@]}"; do
            ip link show "$ifc" &>/dev/null && sudo ip -4 addr flush dev "$ifc"
        done
        ok "CX-7 interface IPv4 addresses cleared"
    fi
    # 3. NCCL / nccl-tests / logs (official rollback: rm -rf ~/nccl ~/nccl-tests)
    for d in "${GEN_DIRS[@]}"; do
        [[ -e "$d" ]] && ask "$d" && rm -rf "$d" && ok "removed $d"
    done
    # 4. SSH artifacts (always confirmed individually; personal keys are never touched)
    echo
    if [[ -f "$AK" ]] && grep -Eq "$SSH_PAT" "$AK" 2>/dev/null; then
        echo "Cluster public key entries in authorized_keys (lines to be removed;"
        echo "all other/personal keys are preserved):"
        grep -En "$SSH_PAT" "$AK"
        read -rp "Remove these entries (the peer node loses passwordless SSH into this node)? (y/N): " R
        if [[ "${R,,}" == "y" ]]; then
            sed -i.bak -E "/$SSH_PAT/d" "$AK" && ok "removed (backup: $AK.bak)"
        fi
    fi
    if [[ -f "$HOME/.ssh/id_ed25519_shared" ]]; then
        read -rp "Delete the discover-sparks shared cluster key id_ed25519_shared (+.pub)? (y/N): " R
        if [[ "${R,,}" == "y" ]]; then
            rm -f "$HOME/.ssh/id_ed25519_shared" "$HOME/.ssh/id_ed25519_shared.pub"
            ok "shared cluster key deleted"
            if [[ -f "$HOME/.ssh/config" ]] && grep -q "id_ed25519_shared" "$HOME/.ssh/config"; then
                echo "~/.ssh/config contents (references the deleted key):"; sed 's/^/    /' "$HOME/.ssh/config"
                read -rp "Delete this config file as well? (y/N): " R2
                [[ "${R2,,}" == "y" ]] && rm -f "$HOME/.ssh/config" && ok "~/.ssh/config deleted"
            fi
        fi
    fi
    if [[ -f "$HOME/.ssh/known_hosts" ]]; then
        read -rp "Remove known_hosts entries for the playbook IPs (192.168.100/200.x)? (y/N): " R
        if [[ "${R,,}" == "y" ]]; then
            local ip; for ip in "${DOC_IPS[@]}"; do ssh-keygen -R "$ip" >/dev/null 2>&1; done
            ok "known_hosts entries removed"
        fi
    fi
    echo "(The local personal key pair ~/.ssh/id_ed25519 and any non-cluster public keys are never deleted.)"

    # Post-cleanup verification
    step "Post-cleanup verification"
    local LEFT=0
    for ifc in "${CX7_IFACES[@]}"; do
        ipv4=$(ip -4 -o addr show dev "$ifc" 2>/dev/null | awk '{print $4}' | paste -sd, -)
        [[ -n "$ipv4" ]] && { echo "  [remaining] $ifc: $ipv4"; LEFT=1; }
    done
    [[ -f "$NETPLAN_FILE" ]] && { echo "  [remaining] $NETPLAN_FILE"; LEFT=1; }
    for d in "${GEN_DIRS[@]}"; do
        [[ -e "$d" ]] && { echo "  [remaining] $d"; LEFT=1; }
    done
    if (( LEFT == 0 )); then
        ok "This node has been restored to its initial state. Re-run ./spark_nccl_setup.sh to start over."
    else
        warn "Items listed above remain (items skipped in per-item mode are expected). Re-run --clean if needed."
    fi
    exit 0
}

usage() {
    cat <<'EOF'
DGX Spark Two-Node NCCL Setup & Bandwidth Validation

Usage:
  ./spark_nccl_setup.sh --start   Run the full setup and validation flow
  ./spark_nccl_setup.sh --clean   Remove all settings/artifacts created by this flow
  ./spark_nccl_setup.sh --help    Show detailed documentation

Environment overrides (with --start):
  THRESH_GBPS=180   Pass threshold for the 16G bandwidth test (Gbps)
  RUN_TIMEOUT=900   Per-test timeout in seconds

Run with --start once on each node (both may run in parallel).
EOF
}

case "${1:-}" in
    -s|--start|start) ;;
    -c|--clean|clean) cleanup_mode ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    "") usage; exit 0 ;;
    *) echo "Unknown option: $1"; echo; usage; exit 1 ;;
esac

mkdir -p "$LOG_DIR"
echo "All output logs are saved under: $LOG_DIR"

# ============================================================================
step "Stage 0: Select node role"
# ============================================================================
echo "Run this script once on each node (both may run in parallel)."
echo "  Node A = Node 1 in the playbook (the bandwidth test runs from this node)"
echo "  Node B = Node 2 in the playbook"
read -rp "Is this machine Node A or Node B? (A/B): " ROLE
ROLE="${ROLE^^}"
[[ "$ROLE" == "A" || "$ROLE" == "B" ]] || { echo "Please enter A or B."; exit 1; }
ok "Node role: Node $ROLE"

# ============================================================================
step "Stage 1: QSFP cable check"
# ============================================================================
echo "Two-node stacking requires only ONE QSFP cable (either port; both ends"
echo "must be on the same cable). Full bandwidth (~200 Gbps) is achieved with"
echo "a single cable."
confirm_enter "Verify the cable connection between the two nodes"

while :; do
    command -v ibdev2netdev >/dev/null || {
        fail_stop "ibdev2netdev command not found." \
            "NVIDIA DOCA / Mellanox userspace tools are not installed." \
            "Check the DGX OS installation (this tool ships with the standard image)."
        break
    }
    UP_COUNT=$(ibdev2netdev 2>/dev/null | grep -c '(Up)')
    echo "--- ibdev2netdev ---"; ibdev2netdev
    if (( UP_COUNT == 2 )); then
        ok "2 interfaces Up = one cable connected (each cable carries two logical interfaces). Expected."
        break
    elif (( UP_COUNT == 4 )); then
        warn "4 interfaces Up = two cables connected."
        echo "  Two-node stacking needs only ONE cable; a second cable adds no bandwidth"
        echo "  between two nodes. Two cables are used for multi-node topologies (ring or"
        echo "  switch-based), which this script does NOT support. Continuing with the"
        echo "  two-node flow; all four interfaces will be configured via netplan."
        confirm_enter "Continuing with two cables connected"
        break
    else
        fail_stop "$UP_COUNT interface(s) Up (expected 2 with one cable)." \
            "Cable not fully seated at one or both ends." \
            "The two ends are not on the same cable." \
            "Link training failed - reseat the cable or reboot both nodes." \
            "Faulty cable or transceiver - try the other port or another cable."
        read -rp "Press Enter to re-scan (or Ctrl-C to abort): " _
    fi
done

# ============================================================================
step "Stage 2: Hardware scan & PCI validation"
# ============================================================================
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
CX7_FUNCS=$(lspci -d 15b3: 2>/dev/null | wc -l)
BOARD_IDS=$(ibv_devinfo -v 2>/dev/null | awk '/board_id/{print $2}' | sort -u)
MEM_TOTAL=$(free -h | awk '/^Mem:/{print $2}')
echo "GPU                 : ${GPU_NAME:-detection failed}"
echo "CX-7 PCI functions  : $CX7_FUNCS (expected: 4)"
echo "NIC board_id        : $BOARD_IDS"
echo "Unified memory total: $MEM_TOTAL"

[[ "$GPU_NAME" == *GB10* ]] || fail_stop "GPU is not GB10 (got: ${GPU_NAME:-none})." \
    "This script targets DGX Spark systems only." \
    "NVIDIA driver not loaded - check nvidia-smi output."
(( CX7_FUNCS == 4 )) || fail_stop "Unexpected CX-7 PCI function count: $CX7_FUNCS (expected 4)." \
    "NIC not fully enumerated - reboot the system." \
    "Check dmesg for PCIe errors."

echo ""
echo "--- Validating PCI capability and status for CX-7 devices ---"
CX7_PCI_ADDRS=$(lspci -d 15b3: -nn | awk '{print $1}')
for addr in $CX7_PCI_ADDRS; do
    echo "Checking PCI address: $addr"
    sudo lspci -vv -s "$addr" 2>/dev/null | grep -iE "LnkCap|LnkSta"
    echo "  Expected: Speed 32GT/s, Width x4"
done

echo ""
echo "--- Generating hardware topology map ---"
if ! command -v lstopo >/dev/null 2>&1; then
    echo "Installing hwloc module for lstopo..."
    sudo apt-get install hwloc -y || warn "Failed to install hwloc. Skipping topology generation."
fi

if command -v lstopo >/dev/null 2>&1; then
    lstopo topology.png && ok "Topology saved to $(pwd)/topology.png" || warn "Failed to generate topology.png"
fi

# Select the Up enp1* interface (per playbook: use enp1*, disregard enP2p*)
IFACE=$(ibdev2netdev | grep '(Up)' | awk '{print $5}' | grep '^enp1' | sort | tail -1)
[[ -n "$IFACE" ]] || { fail_stop "No Up enp1* interface found." \
    "Cable seated in the wrong port or link is down - re-check Stage 1."; exit 1; }
SPEED=$(ethtool "$IFACE" 2>/dev/null | awk -F': ' '/Speed/{print $2}')
echo "Test interface      : $IFACE (link speed: ${SPEED:-unknown})"
[[ "$SPEED" == "200000Mb/s" ]] && ok "Link speed 200Gb/s confirmed" \
    || fail_stop "Link speed is not 200000Mb/s (got: ${SPEED:-unknown})." \
        "Cable/transceiver does not support 200GbE." \
        "Link negotiated a lower rate - reseat the cable and re-check ethtool."

# ============================================================================
step "Stage 3: IP configuration (playbook Option 2: netplan, persistent)"
# ============================================================================
if [[ "$ROLE" == "A" ]]; then
    IP_F0=192.168.100.10; IP_F1=192.168.200.12; IP_P2F0=192.168.100.14; IP_P2F1=192.168.200.16
else
    IP_F0=192.168.100.11; IP_F1=192.168.200.13; IP_P2F0=192.168.100.15; IP_P2F1=192.168.200.17
fi

WRITE_NETPLAN=1
if [[ -f "$NETPLAN_FILE" ]]; then
    echo "Existing netplan config found: $NETPLAN_FILE"
    sed 's/^/    /' "$NETPLAN_FILE"
    read -rp "Overwrite with the playbook configuration for Node $ROLE? (y/N = keep existing): " ANS
    [[ "${ANS,,}" == "y" ]] || WRITE_NETPLAN=0
fi
if (( WRITE_NETPLAN )); then
    sudo tee "$NETPLAN_FILE" > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses:
        - $IP_F0/24
      dhcp4: no
    enp1s0f1np1:
      addresses:
        - $IP_F1/24
      dhcp4: no
    enP2p1s0f0np0:
      addresses:
        - $IP_P2F0/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - $IP_P2F1/24
      dhcp4: no
EOF
    sudo chmod 600 "$NETPLAN_FILE"
    sudo netplan apply || fail_stop "netplan apply failed." \
        "YAML syntax conflict with another file under /etc/netplan/." \
        "Run 'sudo netplan --debug apply' to see details."
    sleep 2
fi

# Determine the test IPs from the selected interface's subnet
case "$IFACE" in
    enp1s0f0np0)  MY_IP="$IP_F0"; PEER_IP=$([[ "$ROLE" == "A" ]] && echo 192.168.100.11 || echo 192.168.100.10) ;;
    enp1s0f1np1)  MY_IP="$IP_F1"; PEER_IP=$([[ "$ROLE" == "A" ]] && echo 192.168.200.13 || echo 192.168.200.12) ;;
esac
ACTUAL_IP=$(ip -4 -o addr show dev "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
if [[ "$ACTUAL_IP" == "$MY_IP" ]]; then
    ok "Interface $IFACE configured: local $MY_IP <-> peer $PEER_IP"
else
    fail_stop "Interface $IFACE has IP '${ACTUAL_IP:-none}', expected $MY_IP." \
        "An existing netplan file was kept but contains different addresses." \
        "Another netplan file overrides this interface." \
        "Run 'ip -4 addr show $IFACE' and 'ls /etc/netplan/' to inspect."
    MY_IP="${ACTUAL_IP:-$MY_IP}"
fi

# ============================================================================
step "Stage 4: Waiting for peer connectivity"
# ============================================================================
echo "Pinging $PEER_IP ... (the peer node must have completed Stage 3)"
TRIES=0
until ping -c1 -W1 "$PEER_IP" >/dev/null 2>&1; do
    ((TRIES++))
    if (( TRIES % 15 == 0 )); then
        warn "Still waiting after ${TRIES}s. Check that:"
        echo "    - The other node has completed Stage 3 (netplan applied)"
        echo "    - Both ends are on the same physical cable"
        echo "    - The other node selected the opposite role (A vs B)"
        read -rp "Press Enter to keep waiting (or Ctrl-C to abort): " _
    fi
    sleep 1
done
ok "Peer $PEER_IP is reachable"

# ============================================================================
step "Stage 5: Passwordless SSH setup and verification"
# ============================================================================
if ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$PEER_IP" true 2>/dev/null; then
    ok "Passwordless SSH already works"
else
    echo "Setting up SSH keys (you may be prompted once for the peer's password)..."
    [[ -f "$HOME/.ssh/id_ed25519" || -f "$HOME/.ssh/id_rsa" ]] || ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519"
    ssh-copy-id "${SSH_OPTS[@]}" "$PEER_IP" || fail_stop "ssh-copy-id failed." \
        "Wrong password, or password authentication is disabled on the peer." \
        "The same username must exist on both nodes (see playbook prerequisites)."
    ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$PEER_IP" true 2>/dev/null && ok "Passwordless SSH configured" \
        || fail_stop "Passwordless SSH still not working after ssh-copy-id." \
            "Permissions on ~/.ssh or authorized_keys on the peer (must be 700/600)." \
            "sshd on the peer restricts key authentication - check /etc/ssh/sshd_config."
fi

# ============================================================================
step "Stage 6: Build NCCL ($NCCL_TAG, Blackwell sm_121)"
# ============================================================================
[[ -x "$CUDA_HOME/bin/nvcc" ]] || fail_stop "nvcc not found at $CUDA_HOME/bin/nvcc." \
    "CUDA toolkit is not installed - see the DGX Spark documentation." \
    "CUDA installed under a non-default path - set CUDA_HOME accordingly."
if [[ -e "$HOME/nccl/build/lib/libnccl.so.2" ]]; then
    ok "Existing NCCL build detected ($HOME/nccl/build/lib/libnccl.so.2); skipping rebuild"
else
    echo "Installing dependencies and building NCCL (first build takes 10-30 minutes)..."
    sudo apt-get update && sudo apt-get install -y libopenmpi-dev git build-essential \
        || fail_stop "apt package installation failed." \
            "No network access / apt sources unreachable." \
            "Another apt process is holding the lock - retry later."
    [[ -d "$HOME/nccl" ]] || git clone -b "$NCCL_TAG" https://github.com/NVIDIA/nccl.git "$HOME/nccl/" \
        || fail_stop "git clone of NCCL failed." \
            "No network access to github.com." \
            "A partial ~/nccl directory exists - remove it and retry."
    ( cd "$HOME/nccl/" && PATH="$CUDA_HOME/bin:$PATH" \
        make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" ) \
        2>&1 | tee "$LOG_DIR/nccl_build.log" | tail -5
    [[ -e "$HOME/nccl/build/lib/libnccl.so.2" ]] && ok "NCCL build complete" \
        || fail_stop "libnccl.so.2 not found after build (full log: $LOG_DIR/nccl_build.log)." \
            "Compiler/toolchain error - inspect the build log." \
            "Out of disk space - check 'df -h \$HOME'."
fi

# ============================================================================
step "Stage 7: Build nccl-tests (MPI enabled)"
# ============================================================================
if [[ -x "$HOME/nccl-tests/build/all_gather_perf" ]]; then
    ok "Existing nccl-tests build detected; skipping rebuild"
else
    [[ -d "$HOME/nccl-tests" ]] || git clone https://github.com/NVIDIA/nccl-tests.git "$HOME/nccl-tests/" \
        || fail_stop "git clone of nccl-tests failed." \
            "No network access to github.com."
    ( cd "$HOME/nccl-tests/" && PATH="$CUDA_HOME/bin:$PATH" \
        make MPI=1 MPI_HOME="$MPI_HOME" NCCL_HOME="$HOME/nccl/build" CUDA_HOME="$CUDA_HOME" ) \
        2>&1 | tee "$LOG_DIR/nccl_tests_build.log" | tail -5
    [[ -x "$HOME/nccl-tests/build/all_gather_perf" ]] && ok "nccl-tests build complete" \
        || fail_stop "all_gather_perf not found after build (log: $LOG_DIR/nccl_tests_build.log)." \
            "NCCL headers/libraries not found - verify Stage 6 completed." \
            "libopenmpi-dev missing - verify Stage 6 dependency installation."
fi

# ============================================================================
# Node B is done here
# ============================================================================
if [[ "$ROLE" == "B" ]]; then
    step "Node B setup complete"
    ok "All stages finished on this node. Run the test stage from Node A."
    exit 0
fi

# ============================================================================
step "Stage 8 (Node A only): Pre-test safety checks"
# ============================================================================
ssh "${SSH_OPTS[@]}" "$PEER_IP" "test -x \$HOME/nccl-tests/build/all_gather_perf && test -e \$HOME/nccl/build/lib/libnccl.so.2" \
    && ok "Peer node NCCL and nccl-tests are ready" \
    || fail_stop "The peer node ($PEER_IP) has not completed its build stages." \
        "Run this script on Node B first and let it finish Stage 7."

# The 16G test allocates ~45 GiB of unified memory per node; require 55 GiB free.
for TARGET in "local" "peer"; do
    if [[ "$TARGET" == "peer" ]]; then
        AVAIL=$(ssh "${SSH_OPTS[@]}" "$PEER_IP" "free -g | awk '/^Mem:/{print \$7}'")
    else
        AVAIL=$(free -g | awk '/^Mem:/{print $7}')
    fi
    echo "$TARGET node available memory: ${AVAIL} GiB"
    (( AVAIL >= 55 )) || fail_stop "$TARGET node has less than 55 GiB of available memory." \
        "A memory-intensive workload (e.g., an inference server) is running - stop it first." \
        "The 16G test allocates ~45 GiB per node; insufficient memory can hang or crash the system."
done
ok "Both nodes have sufficient free memory"

# ============================================================================
step "Stage 9 (Node A only): NCCL bandwidth test"
# ============================================================================
# Per the playbook: pin only the socket/UCX/MPI interface. Do NOT set
# NCCL_IB_HCA - NCCL aggregates the RDMA devices automatically.
unset NCCL_IB_HCA 2>/dev/null || true
export UCX_NET_DEVICES="$IFACE"
export NCCL_SOCKET_IFNAME="$IFACE"
export OMPI_MCA_btl_tcp_if_include="$IFACE"
export LD_LIBRARY_PATH="$FULL_LDPATH"

run_test() {  # $1 = label, $2... = all_gather_perf args
    local TAG="$1"; shift
    local LOG="$LOG_DIR/test_$TAG.log"
    timeout "$RUN_TIMEOUT" mpirun -np 2 -H "$MY_IP:1,$PEER_IP:1" \
        --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
        -x LD_LIBRARY_PATH -x UCX_NET_DEVICES -x NCCL_SOCKET_IFNAME -x OMPI_MCA_btl_tcp_if_include \
        "$HOME/nccl-tests/build/all_gather_perf" "$@" 2>&1 | tee "$LOG"
    BW=$(awk '/Avg bus bandwidth/ {print $NF}' "$LOG")
    WRONG=$(grep -c "Out of bounds values : 0 OK" "$LOG" || true)
}

echo "--- 9a. Smoke test (default 32MB message; verifies communication and data integrity) ---"
run_test sanity
[[ -n "$BW" && "$WRONG" -ge 1 ]] && ok "Smoke test passed (busbw ${BW} GB/s, 0 data errors)" \
    || fail_stop "Smoke test failed (log: $LOG_DIR/test_sanity.log)." \
        "Firewall blocking traffic between the nodes." \
        "A stale mpirun/orted process from a previous run - check 'ps aux' on both nodes." \
        "LD_LIBRARY_PATH not propagated - inspect the log for 'libnccl.so.2' errors."

echo
echo "--- 9b. Full-bandwidth test (16G message; takes about 1-2 minutes) ---"
run_test full -b 16G -e 16G -f 2
if [[ -z "$BW" ]]; then
    fail_stop "The 16G test produced no result (log: $LOG_DIR/test_full.log)." \
        "Timeout - a node may be memory-starved; re-check Stage 8." \
        "The job hung - kill leftover mpirun/all_gather_perf processes on both nodes and retry."
else
    GBPS=$(awk -v v="$BW" 'BEGIN{printf "%.1f", v*8}')
    echo
    echo "=============================================================="
    echo "  Result   : busbw ${BW} GB/s = ${GBPS} Gbps"
    echo "  Threshold: ${THRESH_GBPS} Gbps (single-cable full bandwidth is ~200 Gbps)"
    if awk -v g="$GBPS" -v t="$THRESH_GBPS" 'BEGIN{exit !(g>=t)}'; then
        echo -e "  Verdict  : ${C_G}PASS - full bandwidth achieved${C_0}"
    else
        echo -e "  Verdict  : ${C_R}FAIL - below ${THRESH_GBPS} Gbps${C_0}"
        fail_stop "Bandwidth below threshold." \
            "NCCL_IB_HCA is set in the environment - it must be unset." \
            "Background workloads consuming memory or GPU on either node." \
            "Cable/transceiver issue - re-check link speed with ethtool on both nodes." \
            "Compare against the smoke test: if 9a was also slow, suspect the link; if only 9b, suspect memory."
    fi
    echo "  Full logs: $LOG_DIR"
    echo "=============================================================="
fi
