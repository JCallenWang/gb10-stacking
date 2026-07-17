# gb10-stacking

This repository contains scripts and utilities for stacking gb10-based machines.

## Reference

Based on the official NVIDIA playbooks:
- [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks)
- [NCCL on Stacked Sparks](https://build.nvidia.com/spark/nccl/stacked-sparks)

## Support

**Currently, this script only supports stacking 2 DGX-Spark machines.**

## Compatibility

- **gb10**: Refers to the GPU used by DGX Spark and similar systems.
- Any machine that follows or uses the architecture of DGX Spark and is manufactured by other OEM companies can use this script to stack 2 gb10-DGX OS compatible machines.

## Scripts

- `spark_nccl_setup.sh`: The main script for setting up the environment.

## Validation & Hardware checks

The script now includes additional hardware and topology validation:
- Validates the PCI addresses for CX-7 (`lspci -d 15b3: -nn`).
- Checks PCI lane and speed capability (`LnkCap` / `LnkSta`) expecting **Speed 32GT/s, Width x4**.
- Automatically installs `hwloc` (`sudo apt-get install hwloc -y`) and exports the hardware topology as a PNG image (`topology.png`) into the current working directory.