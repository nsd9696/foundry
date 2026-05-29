#!/usr/bin/env python3
"""Post-SAVE patch: remap non-VMM memset addresses in CUDA graph archives.

During Foundry SAVE with DeepGEMM models, warmup forwards run with VMM
stopped (to pre-compile JIT kernels). This can leave non-VMM addresses in
captured graphs (e.g., padding memset for non-power-of-2 batch sizes).
These addresses don't exist during LOAD, causing cuGraphAddMemsetNode to
fail with CUDA_ERROR_INVALID_VALUE.

This script scans all graph files in a Foundry archive, finds MemsetNode
entries whose dst address is outside the VMM region, and remaps them to a
valid VMM scratch address. Both JSON and binary (.cugraph) files are patched.

Usage:
    python patch_non_vmm_memset.py /path/to/foundry_archive [--base-addr 0x600000000000] [--region-size 256GB]

Example:
    # After SAVE completes:
    python patch_non_vmm_memset.py /tmp/foundry_archive_mm
"""

import argparse
import glob
import json
import os
import struct
import sys


def parse_size(s: str) -> int:
    s = s.strip().upper()
    units = [("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)]
    for suffix, mult in units:
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)


def patch_archive(workspace_root: str, base_addr: int, region_size: int, scratch_offset: int):
    vmm_end = base_addr + region_size
    mapped_addr = base_addr + scratch_offset  # Address within mapped VMM range

    total_patched = 0
    for rank_dir in sorted(glob.glob(os.path.join(workspace_root, "rank_*"))):
        rank = os.path.basename(rank_dir)
        for jf in sorted(glob.glob(os.path.join(rank_dir, "graph_*_FULL_*.json"))):
            with open(jf) as f:
                g = json.load(f)

            non_vmm_addrs = set()
            for n in g["nodes"]:
                if n["type"] == "MemsetNode":
                    dst = n["params"]["dst"]
                    if dst < base_addr or dst >= vmm_end:
                        non_vmm_addrs.add(dst)
                        n["params"]["dst"] = mapped_addr

            if not non_vmm_addrs:
                continue

            # Patch JSON
            with open(jf, "w") as f:
                json.dump(g, f)

            # Patch binary cugraph
            bf = jf.replace(".json", ".cugraph")
            if not os.path.exists(bf):
                print(f"  WARNING: {bf} not found, skipping binary patch")
                continue

            with open(bf, "rb") as f:
                data = f.read()

            for addr in non_vmm_addrs:
                old_bytes = struct.pack("<Q", addr)
                new_bytes = struct.pack("<Q", mapped_addr)
                count = data.count(old_bytes)
                data = data.replace(old_bytes, new_bytes)
                print(f"  {rank}/{os.path.basename(jf)}: {hex(addr)} -> {hex(mapped_addr)} ({count} binary occurrences)")
                total_patched += count

            with open(bf, "wb") as f:
                f.write(data)

    if total_patched == 0:
        print("No non-VMM memset addresses found. Archive is clean.")
    else:
        print(f"\nPatched {total_patched} total occurrences across all ranks.")


def main():
    parser = argparse.ArgumentParser(description="Patch non-VMM memset addresses in Foundry graph archives")
    parser.add_argument("workspace_root", help="Path to Foundry archive (e.g., /tmp/foundry_archive_mm)")
    parser.add_argument("--base-addr", default="0x600000000000", help="VMM base address (default: 0x600000000000)")
    parser.add_argument("--region-size", default="256GB", help="VMM region size (default: 256GB)")
    parser.add_argument("--scratch-offset", default="1GB", help="Offset into VMM for remapped address (default: 1GB)")
    args = parser.parse_args()

    base_addr = int(args.base_addr, 16) if args.base_addr.startswith("0x") else int(args.base_addr)
    region_size = parse_size(args.region_size)
    scratch_offset = parse_size(args.scratch_offset)

    if not os.path.isdir(args.workspace_root):
        print(f"ERROR: {args.workspace_root} is not a directory")
        sys.exit(1)

    print(f"Patching archive: {args.workspace_root}")
    print(f"VMM range: {hex(base_addr)} - {hex(base_addr + region_size)}")
    print(f"Remap target: {hex(base_addr + scratch_offset)}")
    print()

    patch_archive(args.workspace_root, base_addr, region_size, scratch_offset)


if __name__ == "__main__":
    main()
