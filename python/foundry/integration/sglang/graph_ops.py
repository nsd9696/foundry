# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Foundry CUDA graph save/load helpers for SGLang."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import torch

import json
import struct

import foundry as foundry_pkg
from foundry import ops as cge
from foundry.graph import CUDAGraph as FoundryCUDAGraph
from foundry.graph import graph as foundry_graph_ctx
from foundry.integration.sglang.config import (
    CUDAGraphExtensionMode,
    get_config,
    get_graph_extension_mode,
)
from foundry.integration.sglang.runtime import get_state

logger = logging.getLogger(__name__)

_pending_graph_builds: tuple[Any, list[tuple[int, str, dict[str, Any]]]] | None = None
_GRAPH_FILENAME_RE = re.compile(r"^graph_(?P<index>\d+)_FULL_t(?P<bs>\d+)_r\d+_UX_pcN\.json$")


def _batch_size_from_key(key: Any) -> int:
    if isinstance(key, int):
        return key
    key_str = str(key)
    for part in reversed(key_str.split("_")):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot derive batch size from SGLang CUDA graph key: {key!r}")


def _graph_filename(index: int, key: Any) -> str:
    batch_size = _batch_size_from_key(key)
    return f"graph_{index}_FULL_t{batch_size}_r{batch_size}_UX_pcN.json"


def _pack_output(output: Any) -> torch.Tensor:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    if isinstance(output, LogitsProcessorOutput):
        if output.next_token_logits is None:
            raise TypeError("SGLang decode CUDA graph output has no next_token_logits")
        return output.next_token_logits

    if isinstance(output, torch.Tensor):
        return output

    raise TypeError(f"Unsupported SGLang CUDA graph output type: {type(output)!r}")


def _unpack_output(tensors: Any) -> Any:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    if isinstance(tensors, (tuple, list)):
        if len(tensors) != 1:
            raise RuntimeError(f"Expected one SGLang CUDA graph output tensor, got {len(tensors)}")
        tensors = tensors[0]
    return LogitsProcessorOutput(next_token_logits=tensors)


def _grant_all_device_access_on_scratch(cfg) -> None:
    """Grant all-device cuMemSetAccess on the VMM scratch region.

    Binary-patched non-VMM addresses point to VMM scratch (base + 1GB).
    For EP all-to-all, other GPUs need P2P access to this address.
    The default cuMemSetAccess in Foundry's hook only grants local device.
    """
    import ctypes
    from foundry.allocation_region import parse_size

    scratch_addr = cfg.base_addr + parse_size(cfg.scratch_space_size)
    scratch_size = 2 * 1024 * 1024  # 2MB granularity

    libcuda = ctypes.CDLL("libcuda.so.1")

    # Get device count
    dev_count = ctypes.c_int(0)
    libcuda.cuDeviceGetCount(ctypes.byref(dev_count))
    dc = dev_count.value
    if dc <= 1:
        return

    class CUmemAccessDesc(ctypes.Structure):
        _fields_ = [
            ("location_type", ctypes.c_uint),
            ("location_id", ctypes.c_int),
            ("flags", ctypes.c_uint),
        ]

    descs = (CUmemAccessDesc * dc)()
    for i in range(dc):
        descs[i].location_type = 1  # CU_MEM_LOCATION_TYPE_DEVICE
        descs[i].location_id = i
        descs[i].flags = 3  # CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    ret = libcuda.cuMemSetAccess(scratch_addr, scratch_size, descs, dc)
    if ret == 0:
        logger.info("[Foundry] Granted all-device access on VMM scratch 0x%x (%d devices)", scratch_addr, dc)
    else:
        logger.warning("[Foundry] cuMemSetAccess all-device failed (error %d)", ret)


def _premap_non_vmm_addresses(cfg, graph_files) -> None:
    """Pre-map non-VMM addresses found in graph archives.

    During SAVE with NCCL stop/resume, some buffers (e.g., MoE expert
    dispatch) are allocated via standard allocator at non-VMM addresses.
    During LOAD, these addresses don't exist. This function uses CUDA
    driver API to reserve + map memory at those exact addresses so
    graph kernel nodes can reference them.
    """
    import ctypes
    from foundry.allocation_region import parse_size

    vmm_base = cfg.base_addr
    vmm_end = vmm_base + parse_size(cfg.region_size)

    # Collect unique non-VMM addresses from all graph JSONs
    non_vmm_addrs = set()
    for _, filename, _ in graph_files:
        jpath = os.path.join(cfg.workspace_dir, filename)
        with open(jpath) as f:
            g = json.load(f)
        for n in g["nodes"]:
            if n["type"] == "MemsetNode":
                dst = n["params"]["dst"]
                if dst < vmm_base or dst >= vmm_end:
                    non_vmm_addrs.add(dst)

        # Also scan the binary for non-VMM kernel parameter addresses
        # by finding unique addresses that appear in the binary but
        # aren't in VMM. The memset dst is the canonical one.

    if not non_vmm_addrs:
        return

    # Use CUDA driver API to map memory at each non-VMM address
    libcuda = ctypes.CDLL("libcuda.so.1")

    CU_SUCCESS = 0
    CU_MEM_ALLOCATION_TYPE_PINNED = 1
    CU_MEM_LOCATION_TYPE_DEVICE = 1
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
    CU_MEM_HANDLE_TYPE_NONE = 0

    # We need at least 2MB aligned for VMM operations
    ALLOC_ALIGNMENT = 2 * 1024 * 1024

    # Get current device
    device = ctypes.c_int(0)
    libcuda.cuCtxGetDevice(ctypes.byref(device))

    # Get allocation granularity
    granularity = ctypes.c_size_t(0)

    class CUmemAllocationProp(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_uint),
            ("requestedHandleTypes", ctypes.c_uint),
            ("location_type", ctypes.c_uint),
            ("location_id", ctypes.c_int),
            ("win32HandleMetaData", ctypes.c_void_p),
            ("allocFlags", ctypes.c_ulonglong),
        ]

    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location_type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location_id = device.value

    libcuda.cuMemGetAllocationGranularity(
        ctypes.byref(granularity), ctypes.byref(prop), 1  # CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    )
    gran = granularity.value or ALLOC_ALIGNMENT

    for addr in sorted(non_vmm_addrs):
        # Align address down to granularity
        aligned_addr = (addr // gran) * gran
        alloc_size = gran  # Allocate one granularity unit

        # Reserve the virtual address
        reserved = ctypes.c_ulonglong(0)
        ret = libcuda.cuMemAddressReserve(
            ctypes.byref(reserved), alloc_size, gran, aligned_addr, 0
        )
        if ret != CU_SUCCESS:
            logger.warning(
                "[Foundry] Failed to reserve 0x%x (error %d), skipping", aligned_addr, ret
            )
            continue

        if reserved.value != aligned_addr:
            logger.warning(
                "[Foundry] Reserved at 0x%x != requested 0x%x", reserved.value, aligned_addr
            )
            libcuda.cuMemAddressFree(reserved, alloc_size)
            continue

        # Create physical memory
        handle = ctypes.c_ulonglong(0)
        ret = libcuda.cuMemCreate(ctypes.byref(handle), alloc_size, ctypes.byref(prop), 0)
        if ret != CU_SUCCESS:
            logger.warning("[Foundry] cuMemCreate failed (error %d)", ret)
            libcuda.cuMemAddressFree(reserved, alloc_size)
            continue

        # Map physical memory at the reserved address
        ret = libcuda.cuMemMap(aligned_addr, alloc_size, 0, handle, 0)
        if ret != CU_SUCCESS:
            logger.warning("[Foundry] cuMemMap failed at 0x%x (error %d)", aligned_addr, ret)
            libcuda.cuMemRelease(handle)
            libcuda.cuMemAddressFree(reserved, alloc_size)
            continue

        # Set access for current device
        class CUmemAccessDesc(ctypes.Structure):
            _fields_ = [
                ("location_type", ctypes.c_uint),
                ("location_id", ctypes.c_int),
                ("flags", ctypes.c_uint),
            ]

        access = CUmemAccessDesc()
        access.location_type = CU_MEM_LOCATION_TYPE_DEVICE
        access.location_id = device.value
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        ret = libcuda.cuMemSetAccess(aligned_addr, alloc_size, ctypes.byref(access), 1)
        if ret != CU_SUCCESS:
            logger.warning("[Foundry] cuMemSetAccess failed (error %d)", ret)

        logger.info(
            "[Foundry] Pre-mapped non-VMM address 0x%x (%d bytes) for graph LOAD",
            aligned_addr, alloc_size,
        )


def _scan_graph_files(workspace_dir: str) -> list[tuple[int, str, dict[str, Any]]]:
    graph_files = []
    for filename in os.listdir(workspace_dir):
        match = _GRAPH_FILENAME_RE.match(filename)
        if not match:
            continue
        meta = {
            "index": int(match.group("index")),
            "key": int(match.group("bs")),
        }
        graph_files.append((int(meta["index"]), filename, meta))
    graph_files.sort(key=lambda x: x[0])
    return graph_files


def create_device_graph():
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.SAVE:
        return FoundryCUDAGraph()
    return torch.cuda.CUDAGraph()


def capture_graph(graph, pool, stream, run_once_fn):
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.SAVE:
        with foundry_graph_ctx(graph, pool=pool, stream=stream):
            return run_once_fn()
    return None


def save_graph(graph, output: Any, key: Any) -> None:
    cfg = get_config()
    state = get_state()
    if cfg is None or state is None or cfg.workspace_dir is None:
        raise RuntimeError("Foundry SGLang graph extension is not initialized")

    packed_output = _pack_output(output)
    filename = _graph_filename(state.capture_index, key)
    graph_path = os.path.join(cfg.workspace_dir, filename)
    graph.save(graph_path, packed_output)

    state.capture_index += 1
    logger.info("[Foundry] Saved SGLang CUDA graph %s key=%s", filename, key)


def save_graph_manifest() -> None:
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        return
    foundry_pkg.save_graph_manifest(cfg.workspace_dir)


def pack_fatbins() -> None:
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        return
    cge.pack_fatbins_to_folder(cfg.workspace_dir)
    cge.set_pack_fatbins_on_exit(False)


def start_graph_builds() -> None:
    global _pending_graph_builds
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None or cfg.mode != CUDAGraphExtensionMode.LOAD:
        return

    graph_files = _scan_graph_files(cfg.workspace_dir)
    if not graph_files:
        raise RuntimeError(f"No Foundry SGLang graph files found in {cfg.workspace_dir}")

    paths = [os.path.join(cfg.workspace_dir, filename) for _, filename, _ in graph_files]
    t0 = time.perf_counter()
    pending = FoundryCUDAGraph.start_graph_builds(paths, num_threads=4)
    _pending_graph_builds = (pending, graph_files)
    logger.info(
        "[Foundry] Started SGLang graph builds for %d graphs in %.3fs",
        len(paths),
        time.perf_counter() - t0,
    )


def preload_all_graphs() -> None:
    global _pending_graph_builds
    cfg = get_config()
    state = get_state()
    if cfg is None or state is None or cfg.workspace_dir is None:
        raise RuntimeError("Foundry SGLang graph extension is not initialized")

    if _pending_graph_builds is None:
        start_graph_builds()
    assert _pending_graph_builds is not None

    cge.init_nvshmem_for_loaded_modules()

    pending, graph_files = _pending_graph_builds
    _pending_graph_builds = None

    t0 = time.perf_counter()
    results = FoundryCUDAGraph.finish_graph_loads(pending)
    logger.info(
        "[Foundry] Finished SGLang graph loads for %d graphs in %.3fs",
        len(results),
        time.perf_counter() - t0,
    )

    for i, (_index, _filename, meta) in enumerate(graph_files):
        graph, tensors = results[i]
        state.loaded_graphs[meta["key"]] = (graph, _unpack_output(tensors))


def initialize_attention_metadata_for_bs(cuda_graph_runner, bs: int) -> None:
    """Populate ``decode_cuda_graph_metadata[bs]`` for runtime replay.

    The FlashInfer wrappers and their internal ``_int_workspace_buffer``
    are constructed here, outside the captured graph. The graph's
    runtime kernels reference these buffer addresses, so LOAD must
    re-run the same call before runtime replay so the wrappers exist
    at deterministic VMM addresses.
    """
    buffers = cuda_graph_runner.buffers
    num_tokens = bs * cuda_graph_runner.num_tokens_per_bs
    encoder_lens = buffers.encoder_lens[:bs] if cuda_graph_runner.is_encoder_decoder else None
    spec_info = cuda_graph_runner.get_spec_info(num_tokens)
    cuda_graph_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
        bs,
        num_tokens,
        buffers.req_pool_indices[:bs],
        buffers.seq_lens[:bs],
        encoder_lens,
        cuda_graph_runner.capture_forward_mode,
        spec_info,
    )


def initialize_all_attention_metadata(cuda_graph_runner) -> None:
    """Pre-pass: populate ``decode_cuda_graph_metadata`` for all bs at once.

    Called on both SAVE and LOAD before the capture/load loop. Walking
    ``reversed(self.capture_bs)`` (largest first) matches SAVE's natural
    capture order; same order on both sides keeps the VMM cursor
    trajectory identical.
    """
    for bs in reversed(cuda_graph_runner.capture_bs):
        initialize_attention_metadata_for_bs(cuda_graph_runner, bs)


def load_all_graphs(cuda_graph_runner) -> None:
    """LOAD-time replacement for the upstream capture loop.

    All FlashInfer wrappers are pre-allocated by
    ``initialize_all_attention_metadata`` (called by the capture hook
    before this function), so the VMM cursor sits where SAVE recorded
    ``start_base_addr_0``. Load every graph in one
    ``start_graph_builds`` call — this is what enables template +
    on-demand linking in the manifest. ``finish_graph_loads`` then
    replays each graph's alloc events in sequence, advancing the
    cursor exactly the way SAVE did inside its capture loop.
    """
    cfg = get_config()
    state = get_state()
    if cfg is None or state is None or cfg.workspace_dir is None:
        raise RuntimeError("Foundry SGLang graph extension is not initialized")

    graph_files = _scan_graph_files(cfg.workspace_dir)
    if not graph_files:
        raise RuntimeError(f"No Foundry SGLang graph files found in {cfg.workspace_dir}")

    # NVSHMEM init runs once before any graph loads — graphs may reference
    # NVSHMEM symbols. Single-GPU dense models have 0 NVSHMEM modules, so
    # this is a no-op there but kept for EP parity.
    cge.init_nvshmem_for_loaded_modules()

    # All-device access is handled by the C++ hook (cuMemSetAccess
    # grants access to all GPUs at allocation time).
    # _grant_all_device_access_on_scratch(cfg)

    paths = [os.path.join(cfg.workspace_dir, filename) for _, filename, _ in graph_files]
    t0 = time.perf_counter()
    # All graphs in one shot — required for template/on-demand linking
    # in graph_manifest.json (see memory-consistency.md Bug 4).
    # num_threads=4 for template build parallelism. The bg_thread
    # explicitly sets cuCtxSetCurrent(main_ctx) (CUDAGraphParallel.cpp:1897).
    pending = FoundryCUDAGraph.start_graph_builds(paths, num_threads=4)
    results = FoundryCUDAGraph.finish_graph_loads(pending)
    logger.info(
        "[Foundry] Loaded %d SGLang graphs in %.3fs",
        len(results),
        time.perf_counter() - t0,
    )

    for i, (_index, _filename, meta) in enumerate(graph_files):
        graph, tensors = results[i]
        state.loaded_graphs[meta["key"]] = (graph, _unpack_output(tensors))
