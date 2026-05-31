# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime monkey-patch installer for the Foundry SGLang integration."""

from __future__ import annotations

import functools
import logging
import os
import time
from dataclasses import asdict

from foundry.integration.sglang import runtime as rt
from foundry.integration.sglang.config import (
    CUDAGraphExtensionMode,
    get_graph_extension_mode,
    get_workspace_root,
    load_graph_extension_config,
)

logger = logging.getLogger(__name__)
_INSTALLED = False


def _resolve_dp_rank(model_runner) -> int | None:
    dp_rank = getattr(model_runner, "dp_rank", None)
    if dp_rank is not None:
        return dp_rank

    server_args = model_runner.server_args
    if getattr(server_args, "enable_dp_attention", False):
        from sglang.srt.layers.dp_attention import compute_dp_attention_world_info

        _, _, dp_rank = compute_dp_attention_world_info(
            server_args.enable_dp_attention,
            model_runner.tp_rank,
            server_args.tp_size,
            server_args.dp_size,
            server_args.attn_cp_size,
        )
        return dp_rank

    if getattr(server_args, "dp_size", 1) > 1:
        raise RuntimeError(
            "Foundry SGLang integration cannot derive regular DP rank because "
            "ModelRunner.dp_rank is absent. Preserve the constructor dp_rank on "
            "ModelRunner before initializing torch distributed."
        )

    return None


def install_hooks(server_args) -> None:
    global _INSTALLED
    cfg_path = getattr(server_args, "foundry_graph_extension_config_path", None)
    if not cfg_path:
        return
    if _INSTALLED:
        return

    t0_ns = os.environ.get("FOUNDRY_SPAWN_T0_NS")
    if t0_ns:
        logger.info(
            "[Foundry] SGLang spawn -> install_hooks: %.1f ms",
            (time.perf_counter_ns() - int(t0_ns)) / 1e6,
        )

    load_graph_extension_config(cfg_path)
    logger.info(
        "[Foundry] SGLang hooks installing: mode=%s workspace=%s",
        get_graph_extension_mode().value,
        get_workspace_root(),
    )

    _patch_init_torch_distributed()
    _patch_init_memory_pool()
    _patch_load_model()
    ep_size = getattr(server_args, "ep_size", 1)
    if ep_size <= 1:
        # TP-only: Foundry controls graph capture
        _patch_kernel_warmup()
        _patch_deepgemm_sync()
        _patch_cuda_graph_capture()
    else:
        # EP: piecewise CUDA graph — patch per-segment capture/load
        _patch_deepgemm_sync()  # sync no-op during capture (safety for JIT in piecewise segments)
        _patch_piecewise_backend()
        _patch_init_device_graphs_ep()
        logger.info("[Foundry] EP mode: piecewise graph capture patched")
    _patch_spawn_sites()

    _INSTALLED = True
    logger.info("[Foundry] SGLang hooks installed")


def _patch_init_torch_distributed() -> None:
    from sglang.srt.model_executor import model_runner as mr

    cls = mr.ModelRunner
    orig = cls.init_torch_distributed

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig(self, *args, **kwargs)

        import torch
        torch.cuda.set_device(self.tp_rank)

        rt.setup_graph_extension(
            self.server_args,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            dp_rank=_resolve_dp_rank(self),
        )

        rt.log_alloc_offset("after_setup_graph_ext")

        result = orig(self, *args, **kwargs)

        rt.log_alloc_offset("after_init_torch_dist")
        rt.skip_to_scratch_boundary()
        rt.log_alloc_offset("after_scratch_skip")
        return result

    cls.init_torch_distributed = patched


def _patch_init_memory_pool() -> None:
    from sglang.srt.model_executor import model_runner_kv_cache_mixin as kv_mixin
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

    cls = kv_mixin.ModelRunnerKVCacheMixin
    orig = cls.init_memory_pool

    @functools.wraps(orig)
    def patched(self, pre_model_load_memory):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig(self, pre_model_load_memory)

        if mode == CUDAGraphExtensionMode.LOAD:
            import torch

            rt.log_alloc_offset("before_init_memory_pool")
            state = rt.load_warmup_state()
            if not state.memory_pool_config:
                raise RuntimeError("Foundry LOAD requires memory_pool_config")
            self.memory_pool_config = MemoryPoolConfig(**state.memory_pool_config)
            # Mirror SAVE's ``_resolve_memory_pool_config`` ->
            # ``get_available_gpu_memory(empty_cache=True)`` side
            # effect. Without this, torch's caching allocator retains
            # segments that SAVE released — causing the
            # attention-backend init below to take a different
            # cuMemAlloc path and drift the VMM cursor away from
            # SAVE's recorded ``start_base_addr``.
            torch.cuda.empty_cache()
            self._apply_memory_pool_config(self.memory_pool_config)
            rt.log_alloc_offset("after_init_memory_pool")
            logger.info("[Foundry] SGLang reused saved memory pool config")
            return None

        rt.log_alloc_offset("before_init_memory_pool")
        result = orig(self, pre_model_load_memory)
        rt.log_alloc_offset("after_init_memory_pool")
        state = rt.create_warmup_state(asdict(self.memory_pool_config))
        rt.save_warmup_state(state)
        return result

    cls.init_memory_pool = patched


def _patch_load_model() -> None:
    from sglang.srt.model_executor import model_runner as mr

    cls = mr.ModelRunner
    orig = cls.load_model

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        return orig(self, *args, **kwargs)

    cls.load_model = patched


def _patch_deepgemm_sync() -> None:
    """Patch DeepGEMM to skip synchronize() during CUDA graph capture.

    DeepGEMM's JIT compiler calls torch.cuda.current_stream().synchronize()
    after warmup kernels. This is illegal during graph capture. Since the
    warmup kernels just get recorded (not executed), the sync is unnecessary.
    """
    try:
        from sglang.srt.layers.deep_gemm_wrapper import compile_utils as dg_compile
    except ImportError:
        return

    if not hasattr(dg_compile, '_compile_deep_gemm_one_type_all'):
        return

    orig_compile_all = dg_compile._compile_deep_gemm_one_type_all
    _in_capture = [False]

    @functools.wraps(orig_compile_all)
    def patched_compile_all(*args, **kwargs):
        if _in_capture[0]:
            # During graph capture: skip warmup execution + synchronize.
            # JIT compile (CPU-side) still runs but warmup kernels
            # are NOT executed (they would be recorded in the graph
            # and corrupt inference on replay).
            import torch
            orig_sync = torch.cuda.Stream.synchronize

            # Patch both synchronize and warmup executor.execute
            torch.cuda.Stream.synchronize = lambda self: None

            try:
                from sglang.srt.layers.deep_gemm_wrapper.compile_utils import _BaseWarmupExecutor
            except ImportError:
                _BaseWarmupExecutor = None

            if _BaseWarmupExecutor is not None:
                orig_execute = _BaseWarmupExecutor.execute
                _BaseWarmupExecutor.execute = lambda self, **kw: None

            try:
                return orig_compile_all(*args, **kwargs)
            finally:
                torch.cuda.Stream.synchronize = orig_sync
                if _BaseWarmupExecutor is not None:
                    _BaseWarmupExecutor.execute = orig_execute
        return orig_compile_all(*args, **kwargs)

    dg_compile._compile_deep_gemm_one_type_all = patched_compile_all
    dg_compile._foundry_in_capture = _in_capture
    logger.info("[Foundry] DeepGEMM synchronize patched (capture-only no-op)")


def _patch_kernel_warmup() -> None:
    from sglang.srt.model_executor import model_runner as mr

    cls = mr.ModelRunner
    orig = cls.kernel_warmup

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig(self, *args, **kwargs)
        # EP with piecewise: run normally (no Foundry graph capture).
        # TP-only with monolithic: run with VMM active for persistent buffers.
        logger.info("[Foundry] SGLang kernel_warmup with VMM active in %s mode", mode.value)
        result = orig(self, *args, **kwargs)
        # Force-compile ALL DeepGEMM shapes that capture might need.
        # kernel_warmup only covers GEMM_NT shapes; MoE EP uses
        # GROUPED_GEMM shapes that need separate precompilation.
        try:
            from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
                _compile_deep_gemm_one_type_all,
                _INITIALIZATION_DICT,
                _BUILTIN_M_LIST,
                DeepGemmKernelType,
            )
            extra_shapes = [
                (4096, 3072, 1), (3072, 3072, 1), (3072, 1536, 1),
                (2048, 3072, 1), (1536, 3072, 1), (3072, 4096, 1),
            ]
            for kt in DeepGemmKernelType:
                for n, k, ng in extra_shapes:
                    query_key = (kt, n, k, ng)
                    if _INITIALIZATION_DICT.get(query_key) is not None:
                        continue
                    try:
                        _compile_deep_gemm_one_type_all(kt, n, k, ng, _BUILTIN_M_LIST)
                        _INITIALIZATION_DICT[query_key] = query_key
                    except Exception:
                        pass
            logger.info("[Foundry] Force-compiled %d DeepGEMM shapes", len(_INITIALIZATION_DICT))
        except ImportError:
            pass
        return result

    cls.kernel_warmup = patched


def _patch_cuda_graph_capture() -> None:
    from sglang.srt.model_executor import cuda_graph_runner as cgr

    cls = cgr.CudaGraphRunner
    orig_capture = cls.capture
    orig_create_device_graph = cls._create_device_graph
    orig_capture_graph = cls._capture_graph
    orig_capture_one_batch_size = cls.capture_one_batch_size

    @functools.wraps(orig_create_device_graph)
    def patched_create_device_graph(self, *args, **kwargs):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.sglang.graph_ops import create_device_graph

            return create_device_graph()
        return orig_create_device_graph(self, *args, **kwargs)

    @functools.wraps(orig_capture_graph)
    def patched_capture_graph(self, graph, pool, stream, run_once_fn):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.sglang.graph_ops import capture_graph

            return capture_graph(graph, pool, stream, run_once_fn)
        return orig_capture_graph(self, graph, pool, stream, run_once_fn)

    @functools.wraps(orig_capture_one_batch_size)
    def patched_capture_one_batch_size(self, bs, forward, stream_idx=None):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.SAVE:
            # Set DeepGEMM capture flag for sync no-op
            try:
                from sglang.srt.layers.deep_gemm_wrapper import compile_utils as dg_compile
                if hasattr(dg_compile, '_foundry_in_capture'):
                    dg_compile._foundry_in_capture[0] = True
            except ImportError:
                pass
            # Suppress the two pre-capture warmup forwards
            # (cuda_graph_runner.py: ``for _ in range(2): run_once()``).
            # Their non-deterministic activation allocations would pollute
            # the torch caching allocator with freed segments that LOAD
            # cannot reproduce — causing the per-bs init's cache-miss vs
            # cache-hit asymmetry that drifts the VMM cursor away from
            # each saved ``start_base_addr``. JIT / autotune still happens
            # inside the graph capture (3rd run_once invocation) and is
            # recorded as alloc events, mirroring vLLM doc 04 §2.
            counter = [0]
            real_forward = forward

            def warmup_skipping_forward(*args, **kwargs):
                counter[0] += 1
                if counter[0] <= 2:
                    return None
                return real_forward(*args, **kwargs)

            forward = warmup_skipping_forward
        try:
            graph, output = orig_capture_one_batch_size(self, bs, forward, stream_idx)
        finally:
            try:
                from sglang.srt.layers.deep_gemm_wrapper import compile_utils as dg_compile
                if hasattr(dg_compile, '_foundry_in_capture'):
                    dg_compile._foundry_in_capture[0] = False
            except ImportError:
                pass
        if mode == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.sglang.graph_ops import save_graph

            # Mirror the inline key shape upstream uses for self.graphs[key]
            # in `_capture_one_stream`. ``_make_graph_key`` and
            # ``get_capture_lora_variant`` were removed in sglang
            # commit ce2506e1c (record_nolora_graph deprecation).
            key = bs if stream_idx is None else f"{stream_idx}_{bs}"
            save_graph(graph, output, key)
        return graph, output

    @functools.wraps(orig_capture)
    def patched(self, *args, **kwargs):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.LOAD:
            from sglang.srt.distributed.device_communicators.pynccl_allocator import (
                set_graph_pool_id,
            )

            from foundry.integration.sglang.graph_ops import (
                initialize_all_attention_metadata,
                load_all_graphs,
            )

            state = rt.get_state()
            if state is None:
                raise RuntimeError("Foundry SGLang state is not initialized")
            # Set up the graph memory pool once — sglang shares one pool
            # across all captured graphs, and runtime replay also requires
            # it to be set so pynccl knows which pool the graph belongs to.
            if cgr.get_global_graph_memory_pool() is None:
                cgr.set_global_graph_memory_pool(self.device_module.graph_pool_handle())
            set_graph_pool_id(cgr.get_global_graph_memory_pool())
            rt.log_alloc_offset("before_preallocate")
            rt.preallocate_for_load_mode()
            rt.log_alloc_offset("after_preallocate")
            # Pre-pass: allocate every per-bs FlashInfer wrapper in
            # ``reversed(capture_bs)`` order, matching the order SAVE used.
            initialize_all_attention_metadata(self)
            rt.log_alloc_offset("after_pre_init")
            # Single ``start_graph_builds(all_paths)`` call so templates
            # and on-demand graphs link via ``shared_exec`` in the
            # manifest. ``finish_graph_loads`` replays alloc events
            # graph-by-graph in the same order SAVE captured them.
            load_all_graphs(self)
            rt.log_alloc_offset("after_load_all_graphs")
            self.graphs = {k: v[0] for k, v in state.loaded_graphs.items()}
            self.output_buffers = {k: v[1] for k, v in state.loaded_graphs.items()}

            # Post-load: stop VMM (vLLM pattern)
            from foundry.allocation_region import stop_allocation_region
            stop_allocation_region()
            logger.info("[Foundry] Stopped VMM after graph load (LOAD)")
            return None

        if mode == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.sglang.graph_ops import (
                initialize_all_attention_metadata,
            )

            rt.log_alloc_offset("save_before_pre_init")

            initialize_all_attention_metadata(self)
            rt.log_alloc_offset("save_after_pre_init")

            attn_backend = self.attn_backend
            attn_backend.forward_metadata = None

            real_init = attn_backend.init_forward_metadata_capture_cuda_graph

            def reuse_pre_pass_init(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            ):
                # The pre-pass already allocated a wrapper for this
                # bs and stored it in
                # ``decode_cuda_graph_metadata`` /
                # ``prefill_cuda_graph_metadata``. Reuse it directly
                # — no second torch.empty for ``_int_workspace_buffer``.
                # Re-run the planner with the same buffer slices the
                # capture forward uses, then point
                # ``forward_metadata`` at the same wrappers. Same
                # plan call on LOAD via the symmetric pre-pass, so
                # the captured graph kernels reference VMM addresses
                # that LOAD's wrappers actually occupy.
                from sglang.srt.layers.attention.flashinfer_backend import (
                    DecodeMetadata,
                    PrefillMetadata,
                )

                if forward_mode.is_decode_or_idle():
                    wrappers = attn_backend.decode_cuda_graph_metadata.get(bs)
                    if wrappers is None:
                        return real_init(
                            bs,
                            num_tokens,
                            req_pool_indices,
                            seq_lens,
                            encoder_lens,
                            forward_mode,
                            spec_info,
                        )
                    seq_lens_sum = seq_lens.sum().item()
                    attn_backend.indices_updater_decode.update(
                        req_pool_indices,
                        seq_lens,
                        seq_lens.cpu(),
                        seq_lens_sum,
                        decode_wrappers=wrappers,
                        encoder_lens=encoder_lens,
                        spec_info=spec_info,
                        fixed_split_size=None,
                        disable_split_kv=attn_backend.disable_cuda_graph_kv_split,
                    )
                    attn_backend.forward_metadata = DecodeMetadata(wrappers)
                    return
                if (
                    forward_mode.is_target_verify()
                    or forward_mode.is_draft_extend()
                    or forward_mode.is_dllm_extend()
                ):
                    wrappers = attn_backend.prefill_cuda_graph_metadata.get(bs)
                    if wrappers is None:
                        return real_init(
                            bs,
                            num_tokens,
                            req_pool_indices,
                            seq_lens,
                            encoder_lens,
                            forward_mode,
                            spec_info,
                        )
                    seq_lens_sum = seq_lens.sum().item()
                    use_ragged = forward_mode.is_dllm_extend()
                    prefix_lens = (
                        seq_lens - attn_backend.dllm_config.block_size
                        if forward_mode.is_dllm_extend()
                        else None
                    )
                    spec_info_arg = None if forward_mode.is_dllm_extend() else spec_info
                    attn_backend.indices_updater_prefill.update(
                        req_pool_indices,
                        seq_lens,
                        seq_lens.cpu(),
                        seq_lens_sum,
                        prefix_lens=prefix_lens,
                        prefill_wrappers=wrappers,
                        use_ragged=use_ragged,
                        encoder_lens=encoder_lens,
                        spec_info=spec_info_arg,
                    )
                    attn_backend.forward_metadata = PrefillMetadata(wrappers, use_ragged, False)
                    return
                # Unknown mode — fall back to real init.
                return real_init(
                    bs,
                    num_tokens,
                    req_pool_indices,
                    seq_lens,
                    encoder_lens,
                    forward_mode,
                    spec_info,
                )

            attn_backend.init_forward_metadata_capture_cuda_graph = reuse_pre_pass_init
            try:
                result = orig_capture(self, *args, **kwargs)
            finally:
                attn_backend.init_forward_metadata_capture_cuda_graph = real_init

            from foundry.integration.sglang.graph_ops import (
                pack_fatbins,
                save_graph_manifest,
            )

            save_graph_manifest()
            pack_fatbins()
            rt.capture_final_alloc_offset()

            return result

        return orig_capture(self, *args, **kwargs)

    cls._create_device_graph = patched_create_device_graph
    cls._capture_graph = patched_capture_graph
    cls.capture_one_batch_size = patched_capture_one_batch_size
    cls.capture = patched


def _load_piecewise_graphs_skip_compile(model_runner) -> None:
    """LOAD-only: create PiecewiseCudaGraphRunner WITHOUT torch.compile.

    Loads all piecewise graphs from disk and installs a replay wrapper
    on model.forward(). This mirrors vLLM's do_not_compile=True pattern.
    """
    import json as json_mod
    import os
    import time

    import torch
    from foundry.graph import CUDAGraph as FoundryCUDAGraph
    from foundry.integration.sglang.graph_ops import _graph_filename
    from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
        PiecewiseCudaGraphRunner,
    )

    cfg = rt.get_config()
    state = rt.get_state()
    if cfg is None or cfg.workspace_dir is None or state is None:
        raise RuntimeError("Foundry workspace not initialized")

    t0 = time.perf_counter()

    # Read piecewise key map
    map_path = os.path.join(cfg.workspace_dir, "piecewise_key_map.json")
    if not os.path.exists(map_path):
        logger.warning("[Foundry] No piecewise_key_map.json, skipping piecewise LOAD")
        return
    with open(map_path) as f:
        key_map = json_mod.load(f)

    # Collect graph paths sorted by capture index
    entries = []
    for key, info in key_map.items():
        if isinstance(info, dict):
            capture_idx = info["idx"]
            meta = {"structure": info.get("structure"), "output_type": info.get("output_type", "tuple")}
        else:
            capture_idx = info
            meta = {"structure": "tensor", "output_type": "tensor"}
        filename = _graph_filename(capture_idx, key)
        filepath = os.path.join(cfg.workspace_dir, filename)
        if os.path.exists(filepath):
            entries.append((capture_idx, key, filepath, meta))
    entries.sort(key=lambda x: x[0])

    if not entries:
        logger.warning("[Foundry] No piecewise graph files found")
        return

    # Pre-check which graphs have alloc events (only those can be replayed)
    import json as _json
    valid_entries = []
    skipped_no_events = 0
    for cap_idx, key, filepath, meta in entries:
        with open(filepath.replace(".cugraph", ".json").replace(filepath.split("/")[-1],
                  filepath.split("/")[-1].replace(".cugraph", ""))) as f:
            pass  # Just checking existence
        # Read the JSON to check alloc events
        json_path = filepath.replace(".cugraph", ".json") if filepath.endswith(".cugraph") else filepath
        # Actually the filepath is already the JSON path from _graph_filename
        try:
            with open(filepath) as f:
                gj = _json.load(f)
            ae = gj.get("allocator_events", {})
            n_events = len(ae.get("events", []))
        except Exception:
            n_events = 0
        if n_events > 0:
            valid_entries.append((cap_idx, key, filepath, meta))
        else:
            skipped_no_events += 1

    logger.info("[Foundry] Piecewise graphs: %d with alloc events, %d skipped (no events)",
                len(valid_entries), skipped_no_events)

    # Build only graphs that have alloc events
    paths = [e[2] for e in valid_entries]
    if not paths:
        logger.warning("[Foundry] No piecewise graphs with alloc events")
        return

    pending = FoundryCUDAGraph.start_graph_builds(paths, num_threads=1)

    loaded = {}
    num_segments = 0
    batch_sizes = set()
    failed = 0
    for i, (_, key, _, meta) in enumerate(valid_entries):
        try:
            graph, loaded_tensors = FoundryCUDAGraph.finish_one_graph_load(pending, i)
        except RuntimeError as e:
            failed += 1
            if failed <= 3:
                logger.warning("[Foundry] Failed to load piecewise graph %s: %s", key, e)
            continue
        structure = meta["structure"]
        output_type = meta.get("output_type", "tuple")

        if structure == "tensor":
            reconstructed = loaded_tensors
        elif isinstance(structure, list):
            if not isinstance(loaded_tensors, (list, tuple)):
                loaded_tensors = [loaded_tensors]
            slots = []
            for s in structure:
                slots.append(loaded_tensors[s] if s is not None else None)
            reconstructed = tuple(slots) if output_type == "tuple" else slots
        else:
            reconstructed = loaded_tensors

        loaded[key] = (graph, reconstructed)
        # Strong reference
        state.loaded_piecewise_graphs[key] = (graph, reconstructed)

        # Parse key: pw_{seg_idx}_{num_tokens}
        parts = key.split("_")
        seg_idx = int(parts[1])
        num_tokens = int(parts[2])
        num_segments = max(num_segments, seg_idx + 1)
        batch_sizes.add(num_tokens)

    logger.info(
        "[Foundry] Loaded %d/%d piecewise graphs (%d segments × %d sizes, %d failed) in %.3fs",
        len(loaded), len(entries), num_segments, len(batch_sizes), failed,
        time.perf_counter() - t0,
    )

    # Use the runner that was already created by init_piecewise_cuda_graphs
    # (with mocked compile/capture). It has proper buffers and attributes.
    runner = model_runner.piecewise_cuda_graph_runner

    # Install replay wrapper on model.forward
    original_forward = model_runner.model.forward

    _replay_first_call = [True]

    # Track which batch sizes have loaded graphs for can_run check
    _loaded_batch_sizes = set()
    for key in loaded:
        parts = key.split("_")
        if parts[1] == "0":  # segment 0 = first segment
            _loaded_batch_sizes.add(int(parts[2]))

    # Override can_run to reject batch sizes without loaded graphs
    _orig_can_run = type(runner).can_run

    def _patched_can_run(self, forward_batch):
        if not _orig_can_run(self, forward_batch):
            return False
        num_tokens = len(forward_batch.input_ids)
        return num_tokens in _loaded_batch_sizes

    type(runner).can_run = _patched_can_run

    def _replay_forward(input_ids, positions, forward_batch, **kwargs):
        num_tokens = input_ids.shape[0]
        first_key = f"pw_0_{num_tokens}"
        if first_key in loaded:
            if _replay_first_call[0]:
                _replay_first_call[0] = False
                from foundry import ops as _cge
                logger.info("[Foundry] First piecewise replay: num_tokens=%d cursor=%d",
                            num_tokens, _cge.get_current_alloc_offset())
            import torch
            # Try with the MATCHING batch size's graphs
            # If current num_tokens doesn't have alloc events,
            # try using the largest batch's graph instead for debug
            for seg_idx in range(num_segments):
                key = f"pw_{seg_idx}_{num_tokens}"
                if key in loaded:
                    try:
                        torch.cuda.synchronize()
                        loaded[key][0].replay()
                        torch.cuda.synchronize()
                    except Exception as e:
                        logger.error("[Foundry] replay FAILED seg %d/%d key=%s: %s",
                                     seg_idx, num_segments, key, e)
                        raise
            last_key = f"pw_{num_segments - 1}_{num_tokens}"
            return loaded[last_key][1]
        return original_forward(input_ids, positions, forward_batch, **kwargs)

    model_runner.model.forward = _replay_forward
    model_runner.piecewise_cuda_graph_runner = runner
    logger.info("[Foundry] Installed piecewise replay wrapper on model.forward")


def _patch_init_device_graphs_ep() -> None:
    """EP graph lifecycle: SAVE → manifest + fatbins, LOAD → preallocate."""
    from sglang.srt.model_executor import model_runner as mr

    cls = mr.ModelRunner
    orig = cls.init_device_graphs

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        mode = get_graph_extension_mode()

        if mode == CUDAGraphExtensionMode.LOAD:
            # No preallocation for EP — per-segment graph load after warmup
            # handles VMM mapping on-demand via the cuMemAlloc hook.
            rt.log_alloc_offset("ep_load_before_monolithic")

        result = orig(self, *args, **kwargs)

        if mode == CUDAGraphExtensionMode.SAVE:
            # Do NOT stop VMM here — piecewise capture runs after this
            # and needs VMM active so alloc events are recorded.
            # VMM will be stopped after piecewise capture completes.
            logger.info("[Foundry] EP SAVE: monolithic done, VMM stays active for piecewise")
        elif mode == CUDAGraphExtensionMode.LOAD:
            # Do NOT stop VMM here — piecewise LOAD (per-segment
            # finish_graph_loads) runs later and needs VMM active.
            logger.info("[Foundry] EP LOAD: monolithic done, VMM stays active for piecewise")

        return result

    cls.init_device_graphs = patched

    # Patch init_piecewise_cuda_graphs to finalize after piecewise completes
    orig_pw = cls.init_piecewise_cuda_graphs

    @functools.wraps(orig_pw)
    def patched_pw(self, *args, **kwargs):
        mode = get_graph_extension_mode()

        if mode == CUDAGraphExtensionMode.LOAD:
            # vLLM pattern: skip torch.compile + capture on LOAD.
            # Let init_piecewise_cuda_graphs run but with mocked compile/capture.
            from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
                PiecewiseCudaGraphRunner as _PCGRunner,
            )
            from unittest.mock import patch as mock_patch
            from contextlib import ExitStack

            # Replace warmup_compile with a minimal version that only
            # calls init_forward_metadata (allocates attention workspace
            # buffers at deterministic VMM addresses) WITHOUT running
            # model.forward (which triggers torch.compile + GPU allocs).
            def _metadata_only_warmup(runner_self, num_tokens=None, **kw):
                if num_tokens is None:
                    return
                import torch
                from sglang.srt.model_executor.forward_batch_info import (
                    ForwardBatch, ForwardMode, CaptureHiddenMode,
                )
                from sglang.srt.layers.dp_attention import (
                    set_dp_buffer_len, set_is_extend_in_batch,
                )
                try:
                    from sglang.srt.model_executor.forward_batch_info import DpPaddingMode
                except ImportError:
                    DpPaddingMode = None

                buffers = runner_self.buffers
                with torch.device(runner_self.device):
                    fb = ForwardBatch(
                        forward_mode=ForwardMode.EXTEND,
                        batch_size=1,
                        input_ids=buffers.input_ids[:num_tokens],
                        input_embeds=None,
                        req_pool_indices=torch.arange(1, device=runner_self.device),
                        seq_lens=torch.tensor([num_tokens], device=runner_self.device),
                        next_token_logits_buffer=None,
                        orig_seq_lens=torch.tensor([num_tokens], device=runner_self.device),
                        seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                        req_to_token_pool=runner_self.model_runner.req_to_token_pool,
                        token_to_kv_pool=runner_self.model_runner.token_to_kv_pool,
                        attn_backend=runner_self.model_runner.attn_backend,
                        out_cache_loc=buffers.out_cache_loc[:num_tokens],
                        out_cache_loc_swa=None,
                        seq_lens_sum=num_tokens,
                        mamba_track_indices=None,
                        mamba_track_mask=None,
                        mamba_track_seqlens=None,
                        encoder_lens=None,
                        return_logprob=False,
                        extend_num_tokens=num_tokens,
                        extend_seq_lens=torch.tensor([num_tokens], device=runner_self.device),
                        extend_prefix_lens=torch.tensor([0], device=runner_self.device),
                        extend_start_loc=torch.tensor([0], device=runner_self.device),
                        extend_prefix_lens_cpu=torch.tensor([0], device="cpu"),
                        extend_seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                        extend_logprob_start_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                        positions=buffers.positions[:num_tokens],
                        global_num_tokens_gpu=None,
                        global_num_tokens_for_logprob_gpu=None,
                        dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph() if DpPaddingMode else None,
                        global_dp_buffer_len=None,
                        mrope_positions=None,
                        spec_algorithm=None,
                        spec_info=None,
                        capture_hidden_mode=CaptureHiddenMode.NULL,
                        num_token_non_padded=None,
                        num_token_non_padded_cpu=num_tokens,
                        global_forward_mode=ForwardMode.EXTEND,
                        lora_ids=None,
                        return_pooled_hidden_states=False,
                    )
                runner_self.model_runner.attn_backend.init_forward_metadata(fb)
                fb.dp_local_start_pos = fb.dp_local_num_tokens = None
                from sglang.srt.layers.dp_attention import (
                    set_dp_buffer_len, set_is_extend_in_batch,
                )
                set_dp_buffer_len(None, num_tokens,
                                  fb.dp_padding_mode.is_max_len() if fb.dp_padding_mode else False)
                set_is_extend_in_batch(False)
                from sglang.srt.compilation.piecewise_context_manager import set_forward_context
                with set_forward_context(
                    fb,
                    runner_self.attention_layers,
                    runner_self.quant_config,
                    runner_self.moe_layers,
                    runner_self.moe_fusions,
                ):
                    # Run model forward WITHOUT torch.compile to allocate
                    # intermediate buffers (MoE gate, expert weights, etc.)
                    # at deterministic VMM addresses.
                    _ = runner_self.model_runner.model.forward(
                        fb.input_ids, fb.positions, fb,
                    )

            with ExitStack() as stack:
                stack.enter_context(mock_patch.object(_PCGRunner, "capture", lambda self: None))
                # Let EVERYTHING run except capture. torch.compile +
                # warmup_compile execute identically to SAVE, ensuring
                # the same VMM allocation trajectory. Only capture() is
                # skipped (replaced with graph load).
                orig_pw(self, *args, **kwargs)

            # Now model_runner has attention_layers, moe_layers, and
            # piecewise_cuda_graph_runner with proper buffers.
            # Load graphs and install replay wrapper.
            _load_piecewise_graphs_skip_compile(self)
            from foundry.allocation_region import stop_allocation_region
            stop_allocation_region()
            logger.info("[Foundry] EP LOAD: piecewise graphs loaded (torch.compile skipped), VMM stopped")
            return

        if mode == CUDAGraphExtensionMode.SAVE:
            # Let EVERYTHING run: warmup_compile + install_torch_compiled.
            # torch.compile creates the same allocation trajectory as LOAD.
            # Only capture is replaced with merged warmup+capture in patched_call.
            result = orig_pw(self, *args, **kwargs)
        else:
            result = orig_pw(self, *args, **kwargs)

        if mode == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.sglang.graph_ops import pack_fatbins
            pack_fatbins()
            rt.capture_final_alloc_offset()
            from foundry.allocation_region import stop_allocation_region
            stop_allocation_region()
            logger.info("[Foundry] EP SAVE: piecewise done, fatbins saved, VMM stopped")

        return result

    cls.init_piecewise_cuda_graphs = patched_pw

    # Patch capture_one_batch_size to flush caching allocator before each batch.
    # This ensures capture starts with empty free lists, forcing all allocations
    # through cuMemAlloc (recorded as alloc events).
    from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
        PiecewiseCudaGraphRunner as _PCGR,
    )
    # Note: empty_cache per batch is NOT safe — it can unmap VMM segments
    # that previous batch's captured graphs reference. Per-segment empty_cache
    # in patched_call is sufficient (previous segment's output is still live).


def _patch_piecewise_backend() -> None:
    """Patch CUDAPiecewiseBackend for per-segment SAVE/LOAD.

    Mirrors vLLM's CUDAGraphWrapper.__call__ pattern: intercept each
    piecewise segment's graph capture and replace torch.cuda.CUDAGraph
    with Foundry's CUDAGraph for persistence.
    """
    try:
        from sglang.srt.compilation import cuda_piecewise_backend as cpb
    except ImportError:
        logger.info("[Foundry] Piecewise backend not found, skipping")
        return

    cls = cpb.CUDAPiecewiseBackend
    orig_call = cls.__call__

    # Counter for unique graph keys across all segments
    _graph_counter = [0]

    @functools.wraps(orig_call)
    def patched_call(self, *args):
        mode = get_graph_extension_mode()

        if mode == CUDAGraphExtensionMode.NONE:
            return orig_call(self, *args)

        # For first run and non-cudagraph shapes, pass through
        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)

        if len(self.sym_shape_indices) == 0:
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.need_to_compile and not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            entry.runnable = self.sglang_backend.compiler_manager.compile(
                self.graph,
                args,
                self.inductor_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape,
            )
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        if cpb.is_in_pcg_torch_compile():
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < 1:
                entry.num_finished_warmup += 1
                if mode == CUDAGraphExtensionMode.SAVE:
                    # vLLM cudagraph_num_of_warmups=0 pattern:
                    # Merge warmup + capture. First execution happens
                    # inside capture_begin/end so ALL allocations are
                    # recorded as alloc events. No separate warmup.
                    import os, json, torch
                    from foundry import ops as cge
                    from foundry.integration.sglang.graph_ops import _graph_filename

                    try:
                        from sglang.srt.layers.deep_gemm_wrapper import compile_utils as dg_compile
                        if hasattr(dg_compile, '_foundry_in_capture'):
                            dg_compile._foundry_in_capture[0] = True
                    except ImportError:
                        pass

                    # NO empty_cache here (vLLM pattern). Foundry's VMM
                    # cursor is monotonic — freeing unmaps pages and makes
                    # addresses invalid. Instead, let the caching allocator
                    # reuse blocks within alive segments. The first batch's
                    # cuMemAlloc events create the segments; subsequent
                    # batches reuse blocks within those same segments.
                    # On LOAD, the first batch's alloc events recreate the
                    # segments, and all batches' replay accesses valid addresses.

                    foundry_graph = cge.CUDAGraph()
                    stream = cpb.get_pcg_capture_stream()

                    from contextlib import ExitStack
                    from unittest.mock import patch as mock_patch
                    with ExitStack() as stack:
                        # Suppress gc.collect and empty_cache DURING capture
                        # (internal code might call them, disrupting capture).
                        stack.enter_context(mock_patch("gc.collect", lambda: None))
                        stack.enter_context(mock_patch("torch.cuda.empty_cache", lambda: None))
                        with torch.cuda.stream(stream):
                            foundry_graph.capture_begin(pool=self.graph_pool, capture_error_mode="thread_local")
                            try:
                                output = entry.runnable(*args)
                            finally:
                                foundry_graph.capture_end()

                    try:
                        from sglang.srt.layers.deep_gemm_wrapper import compile_utils as dg_compile
                        if hasattr(dg_compile, '_foundry_in_capture'):
                            dg_compile._foundry_in_capture[0] = False
                    except ImportError:
                        pass

                    if self.is_last_graph:
                        output = cpb.weak_ref_tensors(output)

                    key = f"pw_{self.piecewise_compile_index}_{runtime_shape}"
                    _graph_counter[0] += 1

                    if isinstance(output, (tuple, list)):
                        structure = []
                        tensors = []
                        for item in output:
                            if isinstance(item, torch.Tensor):
                                structure.append(len(tensors))
                                tensors.append(item)
                            else:
                                structure.append(None)
                        save_tensors = tensors
                        output_type = "tuple" if isinstance(output, tuple) else "list"
                    elif isinstance(output, torch.Tensor):
                        structure = "tensor"
                        save_tensors = output
                        output_type = "tensor"
                    else:
                        structure = "tensor"
                        save_tensors = output
                        output_type = "tensor"

                    cfg = rt.get_config()
                    state = rt.get_state()
                    filename = _graph_filename(state.capture_index, key)
                    graph_path = os.path.join(cfg.workspace_dir, filename)
                    foundry_graph.save(graph_path, save_tensors)
                    state.capture_index += 1

                    if cfg and cfg.workspace_dir:
                        map_path = os.path.join(cfg.workspace_dir, "piecewise_key_map.json")
                        mapping = {}
                        if os.path.exists(map_path):
                            with open(map_path) as f:
                                mapping = json.load(f)
                        mapping[key] = {
                            "idx": state.capture_index - 1,
                            "structure": structure,
                            "output_type": output_type,
                        }
                        with open(map_path, "w") as f:
                            json.dump(mapping, f)

                    entry.output = cpb.weak_ref_tensors(output)
                    entry.cudagraph = foundry_graph
                    return output
                return entry.runnable(*args)

            if mode == CUDAGraphExtensionMode.LOAD:
                # LOAD: per-segment graph load. Alloc events include
                # warmup+capture allocations (warmup was skipped above).
                import os, json as json_mod
                from foundry.graph import CUDAGraph as FoundryCUDAGraph
                from foundry.integration.sglang.graph_ops import _graph_filename

                cfg = rt.get_config()
                state = rt.get_state()
                key = f"pw_{self.piecewise_compile_index}_{runtime_shape}"

                if cfg and cfg.workspace_dir and state:
                    map_path = os.path.join(cfg.workspace_dir, "piecewise_key_map.json")
                    if os.path.exists(map_path):
                        with open(map_path) as f:
                            key_map = json_mod.load(f)
                        if key in key_map:
                            entry_info = key_map[key]
                            capture_idx = entry_info["idx"]
                            structure = entry_info["structure"]
                            output_type = entry_info.get("output_type", "tuple")
                            filename = _graph_filename(capture_idx, key)
                            graph_path = os.path.join(cfg.workspace_dir, filename)
                            if os.path.exists(graph_path):
                                pending = FoundryCUDAGraph.start_graph_builds(
                                    [graph_path], num_threads=1
                                )
                                results = FoundryCUDAGraph.finish_graph_loads(pending)
                                graph, loaded_tensors = results[0]

                                import torch
                                if structure == "tensor":
                                    reconstructed = loaded_tensors
                                elif isinstance(structure, list):
                                    if not isinstance(loaded_tensors, (list, tuple)):
                                        loaded_tensors = [loaded_tensors]
                                    slots = []
                                    for s in structure:
                                        if s is not None:
                                            slots.append(loaded_tensors[s])
                                        else:
                                            slots.append(None)
                                    reconstructed = tuple(slots) if output_type == "tuple" else slots
                                else:
                                    reconstructed = loaded_tensors

                                entry.cudagraph = graph
                                entry.output = cpb.weak_ref_tensors(reconstructed)
                                state.loaded_piecewise_graphs[key] = (graph, reconstructed)
                                graph.replay()
                                return entry.output

            if mode == CUDAGraphExtensionMode.SAVE:
                # SAVE: capture with Foundry graph
                import os, json, torch
                from foundry import ops as cge
                from foundry.integration.sglang.graph_ops import _graph_filename

                # Set DeepGEMM capture flag so sync is a no-op during capture
                try:
                    from sglang.srt.layers.deep_gemm_wrapper import compile_utils as dg_compile
                    if hasattr(dg_compile, '_foundry_in_capture'):
                        dg_compile._foundry_in_capture[0] = True
                except ImportError:
                    pass

                foundry_graph = cge.CUDAGraph()
                stream = cpb.get_pcg_capture_stream()

                from contextlib import ExitStack
                from unittest.mock import patch as mock_patch

                with ExitStack() as stack:
                    if not self.is_first_graph:
                        stack.enter_context(mock_patch("gc.collect", lambda: None))
                        stack.enter_context(mock_patch("torch.cuda.empty_cache", lambda: None))

                    with torch.cuda.stream(stream):
                        foundry_graph.capture_begin(pool=self.graph_pool, capture_error_mode="thread_local")
                        try:
                            output = entry.runnable(*args)
                        finally:
                            foundry_graph.capture_end()

                # Clear capture flag
                try:
                    from sglang.srt.layers.deep_gemm_wrapper import compile_utils as dg_compile
                    if hasattr(dg_compile, '_foundry_in_capture'):
                        dg_compile._foundry_in_capture[0] = False
                except ImportError:
                    pass

                if self.is_last_graph:
                    output = cpb.weak_ref_tensors(output)

                # Analyze output structure: record tensor positions and None slots
                # so LOAD can reconstruct the exact same tuple structure.
                key = f"pw_{self.piecewise_compile_index}_{runtime_shape}"
                _graph_counter[0] += 1

                if isinstance(output, (tuple, list)):
                    structure = []
                    tensors = []
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            structure.append(len(tensors))
                            tensors.append(item)
                        else:
                            structure.append(None)
                    save_tensors = tensors
                    output_type = "tuple" if isinstance(output, tuple) else "list"
                elif isinstance(output, torch.Tensor):
                    structure = "tensor"
                    save_tensors = output
                    output_type = "tensor"
                else:
                    structure = "tensor"
                    save_tensors = output
                    output_type = "tensor"

                # Save graph with tensors only (bypass _pack_output)
                cfg = rt.get_config()
                state = rt.get_state()
                filename = _graph_filename(state.capture_index, key)
                graph_path = os.path.join(cfg.workspace_dir, filename)
                foundry_graph.save(graph_path, save_tensors)
                state.capture_index += 1

                # Record key→capture_idx mapping + output structure
                if cfg and cfg.workspace_dir:
                    map_path = os.path.join(cfg.workspace_dir, "piecewise_key_map.json")
                    mapping = {}
                    if os.path.exists(map_path):
                        with open(map_path) as f:
                            mapping = json.load(f)
                    mapping[key] = {
                        "idx": state.capture_index - 1,
                        "structure": structure,
                        "output_type": output_type,
                    }
                    with open(map_path, "w") as f:
                        json.dump(mapping, f)

                logger.info("[Foundry] Saved piecewise graph %s key=%s structure=%s",
                            filename, key,
                            f"{len(structure)}slots" if isinstance(structure, list) else structure)

                entry.output = cpb.weak_ref_tensors(output)
                entry.cudagraph = foundry_graph
                return output

        # Replay
        entry.cudagraph.replay()
        return entry.output

    cls.__call__ = patched_call
    logger.info("[Foundry] Patched CUDAPiecewiseBackend for per-segment SAVE")


def _patch_spawn_sites() -> None:
    try:
        from sglang.srt.entrypoints import engine as engine_mod
    except Exception:
        engine_mod = None

    if engine_mod is not None:
        orig_launch = engine_mod.Engine._launch_scheduler_processes

        @functools.wraps(orig_launch)
        def patched_launch(self, *args, **kwargs):
            if get_graph_extension_mode() != CUDAGraphExtensionMode.NONE:
                rt.setup_ld_preload_env()
            return orig_launch(self, *args, **kwargs)

        engine_mod.Engine._launch_scheduler_processes = patched_launch

    try:
        from sglang.srt.managers import data_parallel_controller as dpc
    except Exception:
        dpc = None

    if dpc is not None:
        orig_start = dpc.DataParallelController.launch_tensor_parallel_group

        @functools.wraps(orig_start)
        def patched_start(self, *args, **kwargs):
            if get_graph_extension_mode() != CUDAGraphExtensionMode.NONE:
                rt.setup_ld_preload_env()
            return orig_start(self, *args, **kwargs)

        dpc.DataParallelController.launch_tensor_parallel_group = patched_start
