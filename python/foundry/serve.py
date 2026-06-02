#!/usr/bin/env python3
"""foundry-serve: Launch SGLang with Foundry CUDA graph persistence.

Wraps SGLang's launch_server with automatic SAVE/LOAD detection.
No SGLang source modification required — patches ServerArgs at runtime.

Usage:
    FOUNDRY_CACHE=/data/graphs foundry-serve --model MiniMaxAI/MiniMax-M2.7 --tp 4 --ep-size 4
"""
import os
import sys
import tempfile


def _detect_mode(cache_dir):
    warmup = os.path.join(cache_dir, "warmup_state.json")
    return "load" if os.path.exists(warmup) else "save"


def _make_toml(mode, cache_dir):
    content = f'''mode = "{mode}"
base_addr = "0x600000000000"
region_size = "200GB"
workspace_root = "{cache_dir}"
scratch_space_size = "2GB"
'''
    fd, path = tempfile.mkstemp(suffix=".toml", prefix="foundry_")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def _find_hook_library():
    """Find libcuda_hook.so from the foundry package."""
    try:
        import foundry as _f
        hook = os.path.join(os.path.dirname(_f.__file__), "libcuda_hook.so")
        if os.path.exists(hook):
            return hook
    except Exception:
        pass
    return None


def _ensure_ld_preload():
    """Re-exec with LD_PRELOAD if hook library is not loaded."""
    hook = _find_hook_library()
    if hook is None:
        return

    current = os.environ.get("LD_PRELOAD", "")
    if hook in current:
        return  # Already loaded

    # Set LD_PRELOAD and re-exec
    os.environ["LD_PRELOAD"] = f"{hook}:{current}" if current else hook
    os.environ["FOUNDRY_SPAWN_T0_NS"] = str(__import__("time").perf_counter_ns())
    os.execv(sys.executable, [sys.executable] + sys.argv)


def _patch_sglang(toml_path):
    """Monkey-patch SGLang to support Foundry without source modifications."""
    import sglang.srt.server_args as sa

    ServerArgs = sa.ServerArgs

    # Add foundry field if not present
    if not hasattr(ServerArgs, "foundry_graph_extension_config_path"):
        ServerArgs.__annotations__["foundry_graph_extension_config_path"] = "str | None"
        ServerArgs.foundry_graph_extension_config_path = None

    # Patch __post_init__
    orig_post_init = ServerArgs.__post_init__

    def _patched_post_init(self):
        if not getattr(self, "foundry_graph_extension_config_path", None):
            self.foundry_graph_extension_config_path = toml_path

        orig_post_init(self)

        cfg_path = getattr(self, "foundry_graph_extension_config_path", None)
        if cfg_path:
            ep_size = getattr(self, "ep_size", 1)
            if ep_size <= 1:
                self.disable_piecewise_cuda_graph = True
            self.enable_profile_cuda_graph = False
            self.disable_flashinfer_autotune = True
            from foundry.integration.sglang.hooks import install_hooks
            install_hooks(self)

    ServerArgs.__post_init__ = _patched_post_init


def main():
    cache_dir = os.environ.get("FOUNDRY_CACHE")
    if not cache_dir:
        print("Error: Set FOUNDRY_CACHE environment variable", file=sys.stderr)
        print("Usage: FOUNDRY_CACHE=/data/graphs foundry-serve [sglang args...]",
              file=sys.stderr)
        sys.exit(1)

    # Step 1: Ensure LD_PRELOAD (re-execs if needed)
    _ensure_ld_preload()

    # Step 2: Auto-detect SAVE/LOAD
    os.makedirs(cache_dir, exist_ok=True)
    mode = _detect_mode(cache_dir)
    toml_path = _make_toml(mode, cache_dir)

    print(f"[foundry-serve] mode={mode} cache={cache_dir}", file=sys.stderr)

    # Step 3: Set env vars for hook library
    os.environ["FOUNDRY_MODE"] = mode

    # Step 4: Patch SGLang
    _patch_sglang(toml_path)

    # Step 5: Run SGLang
    import runpy
    sys.argv[0] = "sglang.launch_server"
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
