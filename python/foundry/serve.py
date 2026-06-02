#!/usr/bin/env python3
"""foundry-serve: Launch SGLang with Foundry CUDA graph persistence.

Wraps SGLang's launch_server with automatic SAVE/LOAD detection.
No SGLang source modification required — patches ServerArgs at runtime.

Usage:
    FOUNDRY_CACHE=/data/graphs python -m foundry.serve [sglang args...]

    # Or via shell wrapper:
    FOUNDRY_CACHE=/data/graphs foundry-serve --model MiniMaxAI/MiniMax-M2.7 --tp 4 --ep-size 4
"""
import os
import sys
import tempfile
import logging

logger = logging.getLogger("foundry.serve")


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


def _patch_sglang(toml_path):
    """Monkey-patch SGLang to support Foundry without source modifications."""
    import dataclasses
    import sglang.srt.server_args as sa

    ServerArgs = sa.ServerArgs

    # 1. Add foundry_graph_extension_config_path field to dataclass
    if not hasattr(ServerArgs, "foundry_graph_extension_config_path"):
        # Add the field to the class
        ServerArgs.__annotations__["foundry_graph_extension_config_path"] = "str | None"
        ServerArgs.foundry_graph_extension_config_path = None

    # 2. Patch __post_init__ to inject foundry config + hooks
    orig_post_init = ServerArgs.__post_init__

    def _patched_post_init(self):
        # Set foundry config path if not already set
        if not getattr(self, "foundry_graph_extension_config_path", None):
            self.foundry_graph_extension_config_path = toml_path

        # Call original __post_init__
        orig_post_init(self)

        # Install foundry hooks (equivalent to foundry_shim.apply_server_args)
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

    # 3. Setup LD_PRELOAD for hook library
    from foundry.integration.sglang.runtime import setup_ld_preload_env
    setup_ld_preload_env()


def main():
    cache_dir = os.environ.get("FOUNDRY_CACHE")
    if not cache_dir:
        print("Error: Set FOUNDRY_CACHE environment variable", file=sys.stderr)
        print("Usage: FOUNDRY_CACHE=/data/graphs python -m foundry.serve [sglang args...]", file=sys.stderr)
        sys.exit(1)

    os.makedirs(cache_dir, exist_ok=True)
    mode = _detect_mode(cache_dir)
    toml_path = _make_toml(mode, cache_dir)

    print(f"[foundry-serve] mode={mode} cache={cache_dir}", file=sys.stderr)

    # Patch SGLang BEFORE it processes args
    _patch_sglang(toml_path)

    # Set FOUNDRY_MODE env for hook library
    os.environ["FOUNDRY_MODE"] = mode

    # Run SGLang's launch_server as if called from command line
    import runpy
    sys.argv[0] = "sglang.launch_server"
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
