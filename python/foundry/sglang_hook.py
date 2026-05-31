# SPDX-License-Identifier: Apache-2.0
"""Auto-patch SGLang for Foundry CUDA graph persistence.

Activated by FOUNDRY_CACHE environment variable. When set, patches SGLang's
ServerArgs to enable Foundry graph extension without modifying SGLang source.

Usage:
    FOUNDRY_CACHE=/data/graphs python -m sglang.launch_server --model ...

First run: SAVE (cache empty) → captures graphs + starts server.
Subsequent runs: LOAD (cache exists) → loads graphs (2.5x faster startup).
"""
import os
import sys
import logging

logger = logging.getLogger("foundry.sglang_hook")

_CACHE_DIR = os.environ.get("FOUNDRY_CACHE")


def _should_activate():
    return _CACHE_DIR is not None


def _detect_mode(workspace_root):
    """Auto-detect SAVE or LOAD based on cache state."""
    warmup_state = os.path.join(workspace_root, "warmup_state.json")
    if os.path.exists(warmup_state):
        return "load"
    return "save"


def _generate_config(mode, workspace_root):
    """Generate Foundry config dict (no TOML file needed)."""
    return {
        "mode": mode,
        "base_addr": 0x600000000000,
        "region_size": "200GB",
        "workspace_root": workspace_root,
        "scratch_space_size": "2GB",
    }


def _write_temp_toml(config):
    """Write temporary TOML config file."""
    import tempfile
    content = f'''mode = "{config["mode"]}"
base_addr = "0x{config["base_addr"]:X}"
region_size = "{config["region_size"]}"
workspace_root = "{config["workspace_root"]}"
scratch_space_size = "{config["scratch_space_size"]}"
'''
    fd, path = tempfile.mkstemp(suffix=".toml", prefix="foundry_")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def _setup_ld_preload():
    """Ensure libcuda_hook.so is in LD_PRELOAD."""
    try:
        import foundry
        hook_path = os.path.join(os.path.dirname(foundry.__file__), "libcuda_hook.so")
        if os.path.exists(hook_path):
            current = os.environ.get("LD_PRELOAD", "")
            if hook_path not in current:
                os.environ["LD_PRELOAD"] = f"{hook_path}:{current}" if current else hook_path
    except ImportError:
        pass


def _patch_sglang():
    """Monkey-patch SGLang's ServerArgs to inject Foundry config."""
    mode = _detect_mode(_CACHE_DIR)
    config = _generate_config(mode, _CACHE_DIR)
    toml_path = _write_temp_toml(config)

    logger.info("[Foundry] Auto-configured: mode=%s cache=%s", mode, _CACHE_DIR)

    # Patch ServerArgs.__post_init__ to set foundry config
    import sglang.srt.server_args as sa

    orig_post_init = sa.ServerArgs.__post_init__

    def _patched_post_init(self):
        # Inject foundry config path
        if not getattr(self, "foundry_graph_extension_config_path", None):
            self.foundry_graph_extension_config_path = toml_path

        orig_post_init(self)

    sa.ServerArgs.__post_init__ = _patched_post_init

    # Ensure foundry_graph_extension_config_path field exists
    if not hasattr(sa.ServerArgs, "foundry_graph_extension_config_path"):
        # Add dataclass field dynamically
        sa.ServerArgs.__annotations__["foundry_graph_extension_config_path"] = "str | None"

    # Ensure foundry_shim integration
    try:
        from foundry.integration.sglang.hooks import install_hooks

        # Patch __post_init__ to also call install_hooks
        _prev_post_init = sa.ServerArgs.__post_init__

        def _final_post_init(self):
            _prev_post_init(self)
            cfg_path = getattr(self, "foundry_graph_extension_config_path", None)
            if cfg_path:
                # Apply foundry settings
                ep_size = getattr(self, "ep_size", 1)
                if ep_size <= 1:
                    self.disable_piecewise_cuda_graph = True
                self.enable_profile_cuda_graph = False
                self.disable_flashinfer_autotune = True
                install_hooks(self)

        sa.ServerArgs.__post_init__ = _final_post_init
    except ImportError:
        logger.warning("[Foundry] Could not import foundry.integration.sglang.hooks")


def activate():
    """Main entry point. Called from .pth file or manually."""
    if not _should_activate():
        return

    _setup_ld_preload()
    _patch_sglang()


# Auto-activate when imported
if _should_activate():
    # Defer patching until sglang is actually imported
    import importlib

    class _FoundryMetaPathFinder:
        """Import hook that patches SGLang when server_args is loaded."""

        _patched = False

        def find_module(self, fullname, path=None):
            if fullname == "sglang.srt.server_args" and not self._patched:
                return self
            return None

        def load_module(self, fullname):
            # Remove ourselves to avoid recursion
            self._patched = True
            # Load the real module
            if fullname in sys.modules:
                return sys.modules[fullname]
            mod = importlib.import_module(fullname)
            # Now patch it
            _patch_sglang()
            return mod

    sys.meta_path.insert(0, _FoundryMetaPathFinder())
