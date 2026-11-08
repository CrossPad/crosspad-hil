"""Scenario registry. Import a builtin module to register it; see `load_builtin()`."""
from __future__ import annotations

import importlib

from crosspad_hil.scenarios.base import (
    Artifact,
    Context,
    Progress,
    Report,
    Scenario,
    argparse_to_params,
    get,
    names,
    params_schema,
    params_to_argparse,
    register,
    run_scenario,
)

# Every scenario this package knows how to run. Each module calls register() at
# import time; `load_builtin()` skips a name whose module is not on disk yet, so a
# scenario can be listed here before it is written without breaking the catalog.
BUILTIN_MODULES: tuple[str, ...] = (
    # P0
    "smoke", "app_churn", "back_churn", "kit_churn", "led_state", "usb_mode_cycle",
    # P1
    "fs_transfer",
    "midi_bench", "midi_stress", "velocity", "speedtest", "rt_glitch",
    "stability", "audio_loopback", "speaker_acoustic", "sampler_record", "ble_midi",
    "waveform_cache", "pitched_kit", "kit_swap_audio", "daw_control", "fw_switch", "fw_update",
    "release_check", "firmware_app", "midi_clock", "song_bounce", "seq_kit_change", "wifi",
    "pad_edit_sticky", "studio_session", "midi_trs_loop", "song_elements", "beat_session",
    "beat_to_song",
    "power_state", "live_set", "perf", "latency", "live_jam", "beat",
    # P2 — composes the above into a release checklist
    "release_gate",
)


def load_builtin() -> list[str]:
    """Import every builtin scenario module so `names()` is complete.

    Returns the scenario module names that imported. A builtin whose module file does
    not exist yet is skipped — only that exact module, never an ImportError from inside
    a module that does exist (those are real bugs and propagate).
    """
    loaded: list[str] = []
    for mod in BUILTIN_MODULES:
        qualified = f"{__name__}.{mod}"
        try:
            importlib.import_module(qualified)
        except ModuleNotFoundError as e:
            if e.name != qualified:
                raise
            continue
        loaded.append(mod)
    return loaded


__all__ = [
    "Artifact", "Context", "Progress", "Report", "Scenario",
    "argparse_to_params", "get", "names", "params_schema", "params_to_argparse",
    "register", "run_scenario", "BUILTIN_MODULES", "load_builtin",
]
