"""What the jam's model takes from the board rather than from the code.

Everything here is a number the firmware decides and the harness cannot read
off a kit or a reply: how long a SysEx pad hit takes to sound, whether a
stolen voice is cut or faded, how a pan or a volume of 50 maps to gain. The
defaults are what the code says (or a safe tolerance where it does not);
``--calibrate`` measures them from a jam's own take (``calibrate_from``) and
writes ``jam_calibration.json`` keyed by the firmware's sha, which later runs
on the same firmware load (``load_for``).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path.home() / ".local" / "state" / "crosspad-hil" / "jam_calibration.json"


@dataclass
class Calibration:
    # host send -> the voice's onset in the take (ms), per path; the analysis
    # re-measures the median per block and segment, this only seeds the search
    l_hit_ms: dict[str, float] = field(default_factory=lambda: {
        "sysex": 8.0, "companion": 30.0, "vpad": 6.0, "cdc": 12.0, "preview": 10.0})
    seq_latency_ms: float = 3.0        # a pass position -> the voice in the take
    seq_jitter_ms: float = 5.0         # poll 2 ms + a block + the pad path
    rec_offset_frames: int = 64        # noteIn() position vs the voice's first frame
    ambiguous_ms: float = 3.0          # a take this close to half a grid line: either line
    steal: str = "cut"                 # sampler: a stolen voice stops at once ("fade": 5 ms)
    retrigger: str = "restart"         # sampler: a retriggered pad restarts its sample
    hold_release_ms: float = 5.0       # sampler hold pad: the fade after the release
    pitched_release_ms: float = 100.0
    pitched_steal_fade_ms: float = 5.0
    pitched_attack_ms: float = 2.0
    seq_note_hold_ms: float = 2.0      # a sequencer note is released on the next poll
    swap_max_load_ms: float = 1500.0   # a pad reload without a KIT_SWAP_LOG row to time it
    swap_grace_s: float = 1.0          # a deferred pad waits at most this for its voice
    engine_switch_ms: float = 2500.0   # sampler <-> pitched: the old engine stops within
    fx_render_ms: float = 60_000.0     # first load of a kit with FX: rendered to .fx/ first
    bounce_start_s: float = 1.0        # 1D 0C -> the scene's first bar, at most
    transport_s: float = 0.5           # FA/FB/FC/UI -> the sequencer follows, at most
    stream_settle_s: float = 0.3       # the phone's stream back -> the pilot steady
    level_offset_db: float = 0.0       # take level vs model, whole take (fitted per block)
    level_tol_db: float = 3.0
    vel_curve: str = "linear"          # gain = vel / 127
    preview_vel: int = 127
    lp_atten_db: tuple[float, float] = (-60.0, -6.0)   # the low-pass pad: somewhere in here
    pan_law: str = "linear"
    capture_bus_gain: float = 1.0      # USB rec output volume (Settings), multiplies everything
    fw_sha: str = ""
    measured: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lp_atten_db"] = list(self.lp_atten_db)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Calibration:
        c = Calibration()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                if k == "lp_atten_db":
                    v = tuple(v)
                elif k == "l_hit_ms":
                    v = {**c.l_hit_ms, **v}
                setattr(c, k, v)
        return c

    def vel_gain(self, vel: int) -> float:
        v = max(0, min(127, int(vel)))
        if self.vel_curve == "square":
            return (v / 127.0) ** 2
        return v / 127.0


def load_for(fw_sha: str, path: Path | None = None) -> Calibration:
    """The calibration measured on this firmware, else the defaults."""
    path = path or DEFAULT_PATH
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Calibration(fw_sha=fw_sha)
    entry = doc.get(fw_sha) if isinstance(doc, dict) else None
    c = Calibration.from_dict(entry or {})
    c.fw_sha = fw_sha
    return c


def save_for(cal: Calibration, path: Path | None = None) -> Path:
    path = Path(path or DEFAULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            doc = {}
    except (OSError, ValueError):
        doc = {}
    doc[cal.fw_sha or "unknown"] = cal.to_dict()
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def calibrate_from(analysis: Any, base: Calibration | None = None) -> Calibration:
    """New numbers from one analysed jam: the measured latency per path, the level
    offset, and what the steal/retrigger edges looked like. Only what the take
    measured with enough samples replaces a default."""
    cal = Calibration.from_dict((base or Calibration()).to_dict())
    m = dict(getattr(analysis, "measured", {}) or {})
    for src, ms in (m.get("l_hit_ms") or {}).items():
        if ms is not None:
            cal.l_hit_ms[src] = round(float(ms), 2)
    if m.get("seq_latency_ms") is not None:
        cal.seq_latency_ms = round(float(m["seq_latency_ms"]), 2)
    if m.get("level_offset_db") is not None:
        cal.level_offset_db = round(float(m["level_offset_db"]), 2)
    cal.measured = m
    return cal
