"""host_gain against replayed amixer / wpctl / pw-dump output: the bench's CrossPad
card at PCM 30/50 (-20 dB) and a PipeWire node at 0.40, set back to 0 dB."""
from __future__ import annotations

import json
import subprocess

import pytest

from crosspad_hil import host_gain as hg

SCONTROLS = "Simple mixer control 'PCM',0\nSimple mixer control 'Mic',0\n"


def sget_pcm(raw: int, db: float, on: bool = True) -> str:
    sw = "on" if on else "off"
    return (f"Simple mixer control 'PCM',0\n"
            f"  Capabilities: pvolume pswitch pswitch-joined\n"
            f"  Playback channels: Front Left - Front Right\n"
            f"  Limits: Playback 0 - 50\n  Mono:\n"
            f"  Front Left: Playback {raw} [{raw * 2}%] [{db:.2f}dB] [{sw}]\n"
            f"  Front Right: Playback {raw} [{raw * 2}%] [{db:.2f}dB] [{sw}]\n")


def sget_mic(pct: int) -> str:
    return (f"Simple mixer control 'Mic',0\n  Capabilities: cvolume cvolume-joined\n"
            f"  Capture channels: Mono\n  Limits: Capture 0 - 100\n"
            f"  Mono: Capture {pct} [{pct}%]\n")


PW_DUMP = json.dumps([
    {"id": 51, "type": "PipeWire:Interface:Node", "info": {"props": {"node.name": "crosspad_out"}}},
    {"id": 52, "type": "PipeWire:Interface:Node", "info": {"props": {"node.name": "crosspad_mic"}}},
    {"id": 60, "type": "PipeWire:Interface:Node",
     "info": {"props": {"node.name": "alsa_output.usb-Espressif_Crosspad_Audio-00.analog-stereo"}}},
    {"id": 70, "type": "PipeWire:Interface:Node",
     "info": {"props": {"node.name": "alsa_output.pci-0000_00_1f.3.analog-stereo"}}},
])


class Host:
    """amixer + wpctl + pw-dump with state: sset/set-volume change what get reads."""

    def __init__(self, pcm_raw: int = 30, pcm_on: bool = True, mic: int = 100,
                 vols: dict[int, float] | None = None, sticky: bool = False) -> None:
        self.pcm_raw, self.pcm_on, self.mic = pcm_raw, pcm_on, mic
        self.vols = vols if vols is not None else {51: 0.40, 52: 1.0, 60: 1.0}
        self.sticky = sticky          # a session that puts it straight back
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(argv)
        out = ""
        if argv[:1] == ["amixer"]:
            if argv[3] == "scontrols":
                out = SCONTROLS
            elif argv[3] == "sget":
                out = (sget_pcm(self.pcm_raw, (self.pcm_raw - 50) * 1.0, self.pcm_on)
                       if "PCM" in argv[4] else sget_mic(self.mic))
            elif argv[3] == "sset" and not self.sticky:
                if "PCM" in argv[4]:
                    if argv[5] == "0dB":
                        self.pcm_raw = 50
                    elif argv[5] == "unmute":
                        self.pcm_on = True
                elif argv[5] == "100%":
                    self.mic = 100
        elif argv[0] == "pw-dump":
            out = PW_DUMP
        elif argv[:2] == ["wpctl", "get-volume"]:
            out = f"Volume: {self.vols[int(argv[2])]:.2f}\n"
        elif argv[:2] == ["wpctl", "set-volume"] and not self.sticky:
            self.vols[int(argv[2])] = float(argv[3])
        return subprocess.CompletedProcess(argv, 0, out, "")


def every_tool(_name: str) -> bool:
    return True


def test_parsers() -> None:
    assert hg.parse_scontrols(SCONTROLS) == ["'PCM',0", "'Mic',0"]
    assert hg.parse_sget(sget_pcm(30, -20.0)) == [
        {"direction": "playback", "raw": 30, "percent": 60, "db": -20.0, "on": True}]
    assert hg.parse_sget(sget_mic(80)) == [
        {"direction": "capture", "raw": 80, "percent": 80, "db": None, "on": None}]
    assert hg.parse_wpctl_volume("Volume: 0.40 [MUTED]") == (0.40, True)
    assert hg.parse_wpctl_volume("Volume: 1.00") == (1.0, False)
    assert hg.parse_wpctl_volume("nothing") is None
    assert hg.wpctl_db(1.0) == 0.0 and hg.wpctl_db(0.4) == pytest.approx(-23.88, abs=0.01)


def test_only_the_crosspads_nodes_are_touched() -> None:
    ids = {"crosspad_out": 51, "crosspad_mic": 52, "alsa_output.usb-Espressif_Crosspad_Audio-00": 60,
           "alsa_output.pci-0000": 70, "Firefox": 80}
    assert set(hg.crosspad_nodes(ids)) == {"crosspad_out", "crosspad_mic",
                                           "alsa_output.usb-Espressif_Crosspad_Audio-00"}


def test_the_bench_reading_names_every_deviation() -> None:
    g = hg.read("Audio", run=Host(), have=every_tool)
    assert g.deviations() == ["ALSA 'PCM',0 playback -20.0 dB", "PipeWire crosspad_out -23.9 dB"]
    assert g.worst_db() == pytest.approx(-23.88, abs=0.01)
    assert all(n["id"] != 70 for n in g.nodes)


def test_apply_puts_everything_at_zero_db_and_reads_it_back() -> None:
    host = Host(pcm_on=False, mic=80)
    before, after = hg.apply("Audio", run=host, have=every_tool)
    assert "ALSA 'PCM',0 playback muted" in before.deviations()
    assert after.deviations() == []
    assert ["amixer", "-c", "Audio", "sset", "'PCM',0", "0dB"] in host.calls
    assert ["amixer", "-c", "Audio", "sset", "'PCM',0", "unmute"] in host.calls
    assert ["amixer", "-c", "Audio", "sset", "'Mic',0", "100%"] in host.calls
    assert ["wpctl", "set-volume", "51", "1.0"] in host.calls
    assert not any(c[:3] == ["wpctl", "set-volume", "70"] for c in host.calls)


def test_a_volume_that_will_not_move_is_still_reported() -> None:
    _before, after = hg.apply("Audio", run=Host(sticky=True), have=every_tool)
    assert after.deviations() == ["ALSA 'PCM',0 playback -20.0 dB",
                                  "PipeWire crosspad_out -23.9 dB"]


def test_missing_tools_and_card_are_notes_not_errors() -> None:
    g = hg.read(None, run=Host(), have=lambda name: False)
    assert g.controls == [] and g.nodes == [] and g.deviations() == []
    assert any("no ALSA card" in n for n in g.notes)
    assert any("wpctl" in n for n in g.notes)
    d = g.to_dict()
    assert d["worst_db"] is None and d["deviations"] == []


def test_cli_audio_gain_reads_and_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    from crosspad_hil import cli
    from crosspad_hil.devices import Backends

    host = Host()
    real_read, real_apply = hg.read, hg.apply
    monkeypatch.setattr(hg, "read",
                        lambda card, **kw: real_read("Audio", run=host, have=every_tool))
    monkeypatch.setattr(hg, "apply",
                        lambda card, **kw: real_apply("Audio", run=host, have=every_tool))
    out = io.StringIO()
    deps = cli.CliDeps(backends=Backends(list_serial=lambda: [], list_midi=lambda: [],
                                         list_audio=lambda: []),
                       stdout=out, stderr=io.StringIO())
    assert cli.main(["audio", "gain"], deps=deps) == 0
    assert "ALSA 'PCM',0 playback -20.0 dB" in out.getvalue()
    assert cli.main(["audio", "gain", "--set"], deps=deps) == 0
    assert "now 0 dB" in out.getvalue()
    host.sticky, host.pcm_raw = True, 30
    assert cli.main(["audio", "gain", "--set"], deps=deps) == 2
