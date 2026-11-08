"""audio_nodes: the udev watcher that keeps the CrossPad UAC2 nodes alive."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from crosspad_hil import audio_nodes as an

# The service is a systemd user unit, Linux only; its ExecStart is a POSIX path,
# which Path.is_absolute() rejects on Windows.
linux_only = pytest.mark.skipif(sys.platform == "win32", reason="systemd user units are Linux-only")

CARD_PATH = "/devices/pci0000:00/0000:00:02.1/usb1/1-4/1-4:1.0/sound/card4"
PROPS = {"ID_VENDOR_ID": "303a", "ID_MODEL": "Crosspad_Audio", "SOUND_INITIALIZED": "1"}

MONITOR_OUTPUT = f"""monitor will print the received events for:
UDEV - the event which udev sends out after rule processing

UDEV  [1.0] add      {CARD_PATH} (sound)
ACTION=add
DEVPATH={CARD_PATH}
ID_VENDOR_ID=303a
ID_MODEL=Crosspad_Audio

UDEV  [1.1] add      {CARD_PATH}/controlC4 (sound)
ACTION=add
DEVPATH={CARD_PATH}/controlC4

UDEV  [1.2] change   {CARD_PATH} (sound)
ACTION=change
DEVPATH={CARD_PATH}
ID_VENDOR_ID=303a
ID_MODEL=Crosspad_Audio
SOUND_INITIALIZED=1

UDEV  [9.0] remove   {CARD_PATH} (sound)
ACTION=remove
DEVPATH={CARD_PATH}
ID_MODEL=Crosspad_Audio
SOUND_INITIALIZED=1
"""


class FakeNodes:
    def __init__(self) -> None:
        self.card: int | None = None
        self.log: list[str] = []

    @property
    def running(self) -> bool:
        return self.card is not None

    def start(self, card: int) -> None:
        self.card = card
        self.log.append(f"start {card}")

    def stop(self) -> None:
        if self.card is not None:
            self.log.append(f"stop {self.card}")
        self.card = None


# --------------------------------------------------------------------------- parsing

def test_parse_udev_monitor_yields_one_event_per_block() -> None:
    events = list(an.parse_udev_monitor(MONITOR_OUTPUT.splitlines(keepends=True)))
    assert [(e.action, e.card) for e in events] == [
        ("add", 4), ("add", None), ("change", 4), ("remove", 4)]
    assert events[2].props["SOUND_INITIALIZED"] == "1"
    assert events[1].devpath.endswith("/controlC4")


def test_card_index_only_for_card_level_paths() -> None:
    assert an.UdevEvent("add", CARD_PATH).card == 4
    assert an.UdevEvent("add", CARD_PATH + "/pcmC4D0c").card is None


def test_is_crosspad_card_needs_vendor_and_model() -> None:
    assert an.is_crosspad_card(PROPS)
    assert an.is_crosspad_card({**PROPS, "ID_VENDOR_ID": "303A"})
    assert not an.is_crosspad_card({**PROPS, "ID_MODEL": "CrossPad_MIDI+Serial"})
    assert not an.is_crosspad_card({})


# --------------------------------------------------------------------------- node command

def test_node_command_names_both_directions_after_the_card() -> None:
    mic = an.node_command(7, capture=True)
    out = an.node_command(7, capture=False)
    assert mic[:4] == ["pw-cli", "-m", "create-node", "adapter"]
    assert "api.alsa.pcm.source" in mic[4] and 'api.alsa.path="hw:7,0"' in mic[4]
    assert f"node.name={an.MIC_NODE}" in mic[4] and an.MIC_DESCRIPTION in mic[4]
    assert "audio.channels=2" in mic[4]
    assert "api.alsa.pcm.sink" in out[4] and f"node.name={an.OUT_NODE}" in out[4]
    assert "media.class=Audio/Sink" in out[4] and "audio.channels=2" in out[4]



# --------------------------------------------------------------------------- run loop

def test_run_starts_on_initialized_change_and_stops_on_remove(monkeypatch) -> None:
    monkeypatch.setattr(an, "existing_cards", lambda: [])
    nodes = FakeNodes()
    events = list(an.parse_udev_monitor(MONITOR_OUTPUT.splitlines(keepends=True)))
    an.run(lambda _: None, nodes=nodes, events=events, wait=lambda card: True)
    # The bare `add` (no SOUND_INITIALIZED) and the controlC4 child are ignored.
    assert nodes.log == ["start 4", "stop 4"]


def test_run_brings_up_a_card_that_is_already_there(monkeypatch) -> None:
    monkeypatch.setattr(an, "existing_cards", lambda: [4])
    nodes = FakeNodes()
    an.run(lambda _: None, nodes=nodes, events=[], wait=lambda card: True)
    assert nodes.log == ["start 4", "stop 4"]  # the trailing stop is the shutdown


def test_run_skips_a_card_it_cannot_access(monkeypatch) -> None:
    monkeypatch.setattr(an, "existing_cards", lambda: [])
    nodes = FakeNodes()
    lines: list[str] = []
    change = an.UdevEvent("change", CARD_PATH, dict(PROPS))
    an.run(lines.append, nodes=nodes, events=[change], wait=lambda card: False)
    assert nodes.log == []
    assert any("no access" in line for line in lines)


def test_run_ignores_a_repeat_change_for_the_running_card(monkeypatch) -> None:
    monkeypatch.setattr(an, "existing_cards", lambda: [])
    nodes = FakeNodes()
    change = an.UdevEvent("change", CARD_PATH, dict(PROPS))
    an.run(lambda _: None, nodes=nodes, events=[change, change], wait=lambda card: True)
    assert nodes.log == ["start 4", "stop 4"]


def test_run_calls_the_up_hook_each_time_nodes_come_up(monkeypatch) -> None:
    monkeypatch.setattr(an, "existing_cards", lambda: [4])
    nodes = FakeNodes()
    ups: list[int] = []
    change = an.UdevEvent("change", CARD_PATH, dict(PROPS))
    remove = an.UdevEvent("remove", CARD_PATH, {})
    an.run(lambda _: None, nodes=nodes, events=[remove, change], wait=lambda card: True,
           on_up=lambda log: ups.append(1) or True)
    assert nodes.log == ["start 4", "stop 4", "start 4", "stop 4"]
    assert len(ups) == 2


PW_DUMP = [
    {"id": 40, "type": "PipeWire:Interface:Node",
     "info": {"props": {"node.name": "crosspad_mic", "media.class": "Audio/Source"}}},
    {"id": 41, "type": "PipeWire:Interface:Node",
     "info": {"props": {"node.name": "crosspad_out", "media.class": "Audio/Sink"}}},
    {"id": 183, "type": "PipeWire:Interface:Node",
     "info": {"props": {"application.name": "FL Studio", "media.class": "Stream/Input/Audio"}}},
    {"id": 59, "type": "PipeWire:Interface:Node",
     "info": {"props": {"application.name": "FL Studio", "media.class": "Stream/Output/Audio"}}},
    {"id": 77, "type": "PipeWire:Interface:Node",
     "info": {"props": {"application.name": "Firefox", "media.class": "Stream/Output/Audio"}}},
    {"id": 5, "type": "PipeWire:Interface:Port", "info": {"props": {"application.name": "FL Studio"}}},
]


def test_take_defaults_sets_both_and_moves_the_daw_streams() -> None:
    import json
    calls: list[list[str]] = []

    def run_cmd(argv):
        calls.append(argv)
        out = json.dumps(PW_DUMP) if argv == ["pw-dump"] else ""
        return type("R", (), {"stdout": out})()

    assert an.take_defaults(lambda _: None, sleep=lambda s: None, run_cmd=run_cmd,
                            held=lambda: False)
    assert ["wpctl", "set-default", "40"] in calls      # by id, wpctl ignores names
    assert ["wpctl", "set-default", "41"] in calls
    assert ["pw-metadata", "183", "target.object", an.MIC_NODE] in calls
    assert ["pw-metadata", "59", "target.object", an.OUT_NODE] in calls
    # Other applications keep their devices.
    assert not any(c[:2] == ["pw-metadata", "77"] for c in calls)




def test_take_defaults_leaves_the_desktop_alone_while_a_hil_run_holds_it() -> None:
    calls: list[list[str]] = []

    def run_cmd(argv):
        calls.append(argv)
        return type("R", (), {"stdout": "[]"})()

    assert not an.take_defaults(lambda _: None, sleep=lambda s: None, run_cmd=run_cmd,
                                held=lambda: True)
    assert calls == []


def _defaults_dump(sink: str, source: str) -> list[dict]:
    return [
        {"id": 40, "type": "PipeWire:Interface:Node", "info": {"props": {"node.name": an.MIC_NODE}}},
        {"id": 41, "type": "PipeWire:Interface:Node", "info": {"props": {"node.name": an.OUT_NODE}}},
        {"id": 54, "type": "PipeWire:Interface:Node", "info": {"props": {"node.name": "desk_out"}}},
        {"id": 42, "type": "PipeWire:Interface:Node", "info": {"props": {"node.name": "desk_mic"}}},
        {"id": 36, "type": "PipeWire:Interface:Metadata", "props": {"metadata.name": "default"},
         "metadata": [
             {"subject": 0, "key": "default.configured.audio.sink", "value": {"name": an.OUT_NODE}},
             {"subject": 0, "key": "default.audio.sink", "value": {"name": sink}},
             {"subject": 0, "key": "default.audio.source", "value": {"name": source}},
         ]},
    ]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PipeWire holds are Linux-only")
def test_hold_makes_the_devices_in_use_the_configured_ones(tmp_path: Path) -> None:
    """WirePlumber brings a configured default back when its node reappears, and
    take_defaults configured the CrossPad's: the hold re-configures the desktop's."""
    import json
    calls: list[list[str]] = []

    def run_cmd(argv):
        calls.append(argv)
        out = json.dumps(_defaults_dump("desk_out", "desk_mic")) if argv == ["pw-dump"] else ""
        return type("R", (), {"stdout": out})()

    assert an.hold_defaults(lock_dir=tmp_path, run_cmd=run_cmd, pid=os.getpid() + 1_000_000)
    assert ["wpctl", "set-default", "54"] in calls
    assert ["wpctl", "set-default", "42"] in calls


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PipeWire holds are Linux-only")
def test_hold_keeps_a_crosspad_that_already_is_the_default(tmp_path: Path) -> None:
    import json
    calls: list[list[str]] = []

    def run_cmd(argv):
        calls.append(argv)
        out = json.dumps(_defaults_dump(an.OUT_NODE, an.MIC_NODE)) if argv == ["pw-dump"] else ""
        return type("R", (), {"stdout": out})()

    an.hold_defaults(lock_dir=tmp_path, run_cmd=run_cmd, pid=os.getpid() + 1_000_000)
    assert not any(c[:2] == ["wpctl", "set-default"] for c in calls)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PipeWire holds are Linux-only")
def test_a_hold_lasts_as_long_as_its_process(tmp_path: Path) -> None:
    quiet = lambda argv: type("R", (), {"stdout": "[]"})()  # noqa: E731
    assert not an.defaults_held(tmp_path)
    an.hold_defaults(lock_dir=tmp_path, run_cmd=quiet, pid=os.getpid())
    assert an.defaults_held(tmp_path)
    assert not an.hold_defaults(lock_dir=tmp_path, run_cmd=quiet, pid=os.getpid())  # once
    (tmp_path / an.HOLD_DIR / str(os.getpid())).unlink()

    dead = os.getpid() + 1_000_000      # no such process
    an.hold_defaults(lock_dir=tmp_path, run_cmd=quiet, pid=dead)
    assert not an.defaults_held(tmp_path)
    assert not (tmp_path / an.HOLD_DIR / str(dead)).exists()   # stale hold removed


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PipeWire holds are Linux-only")
def test_a_hold_is_never_written_through_a_link_or_a_shared_dir(tmp_path: Path) -> None:
    """Without $XDG_RUNTIME_DIR the lock dir is under the shared temp dir."""
    quiet = lambda argv: type("R", (), {"stdout": "[]"})()  # noqa: E731
    victim = tmp_path / "victim"
    victim.write_text("keep")
    holds = tmp_path / an.HOLD_DIR
    holds.mkdir(mode=0o700)
    (holds / str(os.getpid())).symlink_to(victim)
    assert not an.hold_defaults(lock_dir=tmp_path, run_cmd=quiet, pid=os.getpid())
    assert victim.read_text() == "keep"

    shared = tmp_path / "shared"
    (shared / an.HOLD_DIR).mkdir(parents=True)
    (shared / an.HOLD_DIR).chmod(0o777)
    assert not an.hold_defaults(lock_dir=shared, run_cmd=quiet, pid=os.getpid())
    assert not an.defaults_held(shared)

@linux_only
def test_install_can_bake_take_defaults_and_apps_into_the_unit(tmp_path: Path) -> None:
    r = an.install("/opt/hil/bin/crosspad-hil", home=tmp_path, systemctl=lambda a: None,
                   take_defaults=True, apps=["FL Studio", "Ableton Live"])
    text = Path(r["unit"]).read_text()
    assert "audio nodes run --take-defaults --apps 'FL Studio' 'Ableton Live'" in text


# --------------------------------------------------------------------------- install

@linux_only
def test_install_writes_unit_and_rule_and_enables(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    r = an.install("/opt/hil/bin/crosspad-hil", home=tmp_path, systemctl=calls.append)
    unit = Path(r["unit"])
    assert unit == tmp_path / ".config/systemd/user" / an.UNIT_NAME
    assert "ExecStart=/opt/hil/bin/crosspad-hil audio nodes run" in unit.read_text()
    rule = Path(r["wireplumber_rule"]).read_text()
    assert "alsa_card.usb-Espressif_Crosspad_Audio*" in rule and "device.disabled" in rule
    assert calls == [["daemon-reload"], ["enable", "--now", an.UNIT_NAME]]

    calls.clear()
    an.uninstall(home=tmp_path, systemctl=calls.append)
    assert not unit.exists()
    assert calls[0] == ["disable", "--now", an.UNIT_NAME]
