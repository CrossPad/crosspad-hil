"""`_audio.ensure_kits`: provision the card instead of dying on "no kit named".

Against a fake card, with the push and the reboot stubbed out. What is being
tested is the *decision* logic and the order of operations, all of which the
scenario depends on and none of which needs hardware:

* a card that already has the kits is left alone — no build, no push, no reboot;
* a card missing one gets both directories made, every file pushed, and a reboot,
  and only then is the kit list believed again;
* the link handed in is closed and a fresh one handed back, because pushing needs
  the raw port and the reboot re-enumerates it;
* every way it can go wrong is exit 2, because none of them is evidence about
  the thing the scenario measures.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from crosspad_hil import fixtures, kits
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import ENV, FLASH_FAILED, HilError
from crosspad_hil.scenarios import _audio, base
from tests.fakes import FakeSerial
from tests.test_scenario_kit_churn import make_device


class CardDevice(FakeSerial):
    """A CDC endpoint with a kit list, an FS_MKDIR that behaves like the device's.

    Kits pushed while it is running do not appear in `KIT_LIST` until `reboot()`
    — the kit scan runs once, after the mount, which is the whole reason
    `ensure_kits` reboots at all.
    """

    def __init__(self, kits: list[str]) -> None:
        super().__init__()
        self.kits = list(kits)
        self.staged: list[str] = []       # kits pushed, not yet scanned
        self.mkdirs: list[str] = []
        self.mkdir_error: str | None = None
        self.scanning = 0                 # KIT_LIST answers "ERR scanning" this many times
        self.scanning_after_boot = 0      # …and this many more once it has rebooted

    def reboot(self) -> None:
        self.kits += self.staged
        self.staged = []
        self.scanning = self.scanning_after_boot
        self.scanning_after_boot = 0

    def write(self, data: bytes) -> int:
        for raw in data.decode("utf-8", "replace").split("\n"):
            cmd = raw.strip()
            if not cmd:
                continue
            self.written.append(cmd)
            if cmd == "KIT_LIST":
                if self.scanning > 0:
                    self.scanning -= 1
                    self.feed(["ERR scanning"])
                    continue
                body = ",".join(f"{i}:{n}" for i, n in enumerate(self.kits))
                self.feed([f"KITS: {body}",
                           f"KITS: end count={len(self.kits)} current=0"])
            elif cmd.startswith("FS_MKDIR "):
                path = cmd.split(" ", 1)[1]
                self.mkdirs.append(path)
                self.feed([self.mkdir_error] if self.mkdir_error else ["OK"])
            elif cmd == "APP_LIST":
                self.feed(["APPS: Sampler running=Sampler"])
            else:
                self._rx.append((cmd + "\r\n").encode())
        return len(data)


class FakeConsole:
    """`Console.reset()` + `wait_boot()` + `forget_boot_history()`, and nothing else."""

    def __init__(self, dev: CardDevice, *, missing: list[str] | None = None) -> None:
        self._dev = dev
        self.missing = missing or []
        self.resets = 0
        self.forgot = 0
        self.seq = 0

    def reset(self) -> None:
        self.resets += 1
        self._dev.reboot()

    def wait_boot(self, timeout_s: float = 0.0, settle_s: float = 0.0) -> Any:
        return type("Boot", (), {"missing": list(self.missing), "seconds": 0.1})()

    def forget_boot_history(self) -> None:
        self.forgot += 1

    def close(self) -> None:
        return None


@dataclass
class FakeContext(base.Context):
    cdc_fake: CardDevice = None  # type: ignore[assignment]
    links: list[CdcLink] = None  # type: ignore[assignment]

    def open_cdc(self) -> CdcLink:
        link = CdcLink("/dev/ttyACM0", serial_factory=lambda path, **kw: self.cdc_fake)
        link.open()
        self.opened.append(link)
        self.links.append(link)
        return link


def make_ctx(tmp_path: Path, kits: list[str]) -> tuple[FakeContext, CardDevice, FakeConsole]:
    dev = CardDevice(kits)
    ctx = FakeContext(device=make_device(), workdir=tmp_path, cancelled=threading.Event(),
                      log=lambda s: None, cdc_fake=dev, links=[])
    return ctx, dev, FakeConsole(dev)


class Pusher:
    """Stands in for `fs.push`; records what was sent where."""

    def __init__(self, dev: CardDevice, *, fail: str | None = None) -> None:
        self._dev = dev
        self._fail = fail
        self.sent: list[tuple[str, str]] = []

    def __call__(self, device: Any, local: Path, remote: str, **kw: Any) -> dict:
        if self._fail is not None:
            raise HilError(FLASH_FAILED, self._fail)
        self.sent.append((Path(local).name, remote))
        if Path(local).name == "kit.json":
            import json
            self._dev.staged.append(json.loads(Path(local).read_text())["name"])
        return {"bytes": Path(local).stat().st_size, "remote": remote}


NAMES = ["HIL SINE A", "HIL SINE B"]


def run(ctx: FakeContext, link: CdcLink, con: FakeConsole, **kw: Any):
    return _audio.ensure_kits(ctx, link, NAMES, fixtures.write_sine_kits,
                              console=con, **kw)


# --------------------------------------------------------------------------- present


def test_a_card_that_already_has_them_is_left_alone(tmp_path: Path) -> None:
    ctx, dev, con = make_ctx(tmp_path, ["Basic", "HIL SINE A", "hil sine b"])
    link = ctx.open_cdc()
    out_link, provisioned = run(ctx, link, con)
    assert provisioned == []
    assert out_link is link                    # same link: nothing was closed
    assert con.resets == 0 and dev.mkdirs == []
    assert not (tmp_path / "fixtures").exists()
    assert not any(c.startswith("FS_MKDIR") for c in dev.written)


def test_the_name_match_is_case_insensitive(tmp_path: Path) -> None:
    ctx, dev, con = make_ctx(tmp_path, ["hil SINE a", "HIL sine B"])
    link = ctx.open_cdc()
    assert run(ctx, link, con)[1] == []


# --------------------------------------------------------------------------- provision


def test_a_missing_kit_is_built_pushed_and_rebooted_into_the_list(tmp_path: Path) -> None:
    ctx, dev, con = make_ctx(tmp_path, ["Basic"])
    push = Pusher(dev)
    link = ctx.open_cdc()
    out_link, provisioned = run(ctx, link, con, push=push)

    assert provisioned == NAMES
    assert dev.kits == ["Basic", *NAMES]
    # both directories and both SAMPLES dirs, before anything was pushed
    assert dev.mkdirs == ["/sdcard/crosspad/kits/HIL_SINE_A",
                          "/sdcard/crosspad/kits/HIL_SINE_A/SAMPLES",
                          "/sdcard/crosspad/kits/HIL_SINE_B",
                          "/sdcard/crosspad/kits/HIL_SINE_B/SAMPLES"]
    # 16 samples + kit.json per kit, kit.json last so an interrupted push leaves
    # an incomplete directory rather than a kit missing its samples
    assert len(push.sent) == 34
    assert push.sent[16] == ("kit.json", "/sdcard/crosspad/kits/HIL_SINE_A/kit.json")
    assert push.sent[0][1].startswith("/sdcard/crosspad/kits/HIL_SINE_A/SAMPLES/")
    # one reboot, and the boot it caused is not left for a later fault check
    assert con.resets == 1 and con.forgot == 1
    # the link handed in was closed and a fresh one handed back
    assert out_link is not link
    assert link._ser is None
    assert out_link is ctx.links[-1]


def test_only_the_missing_kit_is_pushed(tmp_path: Path) -> None:
    ctx, dev, con = make_ctx(tmp_path, ["HIL SINE A"])
    push = Pusher(dev)
    _link, provisioned = run(ctx, ctx.open_cdc(), con, push=push)
    assert provisioned == ["HIL SINE B"]
    assert len(push.sent) == 17
    assert all("HIL_SINE_B" in remote for _name, remote in push.sent)


def test_an_existing_directory_is_not_an_error(tmp_path: Path) -> None:
    """The device answers FS_ERR mkdir_17 (EEXIST) for a directory it already has;
    a card provisioned once and wiped of kit.json only must still be pushable."""
    ctx, dev, con = make_ctx(tmp_path, [])
    dev.mkdir_error = "FS_ERR mkdir_17"
    _link, provisioned = run(ctx, ctx.open_cdc(), con, push=Pusher(dev))
    assert provisioned == NAMES


def test_the_scan_is_waited_out_before_the_list_is_believed(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """KIT_LIST answers "ERR scanning" while listKits() walks the card; asking
    once after the banner would read a card that has not found the new kits yet."""
    monkeypatch.setattr(kits, "SCAN_POLL_S", 0.0)
    ctx, dev, con = make_ctx(tmp_path, [])
    dev.scanning_after_boot = 3
    _link, provisioned = run(ctx, ctx.open_cdc(), con, push=Pusher(dev))
    assert provisioned == NAMES
    assert dev.scanning == 0


# --------------------------------------------------------------------------- refusals


def test_no_provision_keeps_the_old_env_failure(tmp_path: Path) -> None:
    ctx, dev, con = make_ctx(tmp_path, ["Basic"])
    with pytest.raises(HilError) as ei:
        run(ctx, ctx.open_cdc(), con, provision=False)
    assert ei.value.code == ENV
    assert "HIL SINE A" in ei.value.message
    assert "Basic" in (ei.value.hint or "")
    assert con.resets == 0


def test_a_refused_push_is_a_bench_problem(tmp_path: Path) -> None:
    ctx, dev, con = make_ctx(tmp_path, [])
    with pytest.raises(HilError) as ei:
        run(ctx, ctx.open_cdc(), con, push=Pusher(dev, fail="no space left"))
    assert ei.value.code == ENV
    assert "no space left" in ei.value.message
    assert con.resets == 0


def test_a_board_that_does_not_come_back_is_a_bench_problem(tmp_path: Path) -> None:
    ctx, dev, con = make_ctx(tmp_path, [])
    con.missing = ["BOOT_MARKER"]
    with pytest.raises(HilError) as ei:
        run(ctx, ctx.open_cdc(), con, push=Pusher(dev))
    assert ei.value.code == ENV
    assert "BOOT_MARKER" in (ei.value.hint or "")


def test_a_scan_that_never_ends_is_a_bench_problem(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kits, "SCAN_POLL_S", 0.0)
    ctx, dev, con = make_ctx(tmp_path, [])
    dev.scanning_after_boot = 99
    monkeypatch.setattr(kits, "SCAN_TIMEOUT_S", 0.0)   # only after the entry check
    with pytest.raises(HilError) as ei:
        run(ctx, ctx.open_cdc(), con, push=Pusher(dev))
    assert ei.value.code == ENV
    assert "kit scan never finished" in ei.value.message


def test_kits_that_do_not_come_back_after_the_reboot_are_a_bench_problem(
    tmp_path: Path,
) -> None:
    """The push succeeded and the board booted, but the scan did not take them —
    a full card, or JSON the parser rejected. Neither is a firmware verdict."""
    class SwallowingPusher(Pusher):
        def __call__(self, device, local, remote, **kw):     # noqa: ANN001
            self.sent.append((Path(local).name, remote))
            return {"bytes": 1, "remote": remote}

    ctx, dev, con = make_ctx(tmp_path, [])
    with pytest.raises(HilError) as ei:
        run(ctx, ctx.open_cdc(), con, push=SwallowingPusher(dev))
    assert ei.value.code == ENV
    assert "still not in KIT_LIST" in ei.value.message


def test_a_builder_that_does_not_know_the_name_is_a_bench_problem(tmp_path: Path) -> None:
    ctx, dev, con = make_ctx(tmp_path, [])
    with pytest.raises(HilError) as ei:
        _audio.ensure_kits(ctx, ctx.open_cdc(), ["NOT A FIXTURE"],
                           fixtures.write_sine_kits, console=con)
    assert ei.value.code == ENV
    assert "does not know" in ei.value.message
