"""The Firmware app end to end: Library, kit card, Store download, budget, churn.

Drives platform-idf's Firmware app (docs/subsystems/firmware-app.md) with the
encoder and reads FW_UI instead of screenshots:

  packages   30 test packages (manifest + a distinct icon each, no image) pushed
             to the card in a normal boot; the Library takes their kit and
             version from the manifest, so each tile has a kit@version key
  library    APP_START Firmware: the Library lists them (and the slot-only tiles
             of an image flashed over USB); PSRAM over the launcher <= budget; a
             24-tile encoder walk lands on a real tile
  card       the zzui00 tile's card: Pin (Delete from card is then not offered),
             Unpin, Delete from card (confirmed), a power click back to the tab;
             "offered" is what the encoder reaches, not ENC_GROUP, which also
             lists the card's hidden buttons
  store      the Store tab checks the bench catalog live; PSRAM of the tab with
             no card within the budget; the release's card on its own: opened
             within card_budget, closed back within card_tolerance; Download
             -> on the card
  run        (switch=True only) the other slot's package: Run -> the board goes
             away (Restart verifies the image first) and comes back -> it confirms ->
             FW_SWITCH back; a rolled-back slot offers "Try this slot again"
             and "Reinstall" -- ask the bench's owner first
  churn      enter/leave cycles: PSRAM per visit within leak_bytes
  network    NET_MODE 1 TAB=store: the same Store download in network mode

Needs a saved WiFi network on the LAN this machine is on (the bench router is
10.42.0.1, so pass host_ip=10.42.0.1), gh logged in, and a release carrying kit
packages WITH icons and kit text: that is plan C's release (v1.1.0-fwstore.4 or
later; earlier tags carry no icon and the run stops with exit 2). The default
tag is the newest release whose kit packages carry icons -- pass tag= to pin
one. Ends in a normal boot with the test packages removed, the bench catalog's
cache, icons and kit texts gone (the card's own .catalog.json and
.community_ok put back), FW_UPD_URL on DEFAULT, the kit's channel, WiFi, app
and kit as they were.

Exit codes: 0 pass, 1 firmware fault, 2 bench problem.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import audio_state, fs, fw_server, verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import TIMEOUT, HilError
from crosspad_hil.scenarios.app_churn import per_visit, slope
from crosspad_hil.scenarios.base import Context, Progress, Report, register
from crosspad_hil.scenarios.fw_switch import FwSwitchScenario
from crosspad_hil.scenarios.wifi import _live_link, _to_launcher
from crosspad_hil.scenarios.wifi import _reopen as reopen_in_mode

LIBRARY = "/sdcard/crosspad/firmware"
# What the bench catalog's checks leave on the card: the cache files are kept
# aside and put back, the icon and kit-text directories (derived from a
# catalog, fetched again by the next check) go.
CARD_CACHE_FILES = (".catalog.json", ".community_ok")
CARD_CACHE_LEFTOVERS = (".catalog.json.tmp", ".community_ok.tmp")
CARD_CACHE_DIRS = (".icons", ".details")
TEST_KIT = "zzui"
TEST_VERSION = "1.0.0"
ICON_SIDE = 48
LV_IMAGE_MAGIC = 0x19
LV_CF_RGB565A8 = 0x14
ICON_BYTES = 12 + ICON_SIDE * ICON_SIDE * 3
# CrossPad.bin is listed but never pushed: the size only has to be > 0
TEST_IMAGE_BYTES = 1
TEST_IMAGE_SHA = "0" * 64
# The WiFi radio included while Store/Updates hold the lease (spec 3, As built).
BUDGET_BYTES = 128 * 1024
# The kit card over its tab: chips and their launcher icons (bench 47432 B).
CARD_BUDGET_BYTES = 64 * 1024
CARD_TOLERANCE_BYTES = 4 * 1024      # what a closed card may keep (bench: -580 B)
OPAQUE = 0xFF
POLL_S = 0.5
SETTLE_S = 1.5
FREE_STABLE_BYTES = 4 * 1024   # two psram_free readings this close are one settled value
FREE_STABLE_TIMEOUT_S = 5.0    # the longest wait for them before the last reading counts
UI_TIMEOUT_S = 30.0
# Encoder clicks before a control counts as not there (30 tiles, a wrap, 3 tabs).
MAX_STEPS = 64
FOCUS_STEP_S = 1.0             # an ENC_ROTATE reaches the focus on the next indev read
FOCUS_POLL_S = 0.05
# The card's Restart verifies the image (~0.6 s on the bench), then resets.
RESTART_GONE_S = 10.0
BUSY_STATES = ("checking", "downloading", "installing")
MEMORY_REFUSAL = "no_mem"      # no_mem, no_mem:catalog, no_mem:job_task, no_mem:install_task, ...
PLAN_C_RELEASE = "v1.1.0-fwstore.4"
STEPS = 9


@dataclass
class Params:
    tag: str | None = field(default=None, metadata={
        "help": "release to serve (default: the newest whose kit packages carry icons, "
                f"{PLAN_C_RELEASE} or later)"})
    host_ip: str | None = field(default=None, metadata={
        "help": "this machine's address as the board sees it (the bench router: 10.42.0.1)"})
    packages: int = field(default=30, metadata={
        "help": "test packages on the card (the budget's 'entries')"})
    budget: int = field(default=BUDGET_BYTES, metadata={
        "help": "PSRAM the open app may cost over the launcher, the WiFi radio included, bytes "
                "(no kit card open)"})
    card_budget: int = field(default=CARD_BUDGET_BYTES, metadata={
        "help": "PSRAM an open kit card may cost over the tab it opens on, bytes"})
    card_tolerance: int = field(default=CARD_TOLERANCE_BYTES, metadata={
        "help": "PSRAM a closed kit card may keep, bytes"})
    rounds: int = field(default=20, metadata={"help": "enter/leave cycles"})
    leak_bytes: int = field(default=64, metadata={
        "help": "PSRAM loss per visit that counts as a leak"})
    download_timeout: float = field(default=600.0, metadata={"help": "seconds for one download"})
    confirm_timeout: float = field(default=30.0, metadata={
        "help": "seconds for a switched slot to confirm"})
    switch: bool = field(default=False, metadata={
        "help": "also Run the other slot's package and FW_SWITCH back "
                "(a shared bench: ask its owner first)"})


class _Stop(Exception):
    def __init__(self, summary: str, exit_code: int = 1) -> None:
        super().__init__(summary)
        self.summary = summary
        self.exit_code = exit_code


def icon_bin(rgb: tuple[int, int, int]) -> bytes:
    """A solid 48x48 LVGL RGB565A8 image -- what tools/fw_icon.py writes, without the PNG."""
    r, g, b = rgb
    pixel = struct.pack("<H", ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3))
    header = struct.pack("<BBHHHHH", LV_IMAGE_MAGIC, LV_CF_RGB565A8, 0, ICON_SIDE, ICON_SIDE,
                         ICON_SIDE * 2, 0)
    return header + pixel * (ICON_SIDE * ICON_SIDE) + bytes([OPAQUE]) * (ICON_SIDE * ICON_SIDE)


def make_test_packages(workdir: Path, count: int) -> list[tuple[Path, str]]:
    """(local file, card path) pairs: a manifest and a distinct icon per package.

    The manifest is a valid one (core's parseFwManifest: name, kit, version,
    board_rev and a CrossPad.bin entry), so the firmware names each package's
    kit and version from it although the image itself is never pushed."""
    out: list[tuple[Path, str]] = []
    workdir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        kit = f"{TEST_KIT}{i:02d}"
        name = f"{kit}-{TEST_VERSION}"
        icon = workdir / f"{name}.icon.bin"
        icon_data = icon_bin(((i * 37) % 256, (i * 91) % 256, (i * 53) % 256))
        icon.write_bytes(icon_data)
        files = {"CrossPad.bin": {"size": TEST_IMAGE_BYTES, "sha256": TEST_IMAGE_SHA},
                 "icon.bin": {"size": len(icon_data),
                              "sha256": hashlib.sha256(icon_data).hexdigest()}}
        manifest = workdir / f"{name}.manifest.json"
        manifest.write_text(json.dumps({"name": name, "kit": kit, "version": TEST_VERSION,
                                        "board_rev": 2, "display_name": f"Test {i:02d}",
                                        "files": files}))
        out += [(manifest, f"{LIBRARY}/{name}/manifest.json"), (icon, f"{LIBRARY}/{name}/icon.bin")]
    return out


def loss_per_visit(series: list[int]) -> float:
    """app_churn's head/tail windows over free-PSRAM samples, one per visit."""
    s = slope(series)
    if s is None:
        return 0.0
    _, _, visits, drop = s
    return per_visit(drop, visits)


class FirmwareAppScenario:
    name = "firmware_app"
    Params = Params
    description = ("Firmware app: Library, kit card, Store download (normal and network mode), "
                   "budget, churn")
    audio_snap: dict[str, Any] | None = None     # the mixer/codec state the run started with

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        self.ctx, self.p = ctx, params
        self.data: dict[str, Any] = {"steps": {}}
        ctx.partial = self.data
        self.link: Any = None
        self.original: dict[str, Any] = {}
        self.kit = ""
        self.audio_snap: dict[str, Any] | None = None
        srv: fw_server.FwServer | None = None
        try:
            self.link = ctx.open_cdc()
            self.audio_snap = audio_state.snapshot(self.link)
            progress(0, STEPS, "preflight")
            if verbs.net_mode_status(self.link)["active"]:   # an aborted run left it there
                self._mode(False)
            if not verbs.wifi_list(self.link):
                raise _Stop("firmware_app: no WiFi network saved on the board -- WIFI_SET first "
                            "(crosspad-hil run wifi)", 2)
            self.original["wifi"] = verbs.wifi_status(self.link).get("enabled")
            verbs.wifi_enable(self.link, True)
            # A Sampler with a kit is refused the radio by design; that is not what this measures.
            try:
                self.original["app"] = _to_launcher(ctx, self.link)
            except HilError as exc:
                if exc.code != TIMEOUT:
                    raise
                raise _Stop(f"firmware_app: {exc.message}", 2) from exc
            self.original["kit"] = verbs.kit_status(self.link)["current"]
            st = verbs.fw_upd_status(self.link)
            self.kit = st["kit"]
            self.original["channel"] = st["channel"]
            rev = verbs.app_versions(self.link)["board"].get("fw")
            if not rev:
                raise _Stop("firmware_app: APP_VERSIONS has no BOARD row -- firmware too old", 2)
            tag = params.tag or fw_server.newest_kit_release(icons=True)
            real = fw_server.fetch_release(tag, rev, self.kit)
            if "icon" not in real.files:
                raise _Stop(f"firmware_app: release {tag} has no kit icon -- tag one built by the "
                            f"icon CI ({PLAN_C_RELEASE} or later) and pass tag=", 2)
            try:
                srv = fw_server.FwServer([real])
                srv.start()
            except OSError as exc:
                raise _Stop(f"firmware_app: cannot serve the packages: {exc}", 2) from exc
            url = srv.url(self._host_ip())
            verbs.fw_upd_url(self.link, url)
            verbs.fw_upd_channel(self.link, self.kit, real.channel)
            pkg = f"{self.kit}-{real.version}"
            if any(s.get("pkg") == pkg for s in verbs.fw_slots(self.link)["slots"]):
                raise _Stop(f"firmware_app: {pkg} is installed in a slot -- the download step "
                            "needs it off the card", 2)
            self.data.update(kit=self.kit, tag=tag, rev=rev, package=pkg, url=url)
            self._remove(pkg)
            self._snapshot_card_cache()

            progress(1, STEPS, "test packages")
            self._push_test_packages()

            progress(2, STEPS, "library and budget")
            self.ctx.cancelled.wait(SETTLE_S)
            base_free = verbs.mem(self.link)["psram_free"]
            verbs.app_start(self.link, "Firmware")
            self._wait_ui(lambda u: u["tab"] == "library" and (u["items"] or 0) >= params.packages
                          and not u["busy"], "the Library listing the test packages")
            self._budget("library", base_free)
            verbs.enc_rotate(self.link, 16)
            verbs.enc_rotate(self.link, 8)
            ui = self._wait_ui(lambda u: not u["busy"], "the walk to settle")
            if ui["sel"] is None or not ui["selkey"]:
                raise _Stop(f"firmware_app: after a 24-tile walk no tile is selected: {ui}")
            self._budget("library_walked", base_free)

            progress(3, STEPS, "kit card: pin, unpin, delete")
            victim = f"{TEST_KIT}00-{TEST_VERSION}"
            self._open_card(f"{TEST_KIT}00@{TEST_VERSION}")
            self._press("Pin")
            self._wait(lambda: self._exists(f"{LIBRARY}/{victim}/.pinned"), "Pin to write .pinned")
            if self._has_control("Delete from card"):
                raise _Stop("firmware_app: a pinned package's card still offers Delete from card")
            self._press("Unpin")
            self._wait(lambda: not self._exists(f"{LIBRARY}/{victim}/.pinned"),
                       "Unpin to remove .pinned")
            self._press("Delete from card")
            self._press("Delete")                                    # the confirm dialog's button
            self._wait(lambda: not self._exists(f"{LIBRARY}/{victim}"),
                       "Delete from card to remove the package")
            self._close_card()

            progress(4, STEPS, "store download (normal mode)")
            self._store_download(real, pkg, "store", base_free=base_free)
            self._close_card()
            self._remove(pkg)

            progress(5, STEPS, "run the other slot")
            if params.switch:
                self._run_other_slot()
            else:
                self.data["steps"]["run"] = "skipped (switch=False)"

            progress(6, STEPS, "enter/leave")
            self._churn()

            progress(7, STEPS, "store download (network mode)")
            verbs.app_stop(self.link)
            self._mode(True, tab="store")
            self._store_download(real, pkg, "store_net", host="net")
            self._close_card()
            self._mode(False)
            self._remove(pkg)

            progress(STEPS, STEPS, "done")
            summary = ("firmware_app: PASS -- Library, card, Store download in both modes, "
                       f"{params.rounds} cycles")
            ctx.log(f"[firmware_app] {summary}")
            return Report(passed=True, summary=summary, data=self.data, artifacts=[], exit_code=0)
        except _Stop as stop:
            ctx.log(f"[firmware_app] {stop.summary}")
            return Report(passed=False, summary=stop.summary, data=self.data, artifacts=[],
                          exit_code=stop.exit_code)
        finally:
            if isinstance(self.link, CdcLink):
                self._restore()
            if srv is not None:
                srv.stop()
            ctx.close_all()

    # -- restore ---------------------------------------------------------------

    def _step(self, what: str, fn: Callable[..., Any], *args: Any) -> None:
        """One restore step: a failure is logged and the next step still runs."""
        try:
            fn(*args)
        except Exception as exc:
            self.ctx.log(f"[firmware_app] restore: {what} failed: {exc}")

    def _restore(self) -> None:
        self.ctx.cancelled.clear()      # teardown: a cancelled run still puts the board back
        try:
            self.link = _live_link(self.ctx, self.link)
        except Exception as exc:
            self.ctx.log(f"[firmware_app] restore: no link to the board: {exc}")
            return
        try:
            if verbs.net_mode_status(self.link)["active"]:
                self._mode(False)
        except Exception as exc:
            self.ctx.log(f"[firmware_app] restore: could not leave network mode: {exc} -- "
                         "tools/bench.py ready leaves it")
            self._step("link", self._relink)
        self._step("app", verbs.app_stop, self.link)
        self._step("test packages", self._remove_test_packages)
        if self.original.get("card_cache") is not None:
            self._step("the bench catalog's cache", self._restore_card_cache)
        if self.data.get("package"):
            self._step("the downloaded package", self._remove, self.data["package"])
        self._step("FW_UPD_URL", verbs.fw_upd_url, self.link, None)
        if self.kit and self.original.get("channel"):
            self._step("channel", verbs.fw_upd_channel, self.link, self.kit,
                       self.original["channel"])
        if self.original.get("wifi") is not None:
            self._step("WiFi enable", verbs.wifi_enable, self.link, self.original["wifi"] == 1)
        if self.original.get("kit") is not None:
            self._step("kit", self._restore_kit)
        if self.original.get("app"):
            self._step("app", verbs.app_start, self.link, self.original["app"])
        # Last: the encoder walk through the Library, the cards and the Store can land on a
        # volume or mixer control, and that state outlives the run.
        audio_state.put_back(self.link, self.audio_snap, self.ctx.log, self.data)

    def _relink(self) -> None:
        self.link = _live_link(self.ctx, self.link)

    def _restore_kit(self) -> None:
        # A boot out of network mode loads no kit, so the one the run started with comes back here.
        if (verbs.kit_status(self.link)["current"] != self.original["kit"]
                and self.original["kit"] >= 0):
            verbs.kit_load(self.link, self.original["kit"])

    # -- the UI ---------------------------------------------------------------

    def _wait(self, pred: Callable[[], bool], what: str, timeout: float = UI_TIMEOUT_S) -> None:
        deadline = time.monotonic() + timeout
        while not pred():
            if time.monotonic() >= deadline:
                raise _Stop(f"firmware_app: timed out waiting for {what}")
            self.ctx.cancelled.wait(POLL_S)
            self.ctx.check_cancelled()

    def _wait_ui(self, pred: Callable[[dict[str, Any]], bool], what: str,
                 timeout: float = UI_TIMEOUT_S) -> dict[str, Any]:
        holder: dict[str, Any] = {}

        def ok() -> bool:
            holder["ui"] = verbs.fw_ui(self.link)
            return pred(holder["ui"])

        try:
            self._wait(ok, what, timeout)
        except _Stop as stop:
            raise _Stop(f"{stop.summary} (FW_UI {holder.get('ui')})") from None
        return holder["ui"]

    def _has_control(self, label: str) -> bool:
        return any(label in reached for reached in self._reachable())

    def _reachable(self) -> list[str]:
        """The labels one turn of the encoder visits, from the focus back to it.

        ENC_GROUP is no answer: it lists hidden members too, and the kit card
        keeps every action button in its group and hides the ones it does not
        offer. A group that does not wrap ends the turn at its last member."""
        start = verbs.enc_focus(self.link)
        labels = [start["label"]]
        prev = start["ptr"]
        for _ in range(MAX_STEPS):
            verbs.enc_rotate(self.link, 1)
            now = self._focus_after(prev)
            if now["ptr"] in (start["ptr"], prev):
                break
            labels.append(now["label"])
            prev = now["ptr"]
        self.ctx.log(f"[firmware_app] the encoder reaches {labels}")
        return labels

    def _focus_after(self, prev: str) -> dict[str, Any]:
        """ENC_FOCUS once the focus has left `prev`, or as it is after FOCUS_STEP_S."""
        deadline = time.monotonic() + FOCUS_STEP_S
        while True:
            now = verbs.enc_focus(self.link)
            if now["ptr"] != prev or time.monotonic() >= deadline:
                return now
            self.ctx.cancelled.wait(FOCUS_POLL_S)

    def _focus(self, label: str) -> None:
        """Turn the encoder until the focused control's text contains `label`."""
        for _ in range(MAX_STEPS):
            if label in str(verbs.enc_focus(self.link).get("label", "")):
                return
            verbs.enc_rotate(self.link, 1)
        raise _Stop(f"firmware_app: no control named {label!r} within {MAX_STEPS} encoder steps")

    def _press(self, label: str) -> None:
        self._focus(label)
        verbs.enc_press(self.link)
        self.ctx.cancelled.wait(SETTLE_S)

    def _open_card(self, key: str) -> None:
        """Walk the grid of the shown tab to the tile `key` (kit@version) and press it."""
        for _ in range(MAX_STEPS):
            if verbs.fw_ui(self.link)["selkey"] == key:
                verbs.enc_press(self.link)
                self._wait_ui(lambda u: u["card"] == key, f"the card of {key}")
                return
            verbs.enc_rotate(self.link, 1)
        raise _Stop(f"firmware_app: tile {key} not reached within {MAX_STEPS} encoder steps")

    def _close_card(self) -> None:
        verbs.pwr_gesture(self.link)
        self._wait_ui(lambda u: u["card"] is None, "a power click to close the card")

    def _stable_free(self) -> int:
        """psram_free once two readings POLL_S apart agree within FREE_STABLE_BYTES, or the last one
        after FREE_STABLE_TIMEOUT_S: a job's buffers go back to the heap a moment after busy=0."""
        deadline = time.monotonic() + FREE_STABLE_TIMEOUT_S
        last = int(verbs.mem(self.link)["psram_free"])
        while time.monotonic() < deadline:
            self.ctx.cancelled.wait(POLL_S)
            self.ctx.check_cancelled()
            now = int(verbs.mem(self.link)["psram_free"])
            if abs(now - last) <= FREE_STABLE_BYTES:
                return now
            last = now
        return last

    def _budget(self, step: str, base_free: int) -> None:
        used = base_free - self._stable_free()
        self.data["steps"][f"budget_{step}"] = used
        if used > self.p.budget:
            raise _Stop(f"firmware_app: {step}: the app costs {used} B of PSRAM over the launcher, "
                        f"over the {self.p.budget} B budget")

    def _settled_free(self) -> int:
        self._wait_ui(lambda u: not u["busy"], "the views to settle")
        self.ctx.cancelled.wait(SETTLE_S)
        return int(verbs.mem(self.link)["psram_free"])

    def _card_cost(self, key: str, step: str) -> None:
        """The kit card is not in the app's budget: open, it costs at most card_budget over its tab,
        and closed it gives that back to within card_tolerance."""
        before = self._settled_free()
        self._open_card(key)
        opened = before - self._settled_free()
        self._close_card()
        kept = before - self._settled_free()
        self.data["steps"][f"card_{step}"] = {"opened": opened, "kept_after_close": kept}
        if opened > self.p.card_budget:
            raise _Stop(f"firmware_app: {step}: the open kit card costs {opened} B of PSRAM over "
                        f"its tab, over the {self.p.card_budget} B card budget")
        if kept > self.p.card_tolerance:
            raise _Stop(f"firmware_app: {step}: the closed kit card kept {kept} B of PSRAM, "
                        f"over the {self.p.card_tolerance} B tolerance")

    # -- steps ----------------------------------------------------------------

    def _store_download(self, real: fw_server.ServedVersion, pkg: str, step: str, host: str = "app",
                        base_free: int | None = None) -> None:
        if host == "app":
            self._press("Store")                                     # the tab button
        self._wait_ui(lambda u: u["host"] == host and u["tab"] == "store" and u["store"] == "ok"
                      and (u["items"] or 0) >= 1 and not u["busy"],
                      f"{step}: the Store with the bench catalog", timeout=UI_TIMEOUT_S * 2)
        if base_free is not None:
            self._budget(step, base_free)                      # the tab, the radio up, no card
        key = f"{real.kit}@{real.version}"
        self._card_cost(key, step)
        self._open_card(key)
        self._press("Download")
        deadline = time.monotonic() + self.p.download_timeout
        st = verbs.fw_upd_status(self.link)
        while st["state"] in BUSY_STATES:
            if time.monotonic() >= deadline:
                raise _Stop(f"firmware_app: {step}: still {st['state']} after "
                            f"{self.p.download_timeout:.0f}s")
            self.ctx.cancelled.wait(POLL_S)
            self.ctx.check_cancelled()
            st = verbs.fw_upd_status(self.link)
        self.data["steps"][step] = st
        if str(st.get("err") or "").startswith(MEMORY_REFUSAL):
            raise _Stop(f"firmware_app: {step}: Download refused for memory: {st.get('err')}")
        rows = verbs.fw_catalog(self.link, real.kit)["versions"]
        if not any(r["version"] == real.version and r["on_card"] for r in rows):
            raise _Stop(f"firmware_app: {step}: {pkg} is not complete on the card after Download "
                        f"({st})")

    def _run_other_slot(self) -> None:
        slots = verbs.fw_slots(self.link)["slots"]
        running = next(s for s in slots if s["running"])
        other = next(s for s in slots if not s["running"])
        if not other.get("pkg") or not other.get("kit"):
            self.data["steps"]["run"] = "skipped (the other slot holds no named package)"
            return
        self._press("Library")
        self._wait_ui(lambda u: u["tab"] == "library" and not u["busy"], "the Library")
        self._open_card(f"{other['kit']}@{str(other.get('ver') or '').lstrip('v')}")
        if other["state"] == "rolledback":
            # A rolled-back image may be broken: writing it fresh is offered beside trying it again.
            for label in ("Try this slot again", "Reinstall"):
                if not self._has_control(label):
                    raise _Stop("firmware_app: run: the rolled-back slot's card does not offer "
                                f"{label!r}")
            self._press("Try this slot again")
            self._focus("Try again")
        else:
            self._press("Run")
            self._focus("Restart")
        verbs.enc_press(self.link)
        self._wait_gone()
        self.link = FwSwitchScenario._reopen(self.ctx, self.link)
        if FwSwitchScenario._wait_confirmed(self.ctx, self.link, other["slot"],
                                            self.p.confirm_timeout, self.data) is None:
            raise _Stop(f"firmware_app: run: slot {other['slot']} never confirmed after Run")
        verbs.fw_switch(self.link)
        self.link = FwSwitchScenario._reopen(self.ctx, self.link)
        if FwSwitchScenario._wait_confirmed(self.ctx, self.link, running["slot"],
                                            self.p.confirm_timeout, self.data) is None:
            raise _Stop(f"firmware_app: run: slot {running['slot']} never confirmed after "
                        "switching back")
        self.data["steps"]["run"] = f"{running['slot']}->{other['slot']}->{running['slot']}"

    def _wait_gone(self) -> None:
        """The card's Restart verifies the image before it resets: a reopen inside that window talks
        to the boot that is about to go away and then never hears another byte. Wait until the
        port's node is gone or the link stops answering; a reset board never answers on it again."""
        port = getattr(self.link, "port", None)
        deadline = time.monotonic() + RESTART_GONE_S
        while time.monotonic() < deadline:
            if port and not os.path.exists(port):
                return
            try:
                verbs.fw_slots(self.link)
            except HilError:
                return
            self.ctx.cancelled.wait(POLL_S)
            self.ctx.check_cancelled()
        raise _Stop(f"firmware_app: run: the board still answers {RESTART_GONE_S:.0f}s after "
                    "Restart")

    def _churn(self) -> None:
        verbs.app_stop(self.link)
        series: list[int] = []
        for _ in range(self.p.rounds):
            verbs.app_start(self.link, "Firmware")
            verbs.app_stop(self.link)
            self.ctx.cancelled.wait(SETTLE_S)
            self.ctx.check_cancelled()
            series.append(verbs.mem(self.link)["psram_free"])
        loss = loss_per_visit(series)
        self.data["steps"]["churn"] = {"psram_free": series, "loss_per_visit": loss}
        if loss > self.p.leak_bytes:
            raise _Stop(f"firmware_app: churn: {loss:.0f} B of PSRAM lost per visit over "
                        f"{self.p.rounds} visits")
        if verbs.fw_ui(self.link)["host"] is not None:
            raise _Stop("firmware_app: churn: FW_UI still names a host with the app closed")

    # -- modes and the card -----------------------------------------------------

    def _host_ip(self) -> str:
        """Where the board dials this machine: the local address that routes to its WiFi ip."""
        if self.p.host_ip:
            return self.p.host_ip
        board = verbs.wifi_status(self.link).get("ip")
        try:
            if board and board != "0.0.0.0":
                return fw_server.local_ip_for(board)
            return fw_server.lan_ip()
        except OSError as exc:
            raise _Stop(f"firmware_app: no local address to serve from: {exc}", 2) from exc

    def _mode(self, on: bool, tab: str = "wifi") -> None:
        result = verbs.net_mode_set(self.link, on, tab=tab if on else None)
        if result["reboot"]:
            self.link = reopen_in_mode(self.ctx, self.link, on)

    def _exists(self, path: str) -> bool:
        try:
            verbs.fs_stat(self.link, path)
            return True
        except HilError:
            return False

    def _remove(self, name: str) -> None:
        """A package directory. Normal mode only (file verbs)."""
        self._delete(f"{LIBRARY}/{name}")

    def _delete(self, path: str) -> None:
        """A file, or a directory of files. Normal mode only."""
        try:
            st = verbs.fs_stat(self.link, path)
        except HilError:
            return
        if st["dir"]:
            for e in verbs.fs_list(self.link, path)["entries"]:
                verbs.fs_delete(self.link, f"{path}/{e['name']}")
        verbs.fs_delete(self.link, path)

    def _card_cache_dir(self) -> Path:
        return self.ctx.workdir / "firmware_app" / "card_cache"

    def _snapshot_card_cache(self) -> None:
        """Before the first check against the bench server: the card's own cache files, kept
        aside."""
        present = [n for n in CARD_CACHE_FILES if self._exists(f"{LIBRARY}/{n}")]
        if present:
            snap = self._card_cache_dir()
            snap.mkdir(parents=True, exist_ok=True)
            self.link.close()                                        # fs.pull opens the port itself
            try:
                for n in present:
                    fs.pull(self.ctx.device, f"{LIBRARY}/{n}", snap / n)
            finally:
                self.link = self.ctx.open_cdc()
                self.link.wait_ready()
        self.original["card_cache"] = present

    def _restore_card_cache(self) -> None:
        for n in (*CARD_CACHE_DIRS, *CARD_CACHE_FILES, *CARD_CACHE_LEFTOVERS):
            self._delete(f"{LIBRARY}/{n}")
        kept = self.original["card_cache"]
        if not kept:
            return
        self.link.close()                                            # fs.push opens the port itself
        try:
            for n in kept:
                fs.push(self.ctx.device, self._card_cache_dir() / n, f"{LIBRARY}/{n}")
        finally:
            self.link = self.ctx.open_cdc()
            self.link.wait_ready()

    def _push_test_packages(self) -> None:
        files = make_test_packages(self.ctx.workdir / "firmware_app", self.p.packages)
        for d in [LIBRARY, *sorted({remote.rsplit("/", 1)[0] for _, remote in files})]:
            if not self._exists(d):
                verbs.fs_mkdir(self.link, d)
        self.link.close()                                            # fs.push opens the port itself
        for local, remote in files:
            fs.push(self.ctx.device, local, remote)
        self.link = self.ctx.open_cdc()
        self.link.wait_ready()

    def _remove_test_packages(self) -> None:
        for i in range(self.p.packages):
            self._remove(f"{TEST_KIT}{i:02d}-{TEST_VERSION}")


SCENARIO = register(FirmwareAppScenario())
