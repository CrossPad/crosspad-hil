"""A published release, seen from the bench board: catalog, FW_CHECK over CDC
and SysEx, FW_UPD_STATUS / FW_LATEST over SysEx; with --install, the release's
own image in the other slot, then back.

Phase A (always): crosspad.app lists <tag> for the board's rev and kit; FW_CHECK
over CDC; the board's FW_UPD_STATUS and FW_LATEST (SysEx) name the release. A
failure while the board runs an older image is that image's fault: exit 2
(bench), e.g. rc2's stack overflow on CDC FW_CHECK.
Phase B (--install, or when the board already runs <tag>): FW_DOWNLOAD INSTALL,
FW_SWITCH into it; FW_CHECK over CDC and over SysEx each answer OK within 2 s,
reach up_to_date, no reset, CDC loop never blocked; FW_LATEST update=0;
APP_VERSIONS equals the release's components.json; FW_SWITCH back.

The board must be in the CDC profile (CDC + USB MIDI). Exit codes: 0 pass,
1 firmware fault, 2 bench problem. Spec: platform-idf
docs/superpowers/specs/2026-10-10-release-automation-design.md, "release_check".
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import fw_catalog, verbs
from crosspad_hil.devices import discover, select
from crosspad_hil.errors import CANCELLED, NO_DEVICE, TIMEOUT, HilError
from crosspad_hil.midi import MidiIO, MidiRole, companion_command
from crosspad_hil.scenarios.base import Context, Progress, Report, register
from crosspad_hil.scenarios.fw_switch import (
    REBOOT_SETTLE_S,
    REOPEN_RETRY_S,
    REOPEN_TIMEOUT_S,
    FwSwitchScenario,
)

POLL_S = 1.0
OK_WITHIN_S = 2.0
CDC_BLOCK_LIMIT_US = 50_000
BUSY = ("checking", "downloading", "installing")
# The reply each companion verb is answered with. Another listener on the same port (the hub's
# FW_UPD_STATUS / FW_LATEST requests) gets its reply there too; companion_command takes the first
# complete one, so a reply of the wrong shape is dropped and the read goes on. "ERR busy" is
# accepted for the two query verbs: the board refusing ours while a request of the hub's is in
# flight is a reason to ask again (_companion), not to wait for a reply that is never coming.
EXPECT = {
    "FW_UPD_STATUS": ("FWUPD:", "ERR busy"),
    "FW_LATEST": ("FW_LATEST:", "ERR busy"),
    "FW_CHECK": ("OK", "ERR"),
}
# private: gh (logged in on the bench, as fw_update needs) fetches it
RELEASE_REPO = "CrossPad/platform-idf"


@dataclass
class Params:
    tag: str = field(default="", metadata={"help": "release tag, e.g. v1.1.0-rc3 (required)"})
    install: bool = field(default=False, metadata={
        "help": "install the release into the other slot, check it there, switch back"})
    check_timeout: float = field(default=60.0, metadata={"help": "seconds for one catalog check"})
    catalog_timeout: float = field(default=900.0, metadata={
        "help": "seconds to wait for crosspad.app"})
    install_timeout: float = field(default=600.0, metadata={
        "help": "seconds for download + install"})
    confirm_timeout: float = field(default=30.0, metadata={
        "help": "seconds for a switched slot to confirm"})
    restore_timeout: float = field(default=240.0, metadata={
        "help": "seconds the recovery may take to bring the board back to its start slot"})


class _Stop(Exception):
    def __init__(self, summary: str, exit_code: int = 1) -> None:
        super().__init__(summary)
        self.summary, self.exit_code = summary, exit_code


def judge_status(f: dict, version: str, want_beta: bool, running_is_tag: bool) -> list[str]:
    problems = []
    # The board's own check names an update only when it is newer than the running image:
    # on the release itself there is none ("-").
    if f.get("latest") != version and not (running_is_tag and f.get("latest") in ("-", "")):
        problems.append(f"FW_UPD_STATUS latest={f.get('latest')}, want {version}")
    if want_beta and f.get("channel") != "beta":
        problems.append(f"FW_UPD_STATUS channel={f.get('channel')}, want beta")
    if f.get("err") not in (None, "", "-"):
        problems.append(f"FW_UPD_STATUS err={f.get('err')}")
    return problems


def judge_latest(f: dict, version: str, running_is_tag: bool) -> list[str]:
    want = "0" if running_is_tag else "1"
    problems = []
    if f.get("latest") != version and not (running_is_tag and f.get("latest") in ("-", "")):
        problems.append(f"FW_LATEST latest={f.get('latest')}, want {version}")
    if f.get("update") != want:
        problems.append(f"FW_LATEST update={f.get('update')}, want {want}")
    return problems


def components_problems(appver: list[dict], release: dict) -> list[str]:
    """APP_VERSIONS and the release's components.json name the same components at the same
    commits: a missing, an extra or a differing one is a problem, and so is a release file
    that is not shaped like components.json."""
    comps = release.get("components") if isinstance(release, dict) else None
    if not isinstance(comps, dict) or not comps:
        return ["release components.json has no components"]
    problems = []
    want_by_name: dict[str, str] = {}
    for key, value in comps.items():
        if not isinstance(key, str) or "/" not in key or not isinstance(value, str) or not value:
            problems.append(f"release components.json: malformed entry {key!r}: {value!r}")
            continue
        want_by_name[key.rsplit("/", 1)[1]] = value
    if not isinstance(appver, list):
        return problems + ["APP_VERSIONS: no component rows"]
    running: dict[str, str] = {}
    for c in appver:
        name = c.get("component") if isinstance(c, dict) else None
        if not isinstance(name, str) or not name:
            problems.append(f"APP_VERSIONS: row without a component name: {c!r}")
            continue
        running[name] = c.get("commit") or ""
    for name, got in running.items():
        want = want_by_name.get(name)
        if want is None:
            problems.append(f"{name}: running {got or '-'}, not in the release")
        elif not (len(got) >= 7 and want.startswith(got[:12])):
            problems.append(f"{name}: running {got or '-'}, release {want[:12]}")
    for name, want in want_by_name.items():
        if name not in running:
            problems.append(f"{name}: release {want[:12]}, missing from APP_VERSIONS")
    return problems


def catalog_lists(catalog: dict, kit: str, tag: str) -> bool:
    return any(k.get("kit") == kit and any(v.get("tag") == tag for v in k.get("versions", []))
               for k in catalog.get("kits", []))


class ReleaseCheckScenario:
    name = "release_check"
    Params = Params
    description = ("a published release from the bench board: catalog, FW_CHECK over CDC "
                   "and SysEx, versions")

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        self.ctx, self.p = ctx, params
        self.data: dict[str, Any] = {"tag": params.tag, "steps": {}}
        ctx.partial = self.data
        try:
            report = self._run(ctx, params, progress)
        except Exception as exc:  # noqa: BLE001 - a wrong image outranks whatever escaped the body
            if self.data.get("back_on_start_slot") is not False:
                raise              # the board is home (or was never moved): the usual mapping
            if isinstance(exc, HilError):
                text = f"{exc.code}: {exc.message}"
                self.data["error"] = exc.to_dict()
                self.data["cancelled"] = exc.code == CANCELLED
            else:
                text = f"crashed: {exc!r}"
                self.data["error"] = {"message": repr(exc)}
                self.data["cancelled"] = False
            report = Report(passed=False, summary=f"release_check: {text}", data=self.data,
                            artifacts=[], exit_code=2)
        if self.data.get("back_on_start_slot") is False:
            # Whatever the body did, a board that is not back on its start image is the bench's
            # problem now, and the first thing to say.
            note = (f"BENCH LEFT ON THE WRONG IMAGE: start slot {self.data.get('start_slot')} "
                    f"is not confirmed ({self.data.get('recovery_error', 'unknown')})")
            report = Report(passed=False, summary=f"{report.summary} | {note}", data=report.data,
                            artifacts=report.artifacts, exit_code=2)
        return report

    def _run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        try:
            if not params.tag:
                raise _Stop("release_check: --tag is required", 2)
            self.version = params.tag.lstrip("v")
            self.link = ctx.open_cdc()
            self.io = MidiIO(ctx.device, MidiRole.ESP)
            self.io.open()
            st = verbs.fw_upd_status(self.link)
            kit, running = st["kit"], st["running"]
            rev = verbs.app_versions(self.link)["board"].get("fw")
            if not rev:
                raise _Stop("release_check: APP_VERSIONS has no BOARD row", 2)
            self.data.update(kit=kit, rev=rev, running=running)
            progress(1, 4, "catalog")
            self._wait_catalog(rev, kit)
            want_beta = "-" in params.tag
            if want_beta and st.get("channel") != "beta":
                raise _Stop(f"release_check: kit {kit} is on channel {st.get('channel')}; "
                            f"FW_UPD_CHANNEL {kit} beta first", 2)
            running_is_tag = running.lstrip("v") == self.version
            progress(2, 4, "phase A: the running image sees the release")
            try:
                self._phase_a(want_beta, running_is_tag)
            except _Stop as e:
                if running_is_tag:
                    raise
                raise _Stop(f"bench board runs {running}, which fails: {e.summary}; the bench "
                            f"image must be older than {params.tag} (or {params.tag})", 2) from e
            if params.install or running_is_tag:
                progress(3, 4, "phase B: the release's own image")
                self._phase_b(kit, running_is_tag)
            progress(4, 4, "done")
            return Report(passed=True, summary=f"release_check {params.tag}: pass", data=self.data,
                          artifacts=[], exit_code=0)
        except _Stop as e:
            return Report(passed=False, summary=e.summary, data=self.data, artifacts=[],
                          exit_code=e.exit_code)
        finally:
            io = getattr(self, "io", None)
            if io is not None:
                io.close()

    # -- helpers -----------------------------------------------------------

    def _companion(self, verb: str) -> str:
        for _ in range(3):
            text = companion_command(self.io, verb, timeout_s=OK_WITHIN_S + 3.0,
                                     accept=EXPECT.get(verb))
            if not text.startswith("ERR busy"):
                return text
            time.sleep(1.0)        # another listener's request (the hub) is in flight
        return text

    def _wait_catalog(self, rev: str, kit: str) -> None:
        deadline = time.monotonic() + self.p.catalog_timeout
        while True:
            try:
                if catalog_lists(fw_catalog.fetch(rev), kit, self.p.tag):
                    return
            except (OSError, ValueError) as e:
                raise _Stop(f"release_check: crosspad.app unreachable: {e}", 2) from e
            if time.monotonic() >= deadline:
                raise _Stop(f"release_check: crosspad.app does not list {self.p.tag} "
                            f"for {rev}/{kit}", 1)
            self._pause(15.0)

    def _pause(self, seconds: float) -> None:
        """Wait, and stop at once (CANCELLED) when the run is cancelled."""
        self.ctx.cancelled.wait(seconds)
        self.ctx.check_cancelled()

    def _settle(self, read) -> dict:
        deadline = time.monotonic() + self.p.check_timeout
        while True:
            f = read()
            if f.get("state") not in BUSY:
                return f
            if time.monotonic() >= deadline:
                raise _Stop(f"release_check: still {f.get('state')} after "
                            f"{self.p.check_timeout:.0f}s")
            self._pause(POLL_S)

    def _sysex_status(self) -> dict:
        return fw_catalog.reply_fields(self._companion("FW_UPD_STATUS"))

    def _release_components(self) -> dict:
        name = f"CrossPad-{self.p.tag}-components.json"
        try:
            out = subprocess.run(
                ["gh", "release", "download", self.p.tag, "-R", RELEASE_REPO, "-p", name,
                 "-O", "-"],
                check=True, capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError) as e:
            raise _Stop(f"release_check: cannot fetch {name}: {e}", 2) from e
        try:
            release = json.loads(out)
        except ValueError as e:
            raise _Stop(f"release_check: {name} is not JSON: {e}", 2) from e
        if not isinstance(release, dict):
            raise _Stop(f"release_check: {name} is not a JSON object", 2)
        return release

    def _phase_a(self, want_beta: bool, running_is_tag: bool) -> None:
        try:
            verbs.fw_check(self.link)
        except HilError as exc:
            raise _Stop(f"release_check: FW_CHECK over cdc: {exc.message}") from exc
        status = self._settle(self._sysex_status)
        latest = fw_catalog.reply_fields(self._companion("FW_LATEST"))
        self.data["steps"]["a"] = {"status": status, "latest": latest}
        problems = judge_status(status, self.version, want_beta, running_is_tag) + \
            judge_latest(latest, self.version, running_is_tag)
        if problems:
            raise _Stop("release_check: " + "; ".join(problems))

    def _start_check(self, path: str, start) -> tuple[Any, float]:
        """(answer, seconds the accepted attempt took). FW_CHECK answering "ERR busy" -- a
        background check, or a request of the hub's, holds the board -- is no firmware fault: wait
        while FW_UPD_STATUS says it is busy and ask again until check_timeout."""
        deadline = time.monotonic() + self.p.check_timeout
        while True:
            t0 = time.monotonic()
            try:
                answer = start()
            except HilError as exc:
                if "ERR busy" not in exc.message:
                    raise _Stop(f"release_check: FW_CHECK over {path}: {exc.message}") from exc
                answer = "ERR busy"
            if not (isinstance(answer, str) and answer.startswith("ERR busy")):
                return answer, time.monotonic() - t0
            if time.monotonic() >= deadline:
                raise _Stop(f"release_check: FW_CHECK over {path}: still ERR busy after "
                            f"{self.p.check_timeout:.0f}s")
            self._pause(POLL_S)
            self._settle(lambda: verbs.fw_upd_status(self.link))     # waits out a running check

    def _fw_check_ok(self, path: str, start) -> None:
        resets = verbs.reset_count(self.link)
        verbs.cdc_stats(self.link)                      # read-and-clear maxcmd
        answer, took = self._start_check(path, start)
        if took > OK_WITHIN_S or (isinstance(answer, str) and not answer.startswith("OK")):
            raise _Stop(f"release_check: FW_CHECK over {path}: {answer!r} after {took:.1f}s")
        f = self._settle(lambda: verbs.fw_upd_status(self.link))
        maxcmd = verbs.cdc_stats(self.link)["maxcmd_us"]
        self.data["steps"][f"fw_check_{path}"] = {
            "answer": answer, "s": round(took, 2), "status": f, "maxcmd_us": maxcmd}
        if f.get("state") != "up_to_date":
            raise _Stop(f"release_check: FW_CHECK over {path} ended {f.get('state')} "
                        f"err={f.get('err')}")
        if verbs.reset_count(self.link) != resets:
            raise _Stop(f"release_check: the board reset during FW_CHECK over {path}")
        if maxcmd > CDC_BLOCK_LIMIT_US:
            raise _Stop(f"release_check: FW_CHECK over {path}: the CDC loop blocked {maxcmd} us")

    def _phase_b(self, kit: str, running_is_tag: bool) -> None:
        start_slot = None
        if not running_is_tag:
            slots = verbs.fw_slots(self.link)["slots"]
            start_slot = next(s for s in slots if s["running"])["slot"]
            self.data["start_slot"] = start_slot
            verbs.fw_download(self.link, kit, self.version, install=True)
            deadline = time.monotonic() + self.p.install_timeout
            while (st := verbs.fw_upd_status(self.link))["state"] in BUSY:
                if time.monotonic() >= deadline:
                    raise _Stop("release_check: install did not finish")
                self._pause(POLL_S)
            self.ctx.check_cancelled()      # a cancel that landed on the last poll: no FW_SWITCH
            if st["state"] != "ready_to_restart":
                raise _Stop(f"release_check: install ended {st['state']} err={st.get('err')}")
        try:
            if start_slot is not None:
                other = "B" if start_slot == "A" else "A"
                verbs.fw_switch(self.link)
                self.link = FwSwitchScenario._reopen(self.ctx, self.link)
                if FwSwitchScenario._wait_confirmed(
                        self.ctx, self.link, other, self.p.confirm_timeout, self.data) is None:
                    raise _Stop(f"release_check: slot {other} never confirmed")
                self.io.close()
                self.io = MidiIO(self.ctx.device, MidiRole.ESP)
                self.io.open()
            self._fw_check_ok("cdc", lambda: (verbs.fw_check(self.link), "OK")[1])
            self._fw_check_ok("sysex", lambda: self._companion("FW_CHECK").strip())
            latest = fw_catalog.reply_fields(self._companion("FW_LATEST"))
            problems = judge_latest(latest, self.version, running_is_tag=True)
            if problems:
                raise _Stop("release_check: " + "; ".join(problems))
            release = self._release_components()
            problems = components_problems(verbs.app_versions(self.link)["components"], release)
            if problems:
                raise _Stop("release_check: running components differ: " + "; ".join(problems))
        finally:
            if start_slot is not None:
                self._restore(start_slot)

    # -- recovery: the board goes back to the slot it started on, whatever happened ----------

    def _restore(self, start_slot: str) -> None:
        """Never raises and ignores cancellation (a cancelled run must still leave the bench on
        its own image). Records back_on_start_slot / recovery_error in the report data."""
        deadline = time.monotonic() + self.p.restore_timeout
        switches = 0
        error = ""
        try:
            self.link = self._reopen_quiet(self.link, deadline)
            while True:
                if time.monotonic() >= deadline:
                    raise _Stop(f"recovery timed out after {self.p.restore_timeout:.0f}s")
                try:
                    slots = verbs.fw_slots(self.link)["slots"]
                    running = next((s for s in slots if s.get("running")), None)
                except HilError:
                    self.link = self._reopen_quiet(self.link, deadline)    # still rebooting
                    continue
                if running is None:
                    time.sleep(POLL_S)
                    continue
                state = running.get("state")
                if running["slot"] == start_slot:
                    if state == "confirmed":
                        break
                elif state in ("confirmed", "rolledback"):
                    if switches >= 2:
                        raise _Stop("the board keeps booting the other slot")
                    switches += 1
                    self.data.setdefault("recovery", []).append(
                        f"FW_SWITCH from {running['slot']} ({state})")
                    try:
                        verbs.fw_switch(self.link)
                    except HilError as exc:
                        self.data["recovery"].append(f"FW_SWITCH refused: {exc.message}")
                    self.link = self._reopen_quiet(self.link, deadline)
                    continue
                time.sleep(POLL_S)       # a trial slot: it confirms, or the bootloader rolls back
        except Exception as exc:  # noqa: BLE001 - the recovery reports, it never raises
            error = getattr(exc, "message", None) or getattr(exc, "summary", None) or repr(exc)
        self.data["back_on_start_slot"] = not error
        if error:
            self.data["recovery_error"] = error

    def _reopen_quiet(self, link, deadline: float):
        """Like FwSwitchScenario._reopen, without its cancellation checks; bounded by `deadline`
        and REOPEN_TIMEOUT_S."""
        try:
            link.close()
        except Exception:  # noqa: BLE001 - the old link may already be dead
            pass
        until = min(deadline, time.monotonic() + REOPEN_TIMEOUT_S)
        while True:
            try:
                new_link = self.ctx.open_cdc()
                new_link.wait_ready()
                break
            except HilError as exc:
                if exc.code not in (NO_DEVICE, TIMEOUT) or time.monotonic() >= until:
                    raise
                time.sleep(REOPEN_RETRY_S)
                try:
                    self.ctx.device = select(discover(), self.ctx.device.id)
                except HilError:
                    pass
        time.sleep(REBOOT_SETTLE_S)
        return new_link


register(ReleaseCheckScenario())
