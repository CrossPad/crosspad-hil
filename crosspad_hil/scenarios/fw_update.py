"""Firmware updates end to end against a bench copy of crosspad.app.

Serves real CI-signed packages of the board's kit and revision from a GitHub
release (crosspad_hil.fw_server; files cached under ~/.cache/crosspad-hil) on
this machine's LAN address, points the board at it with FW_UPD_URL, and checks
platform-idf's updater (docs/subsystems/firmware-updates.md), in a normal boot
unless said otherwise:

  tls        FW_CHECK against the real crosspad.app (a 404 still runs the whole
             handshake): the job task's unused stack, stack_free=, is recorded
  catalog    FW_CHECK on a WiFi lease; a second FW_CHECK while one runs is ERR busy;
             FW_CATALOG lists the release's version of this kit
  resume     CrossPad.bin is cut half way once; the board resumes with Range and verifies
  signature  the real files under a made-up entry with a broken manifest.sig:
             refused, deleted, blocked until reboot
  mismatch   the real files under another kit's entry: refused, deleted
  no_range   cut again, and the resume answered with the whole file: the board
             starts that file over instead of appending to it
  cleanup    every other package pinned, a lowered limit (FW_UPD_CFG gc_mb=): only
             the unpinned victim is evicted; gc_mb=1 refuses with card_full
  install    FW_DOWNLOAD ... INSTALL -> ready_to_restart (switch=True: FW_SWITCH
             into it, it confirms, FW_SWITCH back -- ask the bench's owner first)
  sha        a flipped byte in CrossPad.bin: one fresh retry, then refused; no slot changes
  fallback   NET_MODE 1 TAB=updates: the tab checks when it opens, the download
             works there too; back in a normal boot (boot=1) the boot check runs on its own

Needs a saved WiFi network (crosspad-hil run wifi, or CROSSPAD_WIFI_SSID /
CROSSPAD_WIFI_PASS in the environment to save it here) on the LAN this machine is
on -- the bench router is 10.42.0.1, so pass host_ip=10.42.0.1 -- gh logged in,
and a release carrying kit packages. Ends in a normal boot, in the app and kit it
started in, with FW_UPD_URL on DEFAULT, the kit's channel as it was and Check at
startup on.

Exit codes: 0 pass, 1 firmware fault, 2 bench problem.
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import fs, fw_server, verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import TIMEOUT, HilError
from crosspad_hil.scenarios.base import Context, Progress, Report, register
from crosspad_hil.scenarios.fw_switch import FwSwitchScenario
from crosspad_hil.scenarios.wifi import _live_link, _to_launcher
from crosspad_hil.scenarios.wifi import _reopen as reopen_in_mode

LIBRARY = "/sdcard/crosspad/firmware"
POLL_S = 1.0
MB = 1024 * 1024
GC_VICTIM = "zz-gc-victim"
GC_VICTIM_BYTES = 3 * MB             # more than the 1 MB rounding of gc_mb
# the device reserves manifest + signature room on top of the files
MANIFEST_MARGIN = 64 * 1024
CARD_MIN_FREE = 10                   # percent, the device's own floor
BOOT_CHECK_WAIT_S = 90.0             # launcher paint + link + one catalog fetch
BUSY_STATES = ("checking", "downloading", "installing")
STEPS = 11
# how long the server sits on the catalog answer while the second check is sent
BUSY_HOLD_S = 5.0
_DESCRIBE_RE = re.compile(r"-\d+-g[0-9a-f]+(-dirty)?$")   # git describe after the tag
# below this a longer URL or TLS record overflows a task's stack
STACK_MARGIN_B = 1024
MEMORY_REFUSAL = "no_mem"             # no_mem, no_mem:catalog, no_mem:job_task, no_mem:install_task


@dataclass
class Params:
    ssid_env: str = field(default="CROSSPAD_WIFI_SSID",
                          metadata={"help": "environment variable holding the bench SSID"})
    pass_env: str = field(default="CROSSPAD_WIFI_PASS",
                          metadata={"help": "environment variable holding the bench password"})
    tag: str | None = field(default=None, metadata={
        "help": "release to serve (default: the newest with kit packages)"})
    host_ip: str | None = field(default=None, metadata={
        "help": "this machine's address as the board sees it (the bench router: 10.42.0.1)"})
    check_timeout: float = field(default=60.0, metadata={"help": "seconds for one catalog check"})
    download_timeout: float = field(default=600.0, metadata={"help": "seconds for one download"})
    install_timeout: float = field(default=300.0,
                                   metadata={"help": "seconds for a download plus install"})
    confirm_timeout: float = field(default=30.0,
                                   metadata={"help": "seconds for a switched slot to confirm"})
    switch: bool = field(default=False, metadata={
        "help": "also FW_SWITCH into the installed image and back "
                "(a shared bench: ask its owner first)"})


class _Stop(Exception):
    def __init__(self, summary: str, exit_code: int = 1) -> None:
        super().__init__(summary)
        self.summary = summary
        self.exit_code = exit_code


def library_bytes(link: Any) -> int:
    """What the device's cleanup counts: package directories' files and loose *.bin."""
    total = 0
    for e in verbs.fs_list(link, LIBRARY)["entries"]:
        if e["dir"]:
            total += sum(f["size"] for f in verbs.fs_list(link, f"{LIBRARY}/{e['name']}")["entries"]
                         if not f["dir"])
        elif e["name"].lower().endswith(".bin"):
            total += e["size"]
    return total


def expected_check(running: str, release: str) -> str | None:
    """What the device's check says of this running version against this release: nothing newer
    -> up_to_date, an older running -> available. None when a git-describe suffix on the running
    version leaves it between the two."""
    running = running.removeprefix("v")
    base = _DESCRIBE_RE.sub("", running)
    a, b = fw_server.version_key(base), fw_server.version_key(release.removeprefix("v"))
    if a < b:
        return "available"
    if a == b and base != running:
        return None
    return "up_to_date"


def _refused(fn: Any, *args: Any) -> tuple[bool, str]:
    try:
        fn(*args)
    except HilError as exc:
        return True, exc.message
    return False, "answered OK"


class FwUpdateScenario:
    name = "fw_update"
    Params = Params
    description = ("catalog, resumable verified downloads, cleanup, install, refusals, "
                   "network-mode fallback")

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        self.ctx, self.p = ctx, params
        self.data: dict[str, Any] = {"steps": {}}
        ctx.partial = self.data
        self.link: Any = None
        self.created_pins: list[str] = []
        self.original: dict[str, Any] = {}
        self.saved_ssid: str | None = None
        srv: fw_server.FwServer | None = None
        try:
            self.link = ctx.open_cdc()
            progress(0, STEPS, "preflight")
            if verbs.net_mode_status(self.link)["active"]:   # an aborted run left it there
                self._mode(False)
            self.original["wifi"] = verbs.wifi_status(self.link).get("enabled")
            if not verbs.wifi_list(self.link):
                ssid, password = (os.environ.get(params.ssid_env, ""),
                                  os.environ.get(params.pass_env, ""))
                if not ssid or not password:
                    raise _Stop("fw_update: no WiFi network saved on the board -- WIFI_SET first "
                                f"(crosspad-hil run wifi) or set {params.ssid_env} / "
                                f"{params.pass_env}", 2)
                verbs.wifi_set(self.link, ssid, password)
                self.saved_ssid = ssid
            verbs.wifi_enable(self.link, True)
            # A Sampler with a kit is refused the radio by design; that is not what this measures.
            try:
                self.original["app"] = _to_launcher(ctx, self.link)
            except HilError as exc:
                if exc.code != TIMEOUT:
                    raise
                raise _Stop(f"fw_update: {exc.message}", 2) from exc
            self.original["kit"] = verbs.kit_status(self.link)["current"]
            st = verbs.fw_upd_status(self.link)
            kit, running = st["kit"], st["running"]
            self.original["channel"] = st["channel"]
            rev = verbs.app_versions(self.link)["board"].get("fw")
            if not rev:
                raise _Stop("fw_update: APP_VERSIONS has no BOARD row -- firmware too old", 2)
            tag = params.tag or fw_server.newest_kit_release()
            real = fw_server.fetch_release(tag, rev, kit)
            bad_sig = fw_server.alias(real, "zzsig", "0.0.1", ctx.workdir / "zzsig", break_sig=True)
            other_kit = fw_server.alias(real, "zzmis", "0.0.2", ctx.workdir / "zzmis")
            self.data.update(kit=kit, rev=rev, tag=tag, running=running)

            progress(1, STEPS, "TLS stack")
            self._tls_stack()

            host = self._host_ip()
            try:
                srv = fw_server.FwServer([real, bad_sig, other_kit])
                srv.start()
            except OSError as exc:
                raise _Stop(f"fw_update: cannot serve the packages: {exc}", 2) from exc
            url = srv.url(host)
            self.data["url"] = url
            verbs.fw_upd_url(self.link, url)
            verbs.fw_upd_channel(self.link, kit, real.channel)
            pkg = f"{kit}-{real.version}"
            bin_name = real.files["bin"].name
            bin_size = real.files["bin"].stat().st_size
            incoming = bin_size + real.files["assets"].stat().st_size
            self._remove(pkg)

            progress(2, STEPS, "catalog")
            self._catalog(real, srv)

            progress(3, STEPS, "resume after a cut")
            srv.faults.reset()
            srv.faults.drop_once_at[bin_name] = bin_size // 2
            self._download_ok(kit, real.version, pkg, "resume")
            if not any(e["status"] == 206 and (e["range"] or "").startswith("bytes=")
                       for e in srv.requests_for(bin_name)):
                raise _Stop("fw_update: resume: no ranged request for CrossPad.bin after the cut")

            progress(4, STEPS, "bad signature, another kit's manifest")
            srv.faults.reset()
            self._refused_download("zzsig", bad_sig.version, "signature")
            refused, msg = _refused(verbs.fw_download, self.link, "zzsig", bad_sig.version)
            if not refused or "blocked" not in msg:
                raise _Stop(f"fw_update: signature: the second attempt answered {msg!r}, "
                            "not ERR blocked")
            self._refused_download("zzmis", other_kit.version, "mismatch")
            for gone in (f"zzsig-{bad_sig.version}", f"zzmis-{other_kit.version}"):
                if self._exists(f"{LIBRARY}/{gone}"):
                    raise _Stop(f"fw_update: the refused package {gone} is still on the card")

            progress(5, STEPS, "a server that ignores Range")
            self._remove(pkg)
            seen = len(srv.requests_for(bin_name))
            srv.faults.reset()
            srv.faults.drop_once_at[bin_name] = bin_size // 2
            srv.faults.ignore_range.add(bin_name)
            self._download_ok(kit, real.version, pkg, "no_range")
            whole = [e for e in srv.requests_for(bin_name)[seen:]
                     if e["status"] == 200 and e["sent"] == bin_size]
            if not whole:
                raise _Stop("fw_update: no_range: the board never fetched CrossPad.bin whole again")
            srv.faults.reset()

            progress(6, STEPS, "card cleanup")
            self._cleanup(kit, real, pkg, incoming)

            progress(7, STEPS, "install")
            self._install(kit, real, tag)

            progress(8, STEPS, "flipped byte")
            self._remove(pkg)
            slots = verbs.fw_slots(self.link)["slots"]
            seen = len(srv.requests_for(bin_name))
            srv.faults.reset()
            srv.faults.flip_at[bin_name] = bin_size // 3
            self._refused_download(kit, real.version, "sha")
            fresh = [e for e in srv.requests_for(bin_name)[seen:] if e["range"] is None]
            if len(fresh) != 2:
                raise _Stop(f"fw_update: sha: expected the file twice (one fresh retry), "
                            f"saw {len(fresh)}")
            if self._exists(f"{LIBRARY}/{pkg}/.complete"):
                raise _Stop("fw_update: sha: the refused package is marked complete")
            if verbs.fw_slots(self.link)["slots"] != slots:
                raise _Stop("fw_update: sha: a slot changed after a refused download")
            srv.faults.reset()

            progress(9, STEPS, "network-mode fallback")
            self._remove(pkg)
            self._fallback(kit, real, pkg)

            progress(10, STEPS, "boot check")
            self._boot_check()

            progress(STEPS, STEPS, "done")
            summary = (f"fw_update: PASS -- {kit} {real.version} from {tag} on {rev}, "
                       "all refusals held")
            ctx.log(f"[fw_update] {summary}")
            return Report(passed=True, summary=summary, data=self.data, artifacts=[], exit_code=0)
        except _Stop as stop:
            ctx.log(f"[fw_update] {stop.summary}")
            return Report(passed=False, summary=stop.summary, data=self.data, artifacts=[],
                          exit_code=stop.exit_code)
        finally:
            if isinstance(self.link, CdcLink):
                self._restore()
            if srv is not None:
                srv.stop()
            ctx.close_all()

    # -- modes and the card -------------------------------------------------

    def _step(self, what: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        """One restore step: a failure is logged and the next step still runs."""
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            self.ctx.log(f"[fw_update] restore: {what} failed: {exc}")

    def _restore(self) -> None:
        self.ctx.cancelled.clear()      # teardown: a cancelled run still puts the board back
        try:
            self.link = _live_link(self.ctx, self.link)
        except Exception as exc:
            self.ctx.log(f"[fw_update] restore: no link to the board: {exc}")
            return
        try:
            if verbs.net_mode_status(self.link)["active"]:
                self._mode(False)
        except Exception as exc:
            self.ctx.log(f"[fw_update] restore: could not leave network mode: {exc} -- "
                         "tools/bench.py ready leaves it")
            self._step("link", self._relink)
        self._step("FW_UPD_URL", verbs.fw_upd_url, self.link, None)
        if self.data.get("kit") and self.original.get("channel"):
            self._step("channel", verbs.fw_upd_channel, self.link, self.data["kit"],
                       self.original["channel"])
        self._step("FW_UPD_CFG", verbs.fw_upd_cfg, self.link, boot=True, gc_mb=0)
        self._step("pins and the victim", self._unpin)
        if self.saved_ssid:
            self._step("saved network", verbs.wifi_forget, self.link, self.saved_ssid)
        if self.original.get("wifi") is not None:
            self._step("WiFi enable", verbs.wifi_enable, self.link, self.original["wifi"] == 1)
        if self.original.get("kit") is not None:
            self._step("kit", self._restore_kit)
        if self.original.get("app"):
            self._step("app", verbs.app_start, self.link, self.original["app"])

    def _relink(self) -> None:
        self.link = _live_link(self.ctx, self.link)

    def _restore_kit(self) -> None:
        # A boot out of network mode loads no kit, so the one the run started with comes back here.
        if (verbs.kit_status(self.link)["current"] != self.original["kit"]
                and self.original["kit"] >= 0):
            verbs.kit_load(self.link, self.original["kit"])

    def _mode(self, on: bool, tab: str = "updates") -> None:
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
        path = f"{LIBRARY}/{name}"
        if not self._exists(path):
            return
        for e in verbs.fs_list(self.link, path)["entries"]:
            verbs.fs_delete(self.link, f"{path}/{e['name']}")
        verbs.fs_delete(self.link, path)

    def _push(self, pairs: list[tuple[Path, str]]) -> None:
        """fs.push opens the CDC port itself: close ours around it."""
        if not pairs:
            return
        self.link.close()
        for local, remote in pairs:
            fs.push(self.ctx.device, local, remote)
        self.link = self.ctx.open_cdc()
        self.link.wait_ready()

    def _unpin(self) -> None:
        for p in self.created_pins:
            try:
                verbs.fs_delete(self.link, p)
            except HilError:
                pass
        self.created_pins = []
        self._remove(GC_VICTIM)

    # -- updater steps --------------------------------------------------------

    def _settle(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            st = verbs.fw_upd_status(self.link)
            if st["state"] not in BUSY_STATES:
                return st
            if time.monotonic() >= deadline:
                raise _Stop(f"fw_update: still {st['state']} ({st.get('progress')} %) "
                            f"after {timeout:.0f}s")
            self.ctx.cancelled.wait(POLL_S)
            self.ctx.check_cancelled()

    def _download_ok(self, kit: str, version: str, pkg: str, step: str) -> None:
        self._settle(self.p.check_timeout)
        verbs.fw_download(self.link, kit, version)
        st = self._settle(self.p.download_timeout)
        self.data["steps"][step] = st
        if st["state"] == "failed":
            raise _Stop(f"fw_update: {step}: download failed: {st.get('err')}")
        if not any(r["on_card"] for r in verbs.fw_catalog(self.link, kit)["versions"]
                   if r["version"] == version):
            raise _Stop(f"fw_update: {step}: {pkg} is not complete on the card")

    def _refused_download(self, kit: str, version: str, err: str) -> None:
        self._settle(self.p.check_timeout)
        verbs.fw_download(self.link, kit, version)
        st = self._settle(self.p.download_timeout)
        self.data["steps"][err] = st
        if st["state"] != "failed" or not str(st.get("err") or "").startswith(err):
            raise _Stop(f"fw_update: {err}: ended {st['state']} err={st.get('err')}, "
                        f"expected a refusal ({err})")

    def _start_check(self) -> None:
        try:
            verbs.fw_check(self.link)
        except HilError as exc:
            if MEMORY_REFUSAL in exc.message:
                raise _Stop(f"fw_update: FW_CHECK refused for memory: {exc.message}") from exc
            raise

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
            raise _Stop(f"fw_update: no local address to serve from: {exc}", 2) from exc

    def _tls_stack(self) -> None:
        """One handshake with the real crosspad.app: how much of the job task's stack it leaves."""
        verbs.fw_upd_url(self.link, None)
        self._settle(self.p.check_timeout)
        self._start_check()
        st = self._settle(self.p.check_timeout)
        self.data["steps"]["tls"] = st
        self.data["stack_free"] = st["stack_free"]
        if st["state"] == "failed" and str(st.get("err") or "").startswith("catalog:network"):
            raise _Stop("fw_update: tls: crosspad.app is unreachable from the bench, "
                        "stack not measured", 2)
        if st["stack_free"] < STACK_MARGIN_B:
            raise _Stop(f"fw_update: tls: {st['stack_free']} B of the job stack left "
                        f"over a real handshake, under {STACK_MARGIN_B}")

    def _catalog(self, real: fw_server.ServedVersion, srv: fw_server.FwServer) -> None:
        self._settle(self.p.check_timeout)          # a boot check may still be running
        # The server holds the catalog answer, so the first check is still running whatever the
        # radio state when the second verb arrives.
        srv.faults.catalog_delay_s = BUSY_HOLD_S
        try:
            self._start_check()
            busy, msg = _refused(verbs.fw_check, self.link)
            st = self._settle(self.p.check_timeout)
        finally:
            srv.faults.reset()
        self.data["steps"]["catalog"] = st
        if not busy or "busy" not in msg:
            raise _Stop(f"fw_update: a second FW_CHECK while one ran answered {msg!r}, "
                        "not ERR busy")
        want = expected_check(st["running"], real.version)
        if st["state"] not in ("up_to_date", "available"):
            raise _Stop(f"fw_update: FW_CHECK ended {st['state']} err={st.get('err')}")
        if want is not None and st["state"] != want:
            raise _Stop(f"fw_update: running {st['running']} against release {real.version} "
                        f"should end {want}, FW_CHECK ended {st['state']}")
        # up_to_date reports no newer release (latest=-); available names it.
        latest = real.version if st["state"] == "available" else None
        if st["latest"] != latest:
            raise _Stop(f"fw_update: FW_CHECK ended {st['state']} with latest={st['latest']}, "
                        f"expected {latest}")
        rows = verbs.fw_catalog(self.link, real.kit)["versions"]
        if not any(r["version"] == real.version for r in rows):
            raise _Stop(f"fw_update: FW_CATALOG does not list {real.kit} {real.version}: {rows}")

    def _cleanup(self, kit: str, real: fw_server.ServedVersion, pkg: str, incoming: int) -> None:
        roots = verbs.fs_roots(self.link)["roots"].get("/sdcard") or {}
        total, used = roots.get("total"), roots.get("used")
        if (total and used is not None
                and (total - used - incoming - GC_VICTIM_BYTES) * 100 < total * CARD_MIN_FREE):
            raise _Stop("fw_update: cleanup: the bench card has under 10 % free "
                        "beside one package", 2)
        self._remove(pkg)
        others = [e["name"] for e in verbs.fs_list(self.link, LIBRARY)["entries"] if e["dir"]]
        work = self.ctx.workdir / "fw_update_push"
        work.mkdir(parents=True, exist_ok=True)
        pin = work / "pin"
        pin.write_bytes(b"1")
        victim = work / "victim.bin"
        victim.write_bytes(os.urandom(GC_VICTIM_BYTES))
        self.created_pins = [f"{LIBRARY}/{d}/.pinned" for d in others
                             if not self._exists(f"{LIBRARY}/{d}/.pinned")]
        verbs.fs_mkdir(self.link, f"{LIBRARY}/{GC_VICTIM}")
        self._push([(pin, p) for p in self.created_pins]
                   + [(victim, f"{LIBRARY}/{GC_VICTIM}/CrossPad.bin")])
        lib = library_bytes(self.link)
        # Over the limit by less than the victim: exactly the victim has to go.
        limit_mb = math.ceil((lib + incoming + MANIFEST_MARGIN - GC_VICTIM_BYTES) / MB)
        self.data["cleanup"] = {"library": lib, "incoming": incoming, "limit_mb": limit_mb,
                                "pinned": list(self.created_pins)}
        verbs.fw_upd_cfg(self.link, gc_mb=limit_mb)
        self._download_ok(kit, real.version, pkg, "cleanup")
        if self._exists(f"{LIBRARY}/{GC_VICTIM}"):
            raise _Stop("fw_update: cleanup: the unpinned victim is still on the card")
        after = {e["name"] for e in verbs.fs_list(self.link, LIBRARY)["entries"] if e["dir"]}
        lost = [d for d in others if d not in after]
        if lost:
            raise _Stop(f"fw_update: cleanup deleted protected packages: {lost}")
        # The real package, off the card again: the device checks manifest, signature and kit
        # before it cleans up, so only a package that passes them reaches the cleanup that refuses.
        self._remove(pkg)
        before = {e["name"] for e in verbs.fs_list(self.link, LIBRARY)["entries"]}
        verbs.fw_upd_cfg(self.link, gc_mb=1)
        self._refused_download(kit, real.version, "card_full")
        if {e["name"] for e in verbs.fs_list(self.link, LIBRARY)["entries"]} != before:
            raise _Stop("fw_update: card_full: the library changed although the download "
                        "was refused")
        verbs.fw_upd_cfg(self.link, gc_mb=0)
        self._unpin()

    def _install(self, kit: str, real: fw_server.ServedVersion, tag: str) -> None:
        before = verbs.fw_slots(self.link)["slots"]
        running0 = next(s for s in before if s["running"])
        other = "B" if running0["slot"] == "A" else "A"
        self._settle(self.p.check_timeout)
        # whole on the card: verified, installed
        verbs.fw_download(self.link, kit, real.version, install=True)
        st = self._settle(self.p.install_timeout)
        self.data["steps"]["install"] = st
        if st["state"] != "ready_to_restart":
            raise _Stop(f"fw_update: install ended {st['state']} err={st.get('err')}")
        self.data["inst_stack_free"] = st["inst_stack_free"]
        if st["inst_stack_free"] < STACK_MARGIN_B:
            raise _Stop(f"fw_update: install: {st['inst_stack_free']} B of the install task's "
                        f"stack left, under {STACK_MARGIN_B}")
        slot = next(s for s in verbs.fw_slots(self.link)["slots"] if s["slot"] == other)
        if slot["state"] != "installed" or slot["pkg"] != f"{kit}-{real.version}":
            raise _Stop(f"fw_update: install: slot {other} reads {slot}")
        if not self.p.switch:
            return
        verbs.fw_switch(self.link)
        self.link = FwSwitchScenario._reopen(self.ctx, self.link)
        landed = FwSwitchScenario._wait_confirmed(self.ctx, self.link, other,
                                                  self.p.confirm_timeout, self.data)
        if landed is None:
            raise _Stop(f"fw_update: install: slot {other} never confirmed after the switch")
        if (landed.get("ver") or "").lstrip("v") != tag.lstrip("v"):
            raise _Stop(f"fw_update: install: slot {other} runs {landed.get('ver')}, not {tag}")
        verbs.fw_switch(self.link)
        self.link = FwSwitchScenario._reopen(self.ctx, self.link)
        if FwSwitchScenario._wait_confirmed(self.ctx, self.link, running0["slot"],
                                            self.p.confirm_timeout, self.data) is None:
            raise _Stop(f"fw_update: install: slot {running0['slot']} never confirmed "
                        "after switching back")

    def _fallback(self, kit: str, real: fw_server.ServedVersion, pkg: str) -> None:
        """The same download where a refused lease sends the user: network mode."""
        self._mode(True, tab="updates")
        # the Updates tab checked when it opened
        opened = self._settle(self.p.check_timeout)
        self.data["steps"]["fallback_check"] = opened
        if opened["state"] not in ("up_to_date", "available") or not opened.get("last_check"):
            raise _Stop(f"fw_update: network mode: opening the Updates tab did not check: {opened}")
        self._download_ok(kit, real.version, pkg, "fallback")
        verbs.fw_upd_cfg(self.link, boot=True)
        self.data["last_check_before_boot"] = verbs.fw_upd_status(self.link)["last_check"]
        self._mode(False)

    def _boot_check(self) -> None:
        """Back in a normal boot with Check at startup on: a check runs with nothing sent."""
        before = self.data["last_check_before_boot"]
        deadline = time.monotonic() + BOOT_CHECK_WAIT_S
        while True:
            st = verbs.fw_upd_status(self.link)
            if st["state"] in ("up_to_date", "available") and (st.get("last_check") or 0) > before:
                self.data["steps"]["boot_check"] = st
                return
            if time.monotonic() >= deadline:
                raise _Stop(f"fw_update: no boot check within {BOOT_CHECK_WAIT_S:.0f}s "
                            f"of the launcher: {st}")
            self.ctx.cancelled.wait(POLL_S)
            self.ctx.check_cancelled()


SCENARIO = register(FwUpdateScenario())
