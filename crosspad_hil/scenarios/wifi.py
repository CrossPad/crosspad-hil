"""WiFi on demand, end to end: join, reach crosspad.app, fail cleanly, give the memory back and
hand BLE over and back, all in a normal boot; then one pass through network mode.

Spec: platform-idf docs/superpowers/specs/2026-10-02-wifi-on-demand-design.md §7, as amended by
Revision 2 -- WiFi is on demand in a normal boot; network mode (``NET_MODE 1`` / ``NET_MODE 0``,
both reboot the board) is the fallback when internal RAM is short.

What it checks, in order:

First the board is put in the launcher (``APP_STOP``, then ``APP_LIST`` until no app runs; the app
it started in goes to ``data["start_app"]``): a Sampler with a kit -- where ``kit_churn`` leaves the
bench -- has too little PSRAM for the radio and is refused by design, which is not what this
scenario measures.

Normal boot (BLE stopped -- runtime only -- for 1-3):
1. ``WIFI_SET`` with the bench network -> ``state=up`` within ``up_timeout`` -> ``WIFI_PING`` ok;
2. a wrong password -> ``failed`` / ``auth`` with ``RESET_LOG`` unchanged; the right one restored
   (``WIFI_SET`` overwrites the saved password);
3. memory: ``cycles`` x (``WIFI_UP`` -> up, ``WIFI_DOWN``, wait out the linger) -> free internal
   RAM within ``leak_tolerance`` bytes of the same point before the cycles;
4. BLE handover: with BLE running, ``WIFI_UP`` stops it while the link is up and ``WIFI_DOWN`` plus
   the linger bring it back. ``BLE_ENABLE 1`` first: the handover back needs it, and it is the
   bench's normal state.
Network mode:
5. the screen's lease connects without a ``WIFI_UP`` (``leases=1``), ``WIFI_PING`` ok, BLE stays off
   (``BLE_STATUS running=0``, ``BLE_START`` refused).

Whatever happens, the ``finally`` puts the board back, each step on its own so one failure does not
skip the rest: it leaves network mode, saves the bench network again with the right password (the
wrong-password check overwrites it), restores ``wireless.enableWiFi`` to what the board had at the
start, and restores BLE to what it was doing at the start -- enabled and started if it was running,
else disabled (no verb reads ``enableBLE``; running is the evidence there is). Exit 2 when the
credentials are not in the environment
(``set -a; . ~/.config/crosspad/bench_wifi.env; set +a``).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.devices import discover, select
from crosspad_hil.errors import NO_DEVICE, TIMEOUT, HilError
from crosspad_hil.scenarios.base import Context, Progress, Report, register

POLL_S = 0.5
BLE_TIMEOUT_S = 15.0           # BLE_ENABLE / the handover back bring NimBLE up in a few seconds
# A reboot into or out of network mode: the port comes back once app_main()
# serves CDC; network mode skips the kit scan, so this is shorter than an A/B
# switch's reboot (fw_switch.REOPEN_TIMEOUT_S) but the same shape.
REOPEN_TIMEOUT_S = 60.0
REOPEN_RETRY_S = 2.0
REBOOT_SETTLE_S = 3.0
LAUNCHER_TIMEOUT_S = 10.0      # APP_STOP tears the app down and rebuilds the main screen


@dataclass
class Params:
    ssid_env: str = field(default="CROSSPAD_WIFI_SSID",
                          metadata={"help": "environment variable holding the bench SSID"})
    pass_env: str = field(default="CROSSPAD_WIFI_PASS",
                          metadata={"help": "environment variable holding the bench password"})
    cycles: int = field(
        default=20,
        metadata={"help": "WIFI_UP/WIFI_DOWN cycles for the leak check (~45 s each)"})
    linger: float = field(default=35.0,
                          metadata={"help": "seconds to wait out the 30 s linger "
                                            "(radio off, task retired)"})
    up_timeout: float = field(default=10.0, metadata={"help": "seconds for state=up after a start"})
    auth_timeout: float = field(default=25.0,
                                metadata={"help": "seconds for failed/auth after a wrong password"})
    leak_tolerance: int = field(default=1024,
                                metadata={"help": "bytes of internal RAM allowed to differ"})


@dataclass
class _Original:
    """What the board held before the scenario touched it; None = not readable (old firmware)."""
    wifi_enabled: bool | None
    ble_running: bool


@dataclass
class _Checks:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.rows.append({"name": name, "ok": bool(ok), "detail": detail})


def _wait_state(ctx: Context, link: CdcLink, want: set[str], timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        st = verbs.wifi_status(link)
        if st.get("state") in want or time.monotonic() >= deadline:
            return st
        ctx.cancelled.wait(POLL_S)
        ctx.check_cancelled()


def _wait_ble(ctx: Context, link: CdcLink, running: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if verbs.ble_status(link).get("running") == running:
            return True
        if time.monotonic() >= deadline:
            return False
        ctx.cancelled.wait(POLL_S)
        ctx.check_cancelled()


def _describe(st: dict[str, Any]) -> str:
    return (f"state={st.get('state')} fail={st.get('fail')} reason={st.get('reason')} "
            f"leases={st.get('leases')}")


def _reopen(ctx: Context, link: CdcLink, want_active: bool) -> CdcLink:
    """The board is rebooting into (or out of) network mode. Network mode has no app registry,
    so CdcLink.wait_ready() (APP_LIST) would never pass there: reopen the port, as fw_switch
    does, and wait for NET_MODE to say where the board landed. A first answer can still come
    from the old boot in the ~150 ms before its restart -- that one says the old mode and is
    skipped."""
    ctx.close_all()
    deadline = time.monotonic() + REOPEN_TIMEOUT_S
    while True:
        try:
            new_link = ctx.open_cdc()
            if verbs.net_mode_status(new_link)["active"] == want_active:
                ctx.cancelled.wait(REBOOT_SETTLE_S)
                ctx.check_cancelled()
                return new_link
            new_link.close()
        except HilError as exc:
            # ctx.open_cdc() registers a link before its wait_ready() can fail; left open it
            # would hold the port's lock against the next attempt (PORT_BUSY).
            ctx.close_all()
            if exc.code not in (NO_DEVICE, TIMEOUT):
                raise
        if time.monotonic() >= deadline:
            raise HilError(TIMEOUT, f"wifi: board did not come back in "
                                    f"{'network' if want_active else 'normal'} mode within "
                                    f"{REOPEN_TIMEOUT_S:.0f}s",
                           hint="tools/bench.py check; tools/bench.py ready leaves network mode")
        ctx.cancelled.wait(REOPEN_RETRY_S)
        ctx.check_cancelled()
        # ctx.open_cdc() trusts ctx.device.ports.cdc; refresh it across the reboot (fw_switch).
        try:
            ctx.device = select(discover(), ctx.device.id)
        except HilError:
            pass


def _to_launcher(ctx: Context, link: CdcLink) -> str | None:
    """Stop whatever app runs and wait until APP_LIST shows none; returns the app it found."""
    start_app = verbs.app_list(link)["running"]
    if start_app is None:
        return None
    verbs.app_stop(link)
    deadline = time.monotonic() + LAUNCHER_TIMEOUT_S
    while True:
        running = verbs.app_list(link)["running"]
        if running is None:
            return start_app
        if time.monotonic() >= deadline:
            raise HilError(TIMEOUT, f"wifi: {running} still running {LAUNCHER_TIMEOUT_S:.0f}s "
                                    "after APP_STOP",
                           hint="tools/bench.py check; the WiFi checks need the launcher")
        ctx.cancelled.wait(POLL_S)
        ctx.check_cancelled()


def _refused(fn: Any, *args: Any) -> tuple[bool, str]:
    try:
        fn(*args)
    except HilError as exc:
        return True, exc.message
    return False, "answered OK"


def _ping(link: CdcLink) -> tuple[bool, str]:
    try:
        ping = verbs.wifi_ping(link)
    except HilError as exc:
        return False, exc.message
    return bool(ping.get("ok")), f"ms={ping.get('ms')} status={ping.get('status')}"


def _step(ctx: Context, what: str, fn: Any, *args: Any) -> None:
    """One restore step: a failure is logged and the next step still runs."""
    try:
        fn(*args)
    except Exception as exc:
        ctx.log(f"[wifi] restore: {what} failed: {exc}")


def _live_link(ctx: Context, link: CdcLink) -> CdcLink:
    """`link` if the board still answers on it, else a fresh one (a reboot closed it)."""
    try:
        verbs.net_mode_status(link)
        return link
    except Exception:
        ctx.close_all()
        return ctx.open_cdc()


def _start_ble_if_down(link: CdcLink) -> None:
    # BLE_ENABLE 1 already starts it; BLE_START on a running stack answers ERR.
    if verbs.ble_status(link).get("running") != 1:
        verbs.ble_start(link)


def _restore(ctx: Context, link: CdcLink, original: _Original, ssid: str, password: str,
             in_net: bool) -> None:
    try:
        link = _live_link(ctx, link)
        if in_net:
            verbs.net_mode_set(link, False)
            link = _reopen(ctx, link, False)
    except Exception as exc:   # the rest still helps if the board is in normal mode
        ctx.log(f"[wifi] restore: could not leave network mode: {exc} -- "
                "tools/bench.py ready leaves it")
        try:
            link = _live_link(ctx, link)
        except Exception as exc2:
            ctx.log(f"[wifi] restore: no link to the board: {exc2}")
            return
    _step(ctx, "re-saving the bench network", verbs.wifi_set, link, ssid, password)
    if original.wifi_enabled is not None:
        _step(ctx, "WiFi enable", verbs.wifi_enable, link, original.wifi_enabled)
    _step(ctx, "BLE enable", verbs.ble_enable, link, original.ble_running)
    if original.ble_running:
        _step(ctx, "BLE start", _start_ble_if_down, link)


class WifiScenario:
    name = "wifi"
    Params = Params
    description = ("WiFi on demand: join, ping, wrong password, no leak, BLE handover; "
                   "one network-mode pass")

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        p = params
        ssid, password = os.environ.get(p.ssid_env, ""), os.environ.get(p.pass_env, "")
        if not ssid or not password:
            return Report(passed=False,
                          summary=f"set {p.ssid_env} / {p.pass_env} "
                                  "(set -a; . ~/.config/crosspad/bench_wifi.env; set +a)",
                          data={}, artifacts=[], exit_code=2)
        checks = _Checks()
        data: dict[str, Any] = {"checks": checks.rows}
        ctx.partial = data
        link = ctx.open_cdc()
        data["start_app"] = _to_launcher(ctx, link)
        in_net = False
        enabled = verbs.wifi_status(link).get("enabled")
        original = _Original(wifi_enabled=None if enabled is None else enabled == 1,
                             ble_running=verbs.ble_status(link).get("running") == 1)
        try:
            verbs.wifi_enable(link, True)
            verbs.ble_stop(link)

            verbs.wifi_set(link, ssid, password)
            st = _wait_state(ctx, link, {"up"}, p.up_timeout)
            checks.add("connect", st.get("state") == "up", _describe(st))
            ok, detail = _ping(link)
            checks.add("ping", ok, detail)

            resets = verbs.reset_count(link)
            verbs.wifi_set(link, ssid, password + "x")
            st = _wait_state(ctx, link, {"failed"}, p.auth_timeout)
            same_boot = verbs.reset_count(link) == resets
            checks.add("wrong_password", st.get("fail") == "auth" and same_boot,
                       f"{_describe(st)} resets_unchanged={same_boot}")
            verbs.wifi_set(link, ssid, password)
            _wait_state(ctx, link, {"up"}, p.up_timeout)

            # WIFI_SET took no lease: once up, the linger stops the radio and retires the task.
            ctx.cancelled.wait(p.linger)
            ctx.check_cancelled()
            base = int(verbs.mem(link)["int_free"])
            for i in range(p.cycles):
                verbs.wifi_up(link)
                st = _wait_state(ctx, link, {"up"}, p.up_timeout)
                if st.get("state") != "up":
                    verbs.wifi_down(link)
                    checks.add("no_leak", False, f"cycle {i + 1}: {_describe(st)}")
                    break
                verbs.wifi_down(link)
                ctx.cancelled.wait(p.linger)
                ctx.check_cancelled()
                progress(i + 1, p.cycles, f"radio cycle {i + 1}/{p.cycles}")
            else:
                end = int(verbs.mem(link)["int_free"])
                checks.add("no_leak", abs(end - base) <= p.leak_tolerance,
                           f"int_free {base} -> {end} after {p.cycles} cycles")

            verbs.ble_enable(link, True)
            ble_before = _wait_ble(ctx, link, 1, BLE_TIMEOUT_S)
            verbs.wifi_up(link)
            st = _wait_state(ctx, link, {"up"}, p.up_timeout)
            ble_off_while_up = verbs.ble_status(link).get("running") == 0
            verbs.wifi_down(link)
            ctx.cancelled.wait(p.linger)
            ctx.check_cancelled()
            ble_back = _wait_ble(ctx, link, 1, BLE_TIMEOUT_S)
            checks.add("ble_handover",
                       ble_before and st.get("state") == "up" and ble_off_while_up and ble_back,
                       f"before={ble_before} {_describe(st)} off_while_up={ble_off_while_up} "
                       f"back={ble_back}")

            # Before the call: a reply that times out can still mean the board is rebooting into
            # network mode, and the restore must then find it there. When it is not, NET_MODE 0
            # answers "OK unchanged" and the restore carries on in the normal boot.
            in_net = True
            verbs.net_mode_set(link, True, tab="wifi")
            link = _reopen(ctx, link, True)
            st = _wait_state(ctx, link, {"up"}, p.up_timeout)
            ping_ok, ping_detail = _ping(link)
            ble = verbs.ble_status(link)
            refused, why = _refused(verbs.ble_start, link)
            checks.add("network_mode",
                       st.get("state") == "up" and st.get("leases") == 1 and ping_ok
                       and ble.get("running") == 0 and refused,
                       f"{_describe(st)} ping: {ping_detail} ble running={ble.get('running')} "
                       f"BLE_START: {why}")
        finally:
            _restore(ctx, link, original, ssid, password, in_net)
            ctx.close_all()

        failed = [c["name"] for c in checks.rows if not c["ok"]]
        summary = ("all checks passed" if not failed else "failed: " + ", ".join(failed))
        ctx.log(f"[wifi] {summary}")
        return Report(passed=not failed, summary=summary, data=data, artifacts=[],
                      exit_code=0 if not failed else 1)


SCENARIO = register(WifiScenario())
