"""Round-trip a file over CDC, and optionally reflash the assets partition.

The bulk verbs (platform-idf `main/file_transfer.cpp`) take the pipe out of
newline framing for an announced byte count. Two things can go wrong that no
build catches, and this scenario exists for exactly those two:

  * the CRC the device computes (`esp_crc32_le`) disagreeing with the host's
    (`zlib.crc32`) — every transfer would then fail, loudly but universally;
  * the framing not switching back to text at the right byte, which corrupts
    the *next* command rather than the transfer, so it looks like an unrelated
    verb misbehaving.

A round trip plus a text verb immediately afterwards covers both. `--assets`
adds the whole-partition write, which reboots the device.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosspad_hil import fs as fsmod
from crosspad_hil import verbs
from crosspad_hil.errors import HilError
from crosspad_hil.scenarios import register
from crosspad_hil.scenarios.base import Context, Progress, Report

# Big enough to span several 4096-byte chunks, small enough to stay quick.
DEFAULT_SIZE = 20_000


@dataclass
class Params:
    root: str = field(
        default="/sdcard",
        metadata={"help": "writable mount to test in (/sdcard, /spiflash, /assets)"},
    )
    size: int = field(
        default=DEFAULT_SIZE, metadata={"help": "bytes of random payload to round-trip"}
    )
    assets: str = field(
        default="", metadata={"help": "assets image to flash (reboots the device)"}
    )
    keep: bool = field(
        default=False, metadata={"help": "leave the test file on the device"}
    )


class FsTransferScenario:
    name = "fs_transfer"
    Params = Params
    description = "round-trip a file over CDC (CRC + framing), optionally reflash assets"

    def run(self, ctx: Context, params: Params, progress: Progress) -> Report:
        checks: list[dict[str, Any]] = []
        data: dict[str, Any] = {"checks": checks}

        def check(name: str, passed: bool, detail: str) -> bool:
            checks.append({"name": name, "passed": passed, "detail": detail})
            ctx.log(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
            return passed

        try:
            # The streaming half opens its own serial handle (like ota.py), and a
            # CdcLink reader thread on the same port would eat half of its replies.
            # So: never hold a link across push/pull — open, ask, close.
            link = ctx.open_cdc()
            try:
                roots = verbs.fs_roots(link)["roots"]
            except HilError as exc:
                # The worker's 5 kB stack must be internal RAM (it writes flash);
                # with BLE up the launcher has none to give. That is the bench's
                # state, not a transfer defect, so it is exit 2 with the cure.
                if "no_worker" in str(exc):
                    return Report(passed=False, data=data, artifacts=[], exit_code=2,
                                  summary=f"no internal RAM for the transfer worker "
                                          f"({exc.details.get('reply')}); stop BLE first "
                                          f"(crosspad-hil ble stop) and retry")
                raise
            finally:
                link.close()
            ctx.log(f"mounts: {roots}")
            data["roots"] = roots
            if params.root not in roots:
                return Report(passed=False, data=data, artifacts=[], exit_code=2,
                              summary=f"{params.root} is not a mount the firmware knows: "
                                      f"{sorted(roots)}")
            if roots[params.root]["total"] is None:
                return Report(passed=False, data=data, artifacts=[], exit_code=2,
                              summary=f"{params.root} is not mounted (no card in the slot?)")

            remote = f"{params.root}/hil_fs_transfer.bin"
            payload = os.urandom(max(1, params.size))
            local = ctx.workdir / "fs_transfer_out.bin"
            local.write_bytes(payload)
            back = ctx.workdir / "fs_transfer_back.bin"

            progress(1, 4, "push")
            pushed = fsmod.push(ctx.device, local, remote, serial_factory=ctx.serial_factory)
            data["push"] = pushed
            check("push", pushed["crc32"] == pushed["device_crc32"].lower(),
                  f"host {pushed['crc32']} vs device {pushed['device_crc32']}")

            # The transfer left binary mode; a text verb must answer normally.
            progress(2, 4, "framing")
            link = ctx.open_cdc()
            try:
                stat = verbs.fs_stat(link, remote)
            finally:
                link.close()
            check("framing_after_push", stat["size"] == len(payload),
                  f"FS_STAT answered {stat['size']} bytes after the stream")

            progress(3, 4, "pull")
            pulled = fsmod.pull(ctx.device, remote, back, serial_factory=ctx.serial_factory)
            data["pull"] = pulled
            same = back.read_bytes() == payload
            check("round_trip", same,
                  f"{pulled['bytes']} bytes back, byte-identical" if same else "bytes differ")

            progress(4, 4, "framing")
            link = ctx.open_cdc()
            try:
                after = verbs.fs_roots(link)["roots"]
                check("framing_after_pull", bool(after), "FS_ROOTS answered after the pull")
                if not params.keep:
                    verbs.fs_delete(link, remote)
            finally:
                link.close()

            if params.assets:
                ctx.log(f"flashing assets from {params.assets} — device reboots")
                res = fsmod.assets_flash(ctx.device, Path(params.assets),
                                         serial_factory=ctx.serial_factory)
                data["assets"] = res
                check("assets_flash", True,
                      f"{res['bytes']} bytes at {res['kbps']} KB/s, device rebooting")
                link = ctx.open_cdc()
                try:
                    # ASSETS_OK is followed by a restart: the node comes back
                    # within seconds, the verbs only once the boot has finished.
                    link.wait_ready()
                    info = verbs.stm_fw_info(link)
                    data["stm_fw"] = info
                    check("reboot_after_assets", True, f"STM_FW_INFO: {info}")
                finally:
                    link.close()

            passed = all(c["passed"] for c in checks)
            summary = f"{sum(c['passed'] for c in checks)}/{len(checks)} checks passed"
            return Report(passed=passed, summary=summary, data=data, artifacts=[],
                          exit_code=0 if passed else 1)
        except HilError:
            raise
        finally:
            ctx.close_all()


SCENARIO = register(FsTransferScenario())
