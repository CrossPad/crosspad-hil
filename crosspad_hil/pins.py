"""A host-side journal of the persistent settings a run changed on a board.

A scenario that pins a setting the board keeps in NVS (the jack preamp, IN2;
the input pair, which can switch the jack loop's driver off)
puts it back in a ``finally``. A run that is killed never gets there: an
``audio_loopback`` stopped mid-run left IN2 at +24 dB in NVS, and the next
person to plug a line source in heard it 24 dB hot. So the old value is
written here *before* the board is changed, and the entry is removed only once
the board is back. Whatever is left in the journal is a promise somebody did
not keep, and the next run -- any run that pins the same setting, or the
bench's normalising step -- keeps it first.

One small JSON file, ``{device_id: {key: previous}}``, next to the bench lease
(``$XDG_STATE_HOME/crosspad-hil/pins.json``, or ``$CROSSPAD_HIL_PINS``).
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from crosspad_hil import verbs
from crosspad_hil.cdc import CdcLink
from crosspad_hil.errors import HilError

PINS_ENV = "CROSSPAD_HIL_PINS"
IN2 = "in2"
IN_PAIR = "in_pair"


def default_path() -> Path:
    env = os.environ.get(PINS_ENV)
    if env:
        return Path(env)
    state = os.environ.get("XDG_STATE_HOME")
    base = Path(state) if state else Path.home() / ".local" / "state"
    return base / "crosspad-hil" / "pins.json"


def _load(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(path: Path, data: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def record(device_id: str, key: str, previous: Any, path: Path | None = None) -> None:
    """Note ``previous`` before the board is changed. An older entry wins: it is
    the value the board had before *any* unfinished run touched it."""
    path = path or default_path()
    data = _load(path)
    data.setdefault(device_id, {}).setdefault(key, previous)
    _save(path, data)


def clear(device_id: str, key: str, path: Path | None = None) -> None:
    path = path or default_path()
    data = _load(path)
    if key in data.get(device_id, {}):
        del data[device_id][key]
        if not data[device_id]:
            del data[device_id]
        _save(path, data)


def pending(device_id: str, path: Path | None = None) -> dict[str, Any]:
    return dict(_load(path or default_path()).get(device_id, {}))


def _restore_in2(link: CdcLink, value: Any) -> bool:
    return verbs.ingain(link, 1, int(value)).get("IN2") == int(value)


def _restore_in_pair(link: CdcLink, value: Any) -> bool:
    return verbs.in_pair(link, int(value))["pair"] == int(value)


RESTORERS: dict[str, Callable[[CdcLink, Any], bool]] = {IN2: _restore_in2,
                                                        IN_PAIR: _restore_in_pair}


def restore_pending(link: CdcLink, device_id: str, log: Callable[[str], None],
                    path: Path | None = None, keys: tuple[str, ...] | None = None
                    ) -> dict[str, Any]:
    """Put back what an unfinished run left changed; ``{key: value}`` restored.
    An entry that cannot be put back stays in the journal."""
    done: dict[str, Any] = {}
    for key, value in pending(device_id, path).items():
        if keys is not None and key not in keys:
            continue
        fn = RESTORERS.get(key)
        if fn is None:
            log(f"pins: no way to restore {key!r} (left in the journal)")
            continue
        try:
            ok = fn(link, value)
        except HilError as e:
            log(f"pins: restoring {key}={value!r} failed: {e.code}: {e.message}")
            continue
        if ok:
            clear(device_id, key, path)
            done[key] = value
            log(f"pins: {key} was left changed by an unfinished run; put back to {value!r}")
        else:
            log(f"pins: {key} did not read back as {value!r} (left in the journal)")
    return done
