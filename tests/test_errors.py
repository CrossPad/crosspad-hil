"""HilError shape — the daemon serialises it verbatim, so the dict keys are a contract."""
from __future__ import annotations

import pytest

from crosspad_hil import __version__, errors
from crosspad_hil.errors import ALL_CODES, HilError


def test_version() -> None:
    assert __version__ == "1.8.0"


def test_to_dict_has_exactly_four_keys() -> None:
    err = HilError(errors.TIMEOUT, "no reply to KIT_STATUS", hint="is the device in UAC2 mode?",
                   port="/dev/ttyACM0", timeout_s=2.0)
    d = err.to_dict()
    assert set(d) == {"code", "message", "hint", "details"}
    assert d["code"] == "TIMEOUT"
    assert d["message"] == "no reply to KIT_STATUS"
    assert d["hint"] == "is the device in UAC2 mode?"
    assert d["details"] == {"port": "/dev/ttyACM0", "timeout_s": 2.0}


def test_hint_defaults_to_none_and_details_to_empty() -> None:
    err = HilError(errors.NO_DEVICE, "no CrossPad found")
    assert err.hint is None
    assert err.details == {}
    assert err.to_dict() == {"code": "NO_DEVICE", "message": "no CrossPad found",
                             "hint": None, "details": {}}


def test_str_is_code_colon_message() -> None:
    assert str(HilError(errors.BAD_ARGS, "pad must be 0..15")) == "BAD_ARGS: pad must be 0..15"


def test_is_an_exception() -> None:
    with pytest.raises(HilError) as ei:
        raise HilError(errors.CANCELLED, "stopped")
    assert ei.value.code == "CANCELLED"


@pytest.mark.parametrize("code", [
    "NO_DEVICE", "AMBIGUOUS_DEVICE", "PORT_BUSY", "NO_CDC_IN_AUDIO_MODE", "TIMEOUT",
    "UNKNOWN_VERB", "DENIED_SYSEX", "BAD_SYSEX", "HANDLE_EXPIRED", "BAD_ARGS",
    "FLASH_FAILED", "ENV", "CANCELLED", "NOT_SUPPORTED",
])
def test_every_code_constant_exists_and_equals_its_name(code: str) -> None:
    assert getattr(errors, code) == code
    assert code in ALL_CODES


def test_all_codes_has_fifteen_entries() -> None:
    assert len(ALL_CODES) == 15
    assert len(set(ALL_CODES)) == 15


def test_unknown_code_is_rejected() -> None:
    with pytest.raises(ValueError):
        HilError("NOT_A_CODE", "x")
