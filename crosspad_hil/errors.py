"""HilError — every failure carries a code, a message and a hint (what to call next)."""
from __future__ import annotations

from typing import Any

NO_DEVICE = "NO_DEVICE"
AMBIGUOUS_DEVICE = "AMBIGUOUS_DEVICE"
PORT_BUSY = "PORT_BUSY"
NO_CDC_IN_AUDIO_MODE = "NO_CDC_IN_AUDIO_MODE"
TIMEOUT = "TIMEOUT"
UNKNOWN_VERB = "UNKNOWN_VERB"
DENIED_SYSEX = "DENIED_SYSEX"
BAD_SYSEX = "BAD_SYSEX"
HANDLE_EXPIRED = "HANDLE_EXPIRED"
BAD_ARGS = "BAD_ARGS"
FLASH_FAILED = "FLASH_FAILED"
ENV = "ENV"
CANCELLED = "CANCELLED"
NOT_SUPPORTED = "NOT_SUPPORTED"
# The board is claimed by another session (bench.py); details name the holder.
BENCH_BUSY = "BENCH_BUSY"

ALL_CODES: tuple[str, ...] = (
    NO_DEVICE,
    AMBIGUOUS_DEVICE,
    PORT_BUSY,
    NO_CDC_IN_AUDIO_MODE,
    TIMEOUT,
    UNKNOWN_VERB,
    DENIED_SYSEX,
    BAD_SYSEX,
    HANDLE_EXPIRED,
    BAD_ARGS,
    FLASH_FAILED,
    ENV,
    CANCELLED,
    NOT_SUPPORTED,
    BENCH_BUSY,
)


class HilError(Exception):
    """Structured error. ``to_dict()`` is what the daemon puts on the wire."""

    code: str
    message: str
    hint: str | None
    details: dict[str, Any]

    def __init__(self, code: str, message: str, hint: str | None = None, **details: Any) -> None:
        if code not in ALL_CODES:
            raise ValueError(f"unknown HilError code {code!r}; must be one of {ALL_CODES}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.hint = hint
        self.details = dict(details)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "details": dict(self.details),
        }

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
