"""What this process is holding, counted from the inside.

An ALSA sequencer client is one open fd on ``/dev/snd/seq``, so the kernel's
per-machine client ceiling is reachable from here long before the program that
finally fails to open a sequencer notices anything.

Nothing raises: a resource probe must never fail the op that asked for it.
"""
from __future__ import annotations

import os

FD_DIR = "/proc/self/fd"
ALSA_SEQ_DEV = "/dev/snd/seq"


def fd_counts(fd_dir: str = FD_DIR) -> dict[str, int]:
    """``{open_fds, alsa_seq_clients}``; zeros where ``/proc`` is not readable."""
    try:
        names = os.listdir(fd_dir)
    except OSError:
        return {"open_fds": 0, "alsa_seq_clients": 0}
    total = 0
    seq = 0
    for name in names:
        total += 1
        try:
            target = os.readlink(os.path.join(fd_dir, name))
        except OSError:
            continue  # closed between listdir and readlink
        if target == ALSA_SEQ_DEV:
            seq += 1
    return {"open_fds": total, "alsa_seq_clients": seq}
