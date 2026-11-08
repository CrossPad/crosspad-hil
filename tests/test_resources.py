"""resources: fd counting from an injected directory, and its failure modes."""
from __future__ import annotations

import os
from pathlib import Path

from crosspad_hil.resources import ALSA_SEQ_DEV, fd_counts


def _link(d: Path, name: str, target: str) -> None:
    os.symlink(target, d / name)


def test_counts_alsa_seq_links_and_total(tmp_path: Path) -> None:
    _link(tmp_path, "0", "/dev/null")
    _link(tmp_path, "1", ALSA_SEQ_DEV)
    _link(tmp_path, "2", ALSA_SEQ_DEV)
    _link(tmp_path, "3", "/dev/ttyACM0")
    assert fd_counts(str(tmp_path)) == {"open_fds": 4, "alsa_seq_clients": 2}


def test_empty_dir_is_all_zero(tmp_path: Path) -> None:
    assert fd_counts(str(tmp_path)) == {"open_fds": 0, "alsa_seq_clients": 0}


def test_missing_dir_does_not_raise(tmp_path: Path) -> None:
    assert fd_counts(str(tmp_path / "nope")) == {"open_fds": 0, "alsa_seq_clients": 0}


def test_dangling_link_still_counts_as_an_fd(tmp_path: Path) -> None:
    _link(tmp_path, "0", "/dev/snd/seq")
    _link(tmp_path, "1", str(tmp_path / "gone"))
    assert fd_counts(str(tmp_path)) == {"open_fds": 2, "alsa_seq_clients": 1}
