from __future__ import annotations

from pathlib import Path

from crosspad_hil.jam import report


def test_console_text_cannot_break_out_of_the_timeline_page(tmp_path: Path) -> None:
    """Causes and console tails come from the board; they stay text in the page."""
    evil = '</script><img src=x onerror=alert(1)>"\''
    doc = {
        "run_dir": "run", "seed": 1, "verdict": {"status": "FAIL"},
        "blocks": [{"name": "b1", "seconds": 10.0, "actions_lane": [], "flags_lane": [],
                    "poly_lane": []}],
        "steps": [],
        "glitches": [{"gid": 'g1"x', "kind": evil, "block": "b1", "status": "glitch",
                      "cause": evil, "axes": {"wav_s": 1.0}}],
    }
    out = tmp_path / "timeline.html"
    report.write_timeline_html(out, doc)
    page = out.read_text(encoding="utf-8")
    assert page.count("</script>") == 1                 # only the page's own
    assert "<img" not in page
    assert 'data-gid="g1&quot;x"' in page
