"""The jam's verdict and the files a person reads it from.

* ``jam_report.json`` -- everything: verdict, per-class and per-cause counts,
  every glitch with all its clocks, its step and actions, what the model and
  the board said around it; coverage of the features; the clocks; what was
  put back;
* ``timeline.md`` -- the same, readable;
* ``timeline.html`` -- one file, no external library: lanes for the steps,
  actions, kits, the sequencer, voices sounding and the glitches (click one for
  its record);
* ``labels.txt`` -- Audacity labels for the takes (steps and glitches);
* ``glitches/gNNNN_{cap,exp,diff}.wav`` -- 0.5 s around each glitch: the take,
  what it should have been, the difference.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

PASS = "ZERO-GLITCH PASS"
FAIL = "FAIL"
NOT_MEASURED = "not measured"


def verdict(glitches: list[dict[str, Any]], counters: list[dict[str, Any]],
            not_measured: list[str], coverage_missing: list[str], allow_partial: bool,
            left_changed: list[str]) -> tuple[str, int, list[str]]:
    """(status, exit code, the lines that decide it)."""
    lines: list[str] = []
    bad = [g for g in glitches if g.get("status") == "glitch"]
    for g in bad[:30]:
        ax = g.get("axes") or {}
        lines.append(f"{g['gid']} {g['block']} t={ax.get('wav_s', 0):.4f}s "
                     f"({ax.get('wall', '')}) step={g.get('step')} {g['kind']} "
                     f"{dict(g['classes'])} cause={g['cause']}({g['confidence']:.2f})")
    for c in counters:
        lines.append(f"counter {c.get('block')}: {c.get('note')}")
    if left_changed:
        lines.append("LEFT CHANGED: " + "; ".join(left_changed))
    if bad or counters:
        return FAIL, 1, lines
    if not_measured or (coverage_missing and not allow_partial):
        why = list(not_measured) + ([f"features not exercised: {', '.join(coverage_missing)}"]
                                    if coverage_missing and not allow_partial else [])
        return NOT_MEASURED, 2, why + lines
    return PASS, 0, lines


def write_json(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=1, default=_default) + "\n", encoding="utf-8")


def _default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)


def write_timeline_md(path: Path, doc: dict[str, Any]) -> None:
    v = doc.get("verdict", {})
    out = [f"# live_jam — {v.get('status')}", "",
           f"- run: `{doc.get('run_dir')}`", f"- seed: {doc.get('seed')}",
           f"- UAC: {doc.get('uac_seconds', 0) / 60:.1f} min in {len(doc.get('blocks', []))} "
           "block(s)", ""]
    if v.get("lines"):
        out += ["## Decides it", ""] + [f"- {ln}" for ln in v["lines"]] + [""]
    out += ["## Glitches", "",
            "| id | block | wav s | wall | dev ms | step | kind | cause | conf | action |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for g in doc.get("glitches", []):
        ax = g.get("axes") or {}
        act = g.get("action") or {}
        out.append(f"| {g['gid']} | {g['block']} | {ax.get('wav_s', 0):.4f} | "
                   f"{ax.get('wall', '')} | {ax.get('dev_log_ms', '')} | {g.get('step')} | "
                   f"{g['kind']}{'' if g['status'] == 'glitch' else ' (' + g['status'] + ')'}"
                   f" | {g['cause']} | {g['confidence']:.2f} | "
                   f"{act.get('kind', '')} a{act.get('aid', '')} {act.get('age_ms', '')} ms |")
    out += ["", "## Counters", ""]
    for c in doc.get("counters", []):
        out.append(f"- {c.get('block')}: {c.get('note')}")
    out += ["", "## Per condition (glitches / minute)", "", "| flag | glitches | min | /min |",
            "|---|---|---|---|"]
    for f, r in (doc.get("by_flag") or {}).items():
        out.append(f"| {f} | {r['glitches']} | {r['minutes']} | {r['per_min']} |")
    out += ["", "## Coverage", "", "| feature | status | steps |", "|---|---|---|"]
    for f, r in (doc.get("coverage") or {}).items():
        out.append(f"| {f} | {r.get('status')} | {', '.join(r.get('steps', [])[:4])} |")
    out += ["", "## Blocks", ""]
    for b in doc.get("blocks", []):
        m = b.get("measured", {})
        out.append(f"### {b['name']} ({b.get('seconds', 0):.1f} s)")
        out.append(f"- voices judged {m.get('judged')} / unverifiable {m.get('unverifiable')}")
        out.append(f"- latency: {json.dumps(m.get('latency_ms'))}")
        out.append(f"- pilot: {json.dumps(b.get('pilot', {}).get('residual'))}")
        if b.get("not_measured"):
            out.append(f"- NOT MEASURED: {b['not_measured']}")
        out.append("")
    if doc.get("product_gaps"):
        out += ["## Product gaps", ""] + [f"- {g}" for g in doc["product_gaps"]] + [""]
    if doc.get("left"):
        out += ["## Put back", "", "```", json.dumps(doc["left"], indent=1), "```"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_audacity_labels(path: Path, doc: dict[str, Any], block: str) -> None:
    rows = []
    for s in doc.get("steps", []):
        if s.get("block") == block and s.get("wav_s") is not None:
            rows.append((s["wav_s"], s.get("wav_s1", s["wav_s"]), s["id"]))
    for g in doc.get("glitches", []):
        if g["block"] != block:
            continue
        ax = g.get("axes") or {}
        t = float(ax.get("wav_s", 0))
        rows.append((t, t + g.get("length_ms", 0) / 1000.0, f"{g['gid']} {g['kind']} "
                     f"{g['cause']}"))
    rows.sort()
    path.write_text("".join(f"{a:.6f}\t{b:.6f}\t{t}\n" for a, b, t in rows), encoding="utf-8")


def write_timeline_html(path: Path, doc: dict[str, Any]) -> None:
    """One self-contained page: an SVG per block, lanes, and a details pane."""
    blocks = doc.get("blocks", [])
    parts = []
    for b in blocks:
        name = b["name"]
        dur = max(1e-3, float(b.get("seconds", 1.0)))
        width = 1400

        def x(t: float, d: float = dur, w: int = width) -> float:
            return 60 + (w - 80) * max(0.0, min(1.0, t / d))

        svg = [f'<svg viewBox="0 0 {width} 230" class="tl" role="img" '
               f'aria-label="block {html.escape(name)} timeline">']
        lanes = ["steps", "actions", "flags", "voices", "glitches"]
        for i, lane in enumerate(lanes):
            y = 20 + i * 42
            svg.append(f'<text x="4" y="{y + 16}" class="lane">{lane}</text>')
            svg.append(f'<line x1="60" x2="{width - 20}" y1="{y + 30}" y2="{y + 30}" '
                       'class="axis"/>')
        for s in doc.get("steps", []):
            if s.get("block") != name or s.get("wav_s") is None:
                continue
            x0, x1 = x(s["wav_s"]), x(s.get("wav_s1", s["wav_s"]))
            svg.append(f'<rect x="{x0:.1f}" y="24" width="{max(1.0, x1 - x0):.1f}" height="22" '
                       f'class="step"><title>{html.escape(str(s["id"]))}</title></rect>')
        for a in b.get("actions_lane", []):
            svg.append(f'<line x1="{x(a[0]):.1f}" x2="{x(a[0]):.1f}" y1="66" y2="88" '
                       f'class="act {html.escape(str(a[1]))}">'
                       f'<title>{html.escape(str(a[2]))}</title></line>')
        for fl in b.get("flags_lane", []):
            x0, x1 = x(fl[0]), x(fl[1])
            svg.append(f'<rect x="{x0:.1f}" y="108" width="{max(1.0, x1 - x0):.1f}" '
                       f'height="22" class="flag"><title>{html.escape(str(fl[2]))}</title></rect>')
        poly = b.get("poly_lane", [])
        if poly:
            pts = " ".join(f"{x(t):.1f},{172 - 2.0 * min(16, n):.1f}" for t, n in poly)
            svg.append(f'<polyline points="{pts}" class="poly"/>')
        for g in doc.get("glitches", []):
            if g["block"] != name:
                continue
            t = float((g.get("axes") or {}).get("wav_s", 0))
            cls = "g" if g["status"] == "glitch" else "u"
            svg.append(f'<circle cx="{x(t):.1f}" cy="205" r="6" class="{cls}" '
                       f'data-gid="{html.escape(str(g["gid"]))}" tabindex="0"><title>'
                       f'{html.escape(str(g["gid"]))} {html.escape(str(g["kind"]))} '
                       f'{html.escape(str(g["cause"]))}</title></circle>')
        svg.append("</svg>")
        parts.append(f"<h2>{html.escape(name)} <small>{dur:.1f} s</small></h2>" + "".join(svg))
    # Console text and causes end up in this JSON; inside <script> a "</script>" in them
    # would end the block, so <, > and & are written as JSON escapes.
    data = (json.dumps({g["gid"]: g for g in doc.get("glitches", [])}, default=_default)
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))
    v = doc.get("verdict", {})
    page = f"""<!doctype html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>live_jam timeline</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1c1a; --muted:#77736b; --step:#dfe6f0; --flag:#efe3cc;
        --act:#5b6b8c; --poly:#2f7d5b; --g:#c0392b; --u:#b9a46a; --axis:#d5d0c6; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#17181a;
  --fg:#e9e6df; --muted:#9b978e; --step:#2b3442; --flag:#3e3526; --act:#93a6cf;
  --poly:#64c39a; --g:#ff6b5b; --u:#d6c27e; --axis:#3a3a3a; }} }}
:root[data-theme="dark"] {{ --bg:#17181a; --fg:#e9e6df; --muted:#9b978e; --step:#2b3442;
  --flag:#3e3526; --act:#93a6cf; --poly:#64c39a; --g:#ff6b5b; --u:#d6c27e; --axis:#3a3a3a; }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif;
       margin:0 16px 40px; }}
.tl {{ width:100%; height:auto; }} .lane {{ fill:var(--muted); font-size:11px; }}
.axis {{ stroke:var(--axis); }} .step {{ fill:var(--step); }} .flag {{ fill:var(--flag); }}
.act {{ stroke:var(--act); stroke-width:1; }} .poly {{ fill:none; stroke:var(--poly); }}
.g {{ fill:var(--g); cursor:pointer; }} .u {{ fill:var(--u); cursor:pointer; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:color-mix(in srgb,var(--fg) 6%,
  transparent); padding:12px; border-radius:6px; max-height:60vh; overflow:auto; }}
small {{ color:var(--muted); font-weight:normal; }}
</style></head><body>
<h1>live_jam — {html.escape(str(v.get('status')))}</h1>
<p>{html.escape(str(doc.get('run_dir', '')))} · seed {html.escape(str(doc.get('seed')))} ·
{len([g for g in doc.get('glitches', []) if g['status'] == 'glitch'])} glitch(es)</p>
{''.join(parts)}
<h2>Glitch</h2><pre id="detail">click a red dot</pre>
<script>
const G = {data};
for (const c of document.querySelectorAll('circle[data-gid]')) {{
  const show = () => {{ document.getElementById('detail').textContent =
    JSON.stringify(G[c.dataset.gid], null, 1); }};
  c.addEventListener('click', show); c.addEventListener('keydown', show);
}}
</script></body></html>
"""
    path.write_text(page, encoding="utf-8")


def cut_snippets(out_dir: Path, x: Any, exp_x: Any, sr: int, glitches: list[dict[str, Any]],
                 limit: int, half_s: float = 0.25) -> None:
    """0.5 s of the take, of the expectation and their difference around each glitch."""
    from crosspad_hil.jam.render import write_wav

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for g in glitches:
        if g["status"] != "glitch" or n >= limit:
            continue
        f = int((g.get("axes") or {}).get("frame", 0))
        a, b = max(0, f - int(half_s * sr)), min(len(x), f + int(half_s * sr))
        if b <= a:
            continue
        cap = x[a:b]
        paths = {"cap": out_dir / f"{g['gid']}_cap.wav"}
        write_wav(paths["cap"], cap, sr)
        if exp_x is not None and len(exp_x) >= b:
            ex = exp_x[a:b]
            paths["exp"] = out_dir / f"{g['gid']}_exp.wav"
            paths["diff"] = out_dir / f"{g['gid']}_diff.wav"
            write_wav(paths["exp"], ex, sr)
            write_wav(paths["diff"], (cap - ex) * 0.5, sr)
        g["snippet"] = {k: str(v) for k, v in paths.items()}
        n += 1
