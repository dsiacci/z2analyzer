"""Record the demo video from z2analyzer's own pages.

    python tools/demo.py --plan plan.html --review review.html --out demo/ [--voice voice/] [--tempo 1.05]

Each shot is one of the tool's pages (plan, review) or a title card,
photographed at 1920×1080 by a headless Chromium with the scene's highlight
drawn in. ffmpeg turns the shots into a video with a slow zoom and
crossfades, each scene paced on its voice-over clip (voice/<scene>.mp3) when
there is one. Writes z2analyzer-demo.mp4 and z2analyzer-demo.srt (captions).
Needs chromium (or google-chrome), ffmpeg and ffprobe.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H, FPS, FADE, LEAD = 1920, 1080, 30, 0.5, 0.35

PALETTE = """:root { --bg:#e9eeec; --card:#ffffff; --ink:#15202a; --muted:#566370; --line:rgba(21,32,42,.12);
  --green:#2f9e6e; --red:#d9534f; --amber:#e0a030; --grey:#8a94a6; }
* { box-sizing:border-box; } html, body { margin:0; height:100%; }
body { background:var(--bg); color:var(--ink); font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  display:grid; place-items:center; }"""


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-1500:]}")
    return r.stdout


def browser() -> str:
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        if shutil.which(name):
            return name
    raise SystemExit("needs chromium or google-chrome")


def shoot(html_file: Path, png: Path) -> None:
    """1280×720 CSS pixels at a scale of 1.5: a 1920×1080 picture with sharp text."""
    run([browser(), "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
         "--force-device-scale-factor=1.5", "--window-size=1280,720", f"--screenshot={png}", html_file.resolve().as_uri()])


def card(kind: str, spec: dict) -> str:
    e = html.escape
    if kind == "title":
        body = (f'<div class="t"><div class="brand">{e(spec["brand"])}</div><div class="line">{e(spec["line"])}</div>'
                f'<div class="sub">{e(spec["sub"])}</div></div>')
        css = """.t { text-align:center; } .brand { font-weight:800; font-size:92px; letter-spacing:-.02em; }
.brand::after { content:""; display:block; height:8px; width:220px; margin:14px auto 0; border-radius:4px;
  background:linear-gradient(90deg, var(--green) 0 70%, var(--red) 70% 100%); }
.line { font-size:44px; font-weight:700; margin-top:28px; } .sub { font-size:24px; color:var(--muted); margin-top:12px; }"""
    elif kind == "proof":
        rows = "".join(
            f'<tr><td>{e(r[0])}</td><td class="{"win" if r[3] == "tabpfn" else ""}">{e(r[1])}</td>'
            f'<td class="{"win" if r[3] == "line" else ""}">{e(r[2])}</td></tr>' for r in spec["rows"])
        body = (f'<div class="p"><h1>{e(spec["title"])}</h1><table><tr><th></th><th>TabPFN v2</th><th>Straight line</th></tr>'
                f'{rows}</table><p>{e(spec["note"])}</p></div>')
        css = """.p { width:1060px; } h1 { font-size:40px; margin:0 0 26px; }
table { width:100%; border-collapse:separate; border-spacing:0 10px; font-size:30px; font-variant-numeric:tabular-nums; }
th { text-align:left; font-size:18px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); padding:0 20px; }
td { background:var(--card); padding:18px 20px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
td:first-child { border-left:1px solid var(--line); border-radius:14px 0 0 14px; font-weight:600; }
td:last-child { border-right:1px solid var(--line); border-radius:0 14px 14px 0; }
td.win { color:var(--green); font-weight:800; } td.win::after { content:" ✓"; }
p { font-size:22px; color:var(--muted); margin-top:18px; }"""
    elif kind == "total":
        tot = spec["road"] + spec["heat"] + spec["me"]
        seg = "".join(f'<div style="flex:{v};background:var(--{c})"><b>{v}</b><span>{lab}</span></div>'
                      for v, c, lab in ((spec["road"], "red", "the road"), (spec["heat"], "amber", "the heat"), (spec["me"], "grey", "me")))
        body = f'<div class="g"><h1>{e(spec["title"])}</h1><div class="bar">{seg}</div><p>{tot} minutes, one cause each</p></div>'
        css = """.g { width:1140px; } h1 { font-size:46px; margin:0 0 30px; }
.bar { display:flex; height:150px; border-radius:18px; overflow:hidden; }
.bar div { display:flex; flex-direction:column; justify-content:center; align-items:center; color:#fff; min-width:120px; }
.bar b { font-size:54px; line-height:1; } .bar span { font-size:22px; margin-top:6px; }
p { font-size:24px; color:var(--muted); }"""
    else:  # outro
        lines = "".join(f"<div>{e(x)}</div>" for x in spec["lines"])
        body = f'<div class="o"><div class="brand">{e(spec["brand"])}</div>{lines}</div>'
        css = """.o { text-align:center; font-size:26px; color:var(--muted); line-height:1.7; }
.brand { font-size:80px; font-weight:800; color:var(--ink); margin-bottom:18px; }"""
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{PALETTE}{css}</style></head><body>{body}</body></html>"


def page_variant(src: Path, view: str, row: str | None) -> str:
    """The tool's own page, with the scene's part on screen and its highlight drawn in."""
    s = src.read_text(encoding="utf-8")
    css = {
        # Headline numbers ringed in their own colour.
        "top": "h1 .g, h1 .r, h1 .a, h1 .m { border-radius:8px; box-shadow:0 0 0 5px var(--card), 0 0 0 9px currentColor; }"
               " .bar { box-shadow:0 0 0 4px var(--card), 0 0 0 8px var(--ink); }",
        # Profile and table of passages, with one row lit up.
        "passage": "header, .stats, .row > .card:first-child { display:none; } .row { grid-template-columns:1fr !important; }"
                   " main { max-width:1500px; zoom:.84; padding-top:8px; } tr.hl td { background:rgba(217,83,79,.14); font-weight:700; }"
                   " tr.hl td:first-child { box-shadow:inset 5px 0 0 var(--red); }",
    }[view]
    if row:
        s = re.sub(r"<tr( class=\"longest\")?><td>" + re.escape(row) + r"</td>", f'<tr class="hl"><td>{row}</td>', s, count=1)
    return s.replace("</style>", css + "</style>", 1)


def duration(path: Path) -> float:
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]).strip())


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="page written by `z2analyzer plan --html`")
    ap.add_argument("--review", required=True, help="page written by `z2analyzer review --html`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", help="folder of <scene>.mp3 voice-over clips")
    ap.add_argument("--tempo", type=float, default=1.0, help="play the voice this much faster")
    a = ap.parse_args()
    conf = json.loads((HERE / "demo" / "scenes.json").read_text(encoding="utf-8"))
    out = Path(a.out)
    shots = out / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    pages = {"plan": Path(a.plan), "review": Path(a.review)}
    voice = Path(a.voice) if a.voice else None

    plan_ = []
    for sc in conf["scenes"]:
        kind, view = sc["shot"].split(":")
        page = shots / f"{sc['id']}.html"
        if kind == "card":
            page.write_text(card(view, conf["cards"][view]), encoding="utf-8")
        else:
            page.write_text(page_variant(pages[kind], view, sc.get("row")), encoding="utf-8")
        png = shots / f"{sc['id']}.png"
        shoot(page, png)
        clip = voice / f"{sc['id']}.mp3" if voice and (voice / f"{sc['id']}.mp3").exists() else None
        spoken = duration(clip) / a.tempo if clip else len(sc["say"]) / 15.0
        plan_.append({**sc, "png": png, "clip": clip, "spoken": spoken,
                      "d": max(sc["min"], LEAD + spoken + 0.9)})

    # Video: a slow 4 % zoom on each shot, then crossfades between them.
    segs = []
    for k, sc in enumerate(plan_):
        seg = shots / f"{sc['id']}.mp4"
        frames = int(round(sc["d"] * FPS))
        run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(sc["png"]), "-t", f"{sc['d']:.3f}",
             "-vf", f"scale={W * 2}:{H * 2},zoompan=z='1+0.04*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                    f":d={frames}:s={W}x{H}:fps={FPS},format=yuv420p",
             "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(seg)])
        segs.append(seg)
    starts, t = [], 0.0
    for k, sc in enumerate(plan_):
        starts.append(t)
        t += sc["d"] - (FADE if k < len(plan_) - 1 else 0)
    total = t
    inputs = sum((["-i", str(s)] for s in segs), [])
    chain, last = [], "[0:v]"
    for k in range(1, len(segs)):
        label = f"[v{k}]"
        chain.append(f"{last}[{k}:v]xfade=transition=fade:duration={FADE}:offset={starts[k]:.3f}{label}")
        last = label
    vfilter = ";".join(chain) if chain else "[0:v]null[v1]"
    vlabel = last if chain else "[v1]"

    audio_in, amix = [], []
    for k, sc in enumerate(plan_):
        if sc["clip"]:
            n = len(segs) + len(audio_in) // 2
            audio_in += ["-i", str(sc["clip"])]
            delay = int(round((starts[k] + LEAD) * 1000))
            amix.append(f"[{n}:a]atempo={a.tempo},adelay={delay}|{delay}[a{k}]")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, *audio_in]
    if amix:
        labels = "".join(f"[a{k}]" for k, sc in enumerate(plan_) if sc["clip"])
        afilter = ";".join(amix) + f";{labels}amix=inputs={len(amix)}:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11,apad[aout]"
        cmd += ["-filter_complex", f"{vfilter};{afilter}", "-map", vlabel, "-map", "[aout]", "-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-filter_complex", vfilter, "-map", vlabel]
    target = out / "z2analyzer-demo.mp4"
    cmd += ["-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(target)]
    run(cmd)

    lines = []
    for k, sc in enumerate(plan_):
        a0 = starts[k] + LEAD
        lines.append(f"{k + 1}\n{srt_time(a0)} --> {srt_time(a0 + sc['spoken'])}\n{sc['say']}\n")
    (out / "z2analyzer-demo.srt").write_text("\n".join(lines), encoding="utf-8")
    print(f"{target}: {total:.1f} s, {len(plan_)} scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
