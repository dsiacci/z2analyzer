"""Record the demo video from the web app's own pages.

    python tools/demo.py --app pages/ --out demo/ [--voice voice/] [--tempo 1.05]

`--app` holds the app's pages as the video shows them, saved with the
functions of `z2analyzer.web` from a real store: home-before.html (the list
before the new analyses), new-review.html and new-plan.html (a form, its file
chosen), progress-review.html and progress-plan.html (an analysis running),
review.html and plan.html (the results), home-after.html (the list at the
end). Each shot is one of those pages or a title card, photographed at
1920×1080 by a headless Chromium with the scene's highlight drawn in. ffmpeg
turns the shots into a video with a slow zoom and crossfades, each scene
paced on its voice-over clip (voice/<scene>.mp3) when there is one. Writes
z2analyzer-demo.mp4 and z2analyzer-demo.srt (captions). Needs chromium (or
google-chrome), ffmpeg and ffprobe.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H, FPS, FADE, LEAD = 1920, 1080, 30, 0.5, 0.35

sys.path.insert(0, str(HERE.parent / "src"))
from z2analyzer.page import CSS, fonts_css  # noqa: E402

CARD_CSS = """
html, body { height:100%; } body { display:grid; place-items:center; background:var(--paper); }
.dawn { position:fixed; inset:0; color:#fff; overflow:hidden;
  background:radial-gradient(circle at 74% 84%, rgba(255,233,190,.95) 0 2.6%, rgba(255,196,130,.55) 5%, rgba(255,160,110,0) 16%),
    linear-gradient(180deg, var(--sky1) 0%, var(--sky2) 34%, var(--sky3) 62%, var(--sky4) 84%, var(--sky5) 100%); }
.dawn svg { position:absolute; left:0; right:0; bottom:0; width:100%; height:50%; }
.dawn .glow { filter:drop-shadow(0 0 5px rgba(255,120,90,.95)); }
.centre { position:absolute; inset:0; display:grid; place-content:center; text-align:center; padding-bottom:12vh; }
.brand { font:800 128px/.9 var(--display); letter-spacing:.01em; }
.line { font:800 58px/1 var(--display); text-transform:uppercase; margin-top:22px; }
.subl { font:500 26px/1.3 var(--ui); color:rgba(255,255,255,.82); margin-top:12px; }
.credits { font:500 23px/1.6 var(--ui); color:rgba(255,255,255,.86); margin-top:20px; }
.board { width:1100px; } .board h1 { font:800 60px/1 var(--display); text-transform:uppercase; margin:0 0 26px; }
.board table { width:100%; border-collapse:separate; border-spacing:0 12px; font:600 32px/1.1 var(--ui); font-variant-numeric:tabular-nums; }
.board th { text-align:left; font:700 18px/1 var(--ui); letter-spacing:.14em; text-transform:uppercase; color:var(--muted); padding:0 22px; }
.board td { background:var(--card); padding:22px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
.board td:first-child { border-left:1px solid var(--line); border-radius:16px 0 0 16px; }
.board td:last-child { border-right:1px solid var(--line); border-radius:0 16px 16px 0; }
.board td.win { color:var(--ok); font:800 44px/1 var(--display); } .board td.lose { color:var(--muted); font:600 40px/1 var(--display); }
.board p { font:500 22px/1.4 var(--ui); color:var(--muted); margin:16px 0 0; }
"""


def run(cmd: list[str], env: dict | None = None) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-1500:]}")
    return r.stdout


def browser() -> str:
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        if shutil.which(name):
            return name
    raise SystemExit("needs chromium or google-chrome")


def shoot(html_file: Path, png: Path) -> None:
    """1280×720 CSS pixels at a scale of 1.5: a 1920×1080 picture with sharp text (dates written day first)."""
    run([browser(), "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--lang=en-GB",
         "--force-device-scale-factor=1.5", "--window-size=1280,720", f"--screenshot={png}", html_file.resolve().as_uri()],
        env=dict(os.environ, LANG="en_GB.UTF-8", LANGUAGE="en_GB"))


def card(kind: str, spec: dict, skyline_svg: str) -> str:
    e = html.escape
    if kind == "title":
        body = (f'<div class="dawn">{skyline_svg}<div class="centre"><div class="brand">{e(spec["brand"])}</div>'
                f'<div class="line">{e(spec["line"])}</div><div class="subl">{e(spec["sub"])}</div></div></div>')
    elif kind == "outro":
        lines = "".join(f"<div>{e(x)}</div>" for x in spec["lines"])
        body = f'<div class="dawn">{skyline_svg}<div class="centre"><div class="brand">{e(spec["brand"])}</div><div class="credits">{lines}</div></div></div>'
    else:  # proof
        rows = "".join(f'<tr><td>{e(r[0])}</td><td class="{"win" if r[3] == "tabpfn" else "lose"}">{e(r[1])}</td>'
                       f'<td class="{"win" if r[3] == "line" else "lose"}">{e(r[2])}</td></tr>' for r in spec["rows"])
        body = (f'<div class="board"><h1>{e(spec["title"])}</h1><table><tr><th></th><th>TabPFN v2</th><th>Straight line</th></tr>{rows}</table>'
                f'<p>{e(spec["note"])}</p></div>')
    return doc(body)


def doc(body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{fonts_css()}{CSS}{CARD_CSS}</style></head>"
            f"<body>{body}</body></html>")


def skyline_of(page_html: str) -> str:
    """The two skyline layers of a page's hero, for the title and closing cards."""
    m = re.search(r'<header class="hero">(.*?)<div class="hero-text">', page_html, re.S)
    return m.group(1) if m else ""


def page_variant(src: Path, view: str | None, row: str | None) -> str:
    """The app's own page, with the scene's part on screen and its highlight drawn in."""
    s = re.sub(r'<meta http-equiv="refresh"[^>]*>', "", src.read_text(encoding="utf-8"))
    css = {
        None: "",
        # The list of earlier analyses, the newest one found in the watched folder.
        "start": ".hero.home { height:350px; } .total { display:none !important; } main { margin-top:14px !important; }"
                 " .list .tile:first-child { box-shadow:0 0 0 4px var(--paper), 0 0 0 8px var(--ok); }",
        # A form with its file chosen, and the button that sends it.
        "drop": ".hero.short .sub { display:none; } .hero.short { min-height:0; padding-bottom:70px; } main { zoom:.8; }"
                " .drop.has { box-shadow:0 0 0 4px var(--card), 0 0 0 9px var(--ok); }"
                " form:has(.drop.has) .btn.big { box-shadow:0 0 0 4px var(--card), 0 0 0 8px var(--ink); }",
        # The headline numbers (plan) or the split of the minutes (review), ringed.
        "top": ".metric { border-radius:14px; padding:10px 12px; background:color-mix(in srgb, var(--ok) 9%, var(--card));"
               " box-shadow:0 0 0 3px var(--ok); } .split { box-shadow:0 0 0 5px var(--card), 0 0 0 9px var(--ink); }",
        # The roadbook and the cards of the passages, one of them lit up.
        "passage": ".appbar, .hero, section.score, .shape, .about, footer { display:none !important; }"
                   " main { margin-top:22px; max-width:1240px; zoom:.8; }"
                   " .climb.hl { background:color-mix(in srgb, var(--no) 14%, var(--card)); box-shadow:0 0 0 3px var(--no); }",
        # The heart rate alone, the minutes above Zone 2 coloured by cause.
        "heart": ".appbar, .hero, main > section:not(:nth-of-type(2)), footer { display:none !important; }"
                 " main { margin-top:40px; max-width:1180px; } .chart { height:420px !important; }",
        # Every analysis kept, and what the rides add up to.
        "total": ".hero.home { height:300px; } .hero .sub, .hero .btn.big { display:none; }"
                 " .total { box-shadow:0 0 0 4px var(--paper), 0 0 0 9px var(--ink); }",
    }[view]
    if row:
        s = re.sub(r'<li class="climb">(<div class="n">\d+</div><div><div class="where">)' + re.escape(row),
                   r'<li class="climb hl">\1' + row, s, count=1)
    return s.replace("</style>", css + "</style>", 1)


def duration(path: Path) -> float:
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]).strip())


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, help="folder of the app's pages (see above)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", help="folder of <scene>.mp3 voice-over clips")
    ap.add_argument("--tempo", type=float, default=1.0, help="play the voice this much faster")
    a = ap.parse_args()
    conf = json.loads((HERE / "demo" / "scenes.json").read_text(encoding="utf-8"))
    out = Path(a.out)
    shots = out / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    app = Path(a.app)
    voice = Path(a.voice) if a.voice else None

    plan_ = []
    for sc in conf["scenes"]:
        kind, name, *view = sc["shot"].split(":")
        page = shots / f"{sc['id']}.html"
        if kind == "card":
            page.write_text(card(name, conf["cards"][name], skyline_of((app / "plan.html").read_text(encoding="utf-8"))),
                            encoding="utf-8")
        else:
            page.write_text(page_variant(app / f"{name}.html", view[0] if view else None, sc.get("row")), encoding="utf-8")
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
        # loudnorm works at 192 kHz: bring the voice back to 48 kHz for the players.
        cmd += ["-filter_complex", f"{vfilter};{afilter}", "-map", vlabel, "-map", "[aout]", "-c:a", "aac", "-b:a", "160k",
                "-ar", "48000"]
    else:
        cmd += ["-filter_complex", vfilter, "-map", vlabel]
    target = out / "z2analyzer-demo.mp4"
    cmd += ["-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(target)]
    run(cmd)

    lines = []
    for k, sc in enumerate(plan_):
        if not sc["say"]:
            continue  # a silent scene
        a0 = starts[k] + LEAD
        lines.append(f"{len(lines) + 1}\n{srt_time(a0)} --> {srt_time(a0 + sc['spoken'])}\n{sc['say']}\n")
    (out / "z2analyzer-demo.srt").write_text("\n".join(lines), encoding="utf-8")
    print(f"{target}: {total:.1f} s, {len(plan_)} scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
