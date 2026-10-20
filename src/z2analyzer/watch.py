"""The lazy mode: every new ride in a folder gets its verdict, waiting for you in the morning.

    z2analyzer watch ~/rides --limits limits.json --terrain dem/ --z2-top 145            # once (cron)
    z2analyzer watch ~/rides --limits limits.json --terrain dem/ --z2-top 145 --every 30 # keeps looking

Point it at the folder where your bike computer's FIT files land. Each new
ride is judged once; its two lines go to the top of `z2analyzer/morning.md`
in that folder, next to the full text and the local page of the ride. A file
that changes (still being copied) is judged again on the next look.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def watch(args) -> int:
    from .cli import log, rider_from_env
    from .dem import Terrain
    from .limits import Limits
    from .structures import Structures

    folder = Path(args.folder)
    out = folder / "z2analyzer"
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / "done.json"
    rider = rider_from_env(args.mass_kg)
    lim = Limits.load(args.limits)
    terrain = Terrain(args.terrain)
    structures = Structures(args.structures) if args.structures else None
    cool = float((lim.notes or {}).get("cool_c", 15.0))
    while True:
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
        new = 0
        for f in sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime):
            if not f.is_file() or f.suffix.lower() != ".fit":
                continue
            stamp = f"{f.stat().st_size}:{int(f.stat().st_mtime)}"
            if state.get(f.name) == stamp:
                continue
            try:
                lines = judge_one(f, out, rider, lim, terrain, structures, args.z2_top, cool, args.edge_temp)
                new += 1
                print("\n".join(lines[:2]), flush=True)
            except ValueError as e:
                log(f"skipped: {e}")
            state[f.name] = stamp
            state_file.write_text(json.dumps(state, indent=1), encoding="utf-8")
        if new:
            log(f"{new} new ride(s) judged; verdicts in {out / 'morning.md'}")
        if args.every <= 0:
            return 0
        time.sleep(args.every * 60)


def judge_one(fit: Path, out: Path, rider, lim, terrain, structures, z2_top: float, cool: float, edge_temp: bool) -> list[str]:
    """Judge one ride, write its text and page, and put its two lines on top of morning.md."""
    from .cli import ride_from_fit
    from .report import review_text, write_review_html
    from .verdict import review

    rd = ride_from_fit(fit, terrain, structures, edge_temp)
    r = review(dist_m=rd.dist, speed_ms=rd.speed, hr=rd.hr, cadence_rpm=rd.cadence, moving=rd.moving,
               temp_air_c=rd.temp_air, profile=rd.profile, rider=rider, limits=lim, z2_top_bpm=z2_top,
               cool_temp_c=cool, power_w=rd.power)
    text = review_text(r, rd, z2_top, cool)
    (out / f"{fit.stem}.txt").write_text(text + "\n", encoding="utf-8")
    write_review_html(out / f"{fit.stem}.html", r, rd, z2_top, cool)
    when = datetime.fromtimestamp(rd.start_epoch, timezone.utc).astimezone()
    lines = text.splitlines()
    entry = (f"## {when:%a %d %b %Y, %H:%M} · {fit.name}\n\n{lines[0]}\n{lines[1]}\n"
             f"Air {float(np.nanmedian(rd.temp_air[rd.moving])):.0f} °C · {r.moving_min:.0f} min moving · "
             f"[the ride's page]({fit.stem}.html)\n")
    morning = out / "morning.md"
    old = morning.read_text(encoding="utf-8") if morning.exists() else "# z2analyzer, for the morning\n"
    head, _, rest = old.partition("\n")
    morning.write_text(f"{head}\n\n{entry}\n{rest.lstrip()}", encoding="utf-8")
    return lines
