"""The verdict as two lines of text, then the details; the local page is drawn by `page.py`."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np


def pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def times(r: float) -> str:
    """A multiple of the ceiling: two decimals near 1, where one would round 1.04 down to "1.0"."""
    return f"{r:.2f}×" if r < 1.1 else f"{r:.1f}×"


def hm(seconds: float) -> str:
    m = int(round(seconds / 60))
    return f"{m // 60} h {m % 60:02d}" if m >= 60 else f"{m} min"


# ----------------------------------------------------------------- text


def plan_text(p, name: str, cond: str, prof, watts: bool = False) -> str:
    sh, lo, hi = p.shares, p.shares_low, p.shares_high
    up = float(np.sum(np.clip(np.diff(prof.elev), 0, None)))
    lines = [
        f"Zone 2 is possible on {pct(sh.km_share)} of the kilometres, but {pct(sh.time_share)} of the time.",
        f"{len(p.passages)} passage{'s' if len(p.passages) != 1 else ''} where it is not: {sum(x.length_m for x in p.passages) / 1000:.1f} km, "
        f"{hm(sum(x.time_s for x in p.passages))} even at your slowest pace.",
        "",
        f"{name}: {sh.km_total:.1f} km, {up:.0f} m of climbing (terrain model), about {hm(sh.time_total_s)} riding easy.",
        f"Conditions: {cond}.",
        f"Range over the ceiling's uncertainty: {pct(lo.km_share)}–{pct(hi.km_share)} of the km, "
        f"{pct(lo.time_share)}–{pct(hi.time_share)} of the time.",
    ]
    if p.passages:
        lines += ["", "Where Zone 2 is impossible:"]
        for x in p.passages:
            mid = [s for s in p.stretches if x.start_m <= s.start_m < x.end_m]
            extra = ""
            if watts and mid:
                worst = max(mid, key=lambda s: s.ratio)
                extra = f"  ({worst.power_min_w:.0f} W needed, ceiling {worst.ceiling_w:.0f} W)"
            lines.append(f"  km {x.start_m / 1000:5.1f}  {x.length_m / 1000:4.1f} km at {100 * x.mean_grade:4.1f} % "
                         f"(up to {100 * x.max_grade:4.1f} %), {times(x.max_ratio)} the ceiling{extra}")
    return "\n".join(lines)


def review_text(r, rd, z2_top: float, cool: float) -> str:
    when = datetime.fromtimestamp(rd.start_epoch, timezone.utc).astimezone()
    air = float(np.nanmedian(rd.temp_air[rd.moving]))
    lines = [
        f"{r.above_min:.0f} min above Zone 2: {r.road_min:.0f} imposed by the road, {r.heat_min:.0f} by the heat, "
        f"{r.me_min:.0f} mine.",
        f"Zone 2 was possible on {pct(r.shares.km_share)} of the kilometres, {pct(r.shares.time_share)} of the time.",
        "",
        f"Ride of {when:%Y-%m-%d %H:%M}: {r.shares.km_total:.1f} km, {r.moving_min:.0f} min moving, air {air:.0f} °C "
        f"(median), Zone 2 up to {z2_top:.0f} bpm.",
        f"Ranges over the ceiling's uncertainty: road {r.road_range[0]:.0f}–{r.road_range[1]:.0f} min, "
        f"heat {r.heat_range[0]:.0f}–{r.heat_range[1]:.0f}, me {r.me_range[0]:.0f}–{r.me_range[1]:.0f}.",
        f"The road: Zone 2 impossible even in the cool ({cool:.0f} °C). The heat: possible in the cool, not that day.",
    ]
    if abs(r.day_offset_bpm) >= 0.5:
        lines.append(f"That day the heart ran {abs(r.day_offset_bpm):.0f} bpm {'above' if r.day_offset_bpm > 0 else 'below'} "
                     f"its usual level at the same power: the ceilings take it into account.")
    g = r.gear
    if g.get("impossible_s"):
        lines.append(f"Gear check: where Zone 2 was impossible, the smallest gear was in "
                     f"{pct(g['impossible_smallest_gear_share'])} of the time.")
    if g.get("possible_above_s"):
        lines.append(f"Where it was possible and the heart was above it, an easier gear was left "
                     f"{pct(g['possible_above_easier_gear_left_share'])} of the time.")
    return "\n".join(lines)


# ----------------------------------------------------------------- page




def write_plan_html(path, p, name: str, cond: str, prof, watts: bool = False) -> None:
    from .page import plan_page

    with open(path, "w", encoding="utf-8") as f:
        f.write(plan_page(p, name, cond, prof))


def write_review_html(path, r, rd, z2_top: float, cool: float) -> None:
    from .page import review_page

    with open(path, "w", encoding="utf-8") as f:
        f.write(review_page(r, rd, z2_top, cool))
