"""The verdict as two lines of text, then the details, and a local page that loads nothing from anywhere.

The page draws the route's shape and profile as inline SVG, coloured by
verdict. No map tiles: a tile server would learn which area you looked at.
The first and last 2 km of the shape are left out, so a route starting at
home does not point at home.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

import numpy as np

from .geo import trim_mask

GREEN = "#2f9e6e"
RED = "#d9534f"
AMBER = "#e0a030"
GREY = "#8a94a6"
CREDITS = ("Built with PriorLabs-TabPFN · terrain: IGN RGE ALTI® (Licence Ouverte 2.0) · bridges and tunnels: "
           "© OpenStreetMap contributors (ODbL) · weather: Open-Meteo, Météo-France models (CC BY 4.0)")


def pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def hm(seconds: float) -> str:
    m = int(round(seconds / 60))
    return f"{m // 60} h {m % 60:02d}" if m >= 60 else f"{m} min"


# ----------------------------------------------------------------- text


def plan_text(p, name: str, cond: str, prof, watts: bool = False) -> str:
    sh, lo, hi = p.shares, p.shares_low, p.shares_high
    up = float(np.sum(np.clip(np.diff(prof.elev), 0, None)))
    lines = [
        f"Zone 2 is possible on {pct(sh.km_share)} of the kilometres, but {pct(sh.time_share)} of the time.",
        f"{len(p.passages)} passages where it is not: {sum(x.length_m for x in p.passages) / 1000:.1f} km, "
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
                         f"(up to {100 * x.max_grade:4.1f} %), {x.max_ratio:.1f}× the ceiling{extra}")
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
    g = r.gear
    if g.get("impossible_s"):
        lines.append(f"Gear check: where Zone 2 was impossible, the smallest gear was in "
                     f"{pct(g['impossible_smallest_gear_share'])} of the time.")
    if g.get("possible_above_s"):
        lines.append(f"Where it was possible and the heart was above it, an easier gear was left "
                     f"{pct(g['possible_above_easier_gear_left_share'])} of the time.")
    return "\n".join(lines)


# ----------------------------------------------------------------- page


def _svg_profile(dist_km, elev, ok, width=960, height=220) -> str:
    """Elevation profile, filled green where Zone 2 is possible and red where it is not."""
    x0, x1 = float(dist_km[0]), float(dist_km[-1])
    lo, hi = float(np.min(elev)), float(np.max(elev))
    pad = max(10.0, 0.08 * (hi - lo))
    lo, hi = lo - pad, hi + pad
    X = lambda d: 40 + (d - x0) / max(x1 - x0, 1e-9) * (width - 50)  # noqa: E731
    Y = lambda z: height - 25 - (z - lo) / max(hi - lo, 1e-9) * (height - 40)  # noqa: E731
    parts = []
    for i in range(len(dist_km) - 1):
        c = GREEN if ok[i] else RED
        parts.append(f'<polygon points="{X(dist_km[i]):.1f},{Y(lo):.1f} {X(dist_km[i]):.1f},{Y(elev[i]):.1f} '
                     f'{X(dist_km[i + 1]):.1f},{Y(elev[i + 1]):.1f} {X(dist_km[i + 1]):.1f},{Y(lo):.1f}" fill="{c}" />')
    line = " ".join(f"{X(d):.1f},{Y(z):.1f}" for d, z in zip(dist_km, elev))
    ticks = "".join(f'<text x="{X(k):.1f}" y="{height - 6}" class="tick">{k:.0f}</text>'
                    for k in np.arange(np.ceil(x0), x1 + 1e-9, max(1, round((x1 - x0) / 10))))
    zt = "".join(f'<text x="34" y="{Y(z) + 4:.1f}" class="tick" text-anchor="end">{z:.0f}</text>'
                 for z in np.linspace(lo + pad, hi - pad, 3))
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Elevation profile">{"".join(parts)}'
            f'<polyline points="{line}" fill="none" stroke="var(--ink)" stroke-width="1.2" />{ticks}{zt}'
            f'<text x="{width - 10}" y="{height - 6}" class="tick" text-anchor="end">km</text></svg>')


def _svg_shape(lat, lon, ok, keep, width=420, height=420) -> str:
    """The route's shape, without its first and last 2 km and without any map underneath."""
    if keep.sum() < 2:
        return ""
    la, lo_ = lat[keep], lon[keep]
    k = np.cos(np.radians(np.mean(la)))
    x, y = lo_ * k, la
    span = max(x.max() - x.min(), y.max() - y.min(), 1e-9)
    X = 20 + (x - x.min()) / span * (width - 40)
    Y = height - 20 - (y - y.min()) / span * (height - 40)
    o = ok[keep]
    segs = []
    for i in range(len(X) - 1):
        segs.append(f'<line x1="{X[i]:.1f}" y1="{Y[i]:.1f}" x2="{X[i + 1]:.1f}" y2="{Y[i + 1]:.1f}" '
                    f'stroke="{GREEN if o[i] else RED}" stroke-width="4" stroke-linecap="round" />')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Route shape">{"".join(segs)}</svg>'


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<title>{title}</title>
<style>
:root {{ --bg:#f6f4ef; --card:#ffffff; --ink:#1d2433; --muted:#5d6678; --green:{green}; --red:{red}; --amber:{amber}; --grey:{grey}; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141821; --card:#1d2330; --ink:#e8ebf2; --muted:#9aa3b5; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1040px; margin:0 auto; padding:24px 16px 48px; }}
header .kicker {{ text-transform:uppercase; letter-spacing:.12em; font-size:12px; color:var(--muted); }}
h1 {{ font-size:clamp(26px,4.4vw,42px); line-height:1.15; margin:.2em 0 .3em; }}
h1 .g {{ color:var(--green); }} h1 .r {{ color:var(--red); }}
.lede {{ font-size:18px; color:var(--muted); margin:0 0 20px; }}
.card {{ background:var(--card); border-radius:14px; padding:18px; margin:16px 0; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
.row {{ display:grid; grid-template-columns:1fr; gap:16px; }}
@media (min-width:860px) {{ .row {{ grid-template-columns:1.6fr 1fr; }} }}
svg {{ width:100%; height:auto; display:block; }}
.tick {{ font-size:11px; fill:var(--muted); }}
table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid rgba(127,127,127,.2); }}
th {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }}
.bar {{ display:flex; height:34px; border-radius:8px; overflow:hidden; margin:8px 0 6px; }}
.bar div {{ display:flex; align-items:center; justify-content:center; color:#fff; font-weight:600; font-size:14px; }}
.legend span {{ display:inline-block; margin-right:16px; font-size:14px; }}
.legend i {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }}
footer {{ color:var(--muted); font-size:13px; margin-top:24px; }}
.big {{ font-size:34px; font-weight:700; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
.stats div small {{ display:block; color:var(--muted); font-size:13px; }}
</style></head><body><main>
{body}
<footer>{credits}<br>A finding about a route or a ride, not training or medical advice. Wind is not counted.</footer>
</main></body></html>
"""


def _page(title: str, body: str) -> str:
    return PAGE.format(title=html.escape(title), body=body, credits=html.escape(CREDITS), green=GREEN, red=RED,
                       amber=AMBER, grey=GREY)


def write_plan_html(path, p, name: str, cond: str, prof, watts: bool = False) -> None:
    sh = p.shares
    st = p.stretches
    mids = np.array([(s.start_m + s.end_m) / 2 for s in st])
    ok_at = np.array([s.possible for s in st])
    d_km = prof.dist / 1000.0
    ok_prof = ok_at[np.clip(np.searchsorted([s.end_m for s in st], prof.dist), 0, len(st) - 1)] if st else np.ones(prof.dist.size, bool)
    keep = trim_mask(prof.dist, 2000.0) & np.isfinite(prof.lat)
    rows = "".join(
        f"<tr><td>km {x.start_m / 1000:.1f}</td><td>{x.length_m / 1000:.1f} km</td><td>{100 * x.mean_grade:.1f} %</td>"
        f"<td>{100 * x.max_grade:.1f} %</td><td>{x.max_ratio:.1f}×</td><td>{hm(x.time_s)}</td></tr>" for x in p.passages)
    table = (f'<div class="card"><h2>Where Zone 2 is impossible</h2><table><tr><th>From</th><th>Length</th><th>Grade</th>'
             f'<th>Steepest</th><th>Needs</th><th>Time</th></tr>{rows}</table>'
             f'<p class="tick">"Needs": the power of your slowest pace there, as a multiple of your Zone 2 ceiling.</p></div>'
             if p.passages else "")
    shape = _svg_shape(prof.lat, prof.lon, ok_prof, keep)
    body = f"""
<header><div class="kicker">z2analyzer · before the ride</div>
<h1><span class="g">{pct(sh.km_share)}</span> of the kilometres, but <span class="r">{pct(sh.time_share)}</span> of the time.</h1>
<p class="lede">That is how much of {html.escape(name)} you can ride in Zone 2, even in your smallest gear at the slowest cadence you really hold. {html.escape(cond)}.</p></header>
<div class="card stats"><div><span class="big">{sh.km_total:.1f} km</span><small>route</small></div>
<div><span class="big">{len(p.passages)}</span><small>passages where Zone 2 is impossible</small></div>
<div><span class="big">{sum(x.length_m for x in p.passages) / 1000:.1f} km</span><small>of them, {hm(sum(x.time_s for x in p.passages))} at your slowest</small></div>
<div><span class="big">{hm(sh.time_total_s)}</span><small>riding easy</small></div></div>
<div class="row"><div class="card"><h2>Profile</h2>{_svg_profile(d_km, prof.elev, ok_prof)}
<div class="legend"><span><i style="background:var(--green)"></i>Zone 2 possible</span><span><i style="background:var(--red)"></i>impossible</span></div></div>
<div class="card"><h2>Route</h2>{shape}<p class="tick">First and last 2 km left out.</p></div></div>
{table}"""
    _ = mids
    with open(path, "w", encoding="utf-8") as f:
        f.write(_page(f"z2analyzer · {name}", body))


def write_review_html(path, r, rd, z2_top: float, cool: float) -> None:
    tot = max(r.above_min, 1e-9)
    parts = [("the road", r.road_min, RED), ("the heat", r.heat_min, AMBER), ("me", r.me_min, GREY)]
    bar = "".join(f'<div style="width:{100 * m / tot:.2f}%;background:{c}">{m:.0f}</div>' for _, m, c in parts if m >= 0.5)
    legend = "".join(f'<span><i style="background:{c}"></i>{lab}: {m:.0f} min</span>' for lab, m, c in parts)
    prof = rd.profile
    d_km = prof.dist / 1000.0
    ps = r.per_second
    # Profile coloured by what the road allowed that day.
    poss = np.interp(prof.dist, rd.dist, ps["possible"].astype(float)) >= 0.5
    when = datetime.fromtimestamp(rd.start_epoch, timezone.utc).astimezone()
    g = r.gear
    gear_html = ""
    if g.get("impossible_s"):
        gear_html = (f'<div class="card"><h2>The check that needs no model</h2><p>Where Zone 2 was impossible, the smallest '
                     f'gear was already in <b>{pct(g["impossible_smallest_gear_share"])}</b> of the time. Where it was possible '
                     f'and the heart was above it anyway, an easier gear was still there <b>'
                     f'{pct(g.get("possible_above_easier_gear_left_share", float("nan")))}</b> of the time.</p></div>')
    body = f"""
<header><div class="kicker">z2analyzer · after the ride · {when:%Y-%m-%d}</div>
<h1>{r.above_min:.0f} min above Zone 2: <span class="r">{r.road_min:.0f}</span> imposed by the road, {r.heat_min:.0f} by the heat, {r.me_min:.0f} mine.</h1>
<p class="lede">Zone 2 up to {z2_top:.0f} bpm. Each minute above it goes to the road (impossible even in the cool, {cool:.0f} °C), the heat (possible in the cool, not that day) or me (it was possible).</p></header>
<div class="card"><div class="bar">{bar}</div><div class="legend">{legend}</div>
<p class="tick">Ranges over the ceiling's uncertainty: road {r.road_range[0]:.0f}–{r.road_range[1]:.0f} min, heat {r.heat_range[0]:.0f}–{r.heat_range[1]:.0f}, me {r.me_range[0]:.0f}–{r.me_range[1]:.0f}.</p></div>
<div class="card stats"><div><span class="big">{pct(r.shares.km_share)}</span><small>of the kilometres could stay in Zone 2</small></div>
<div><span class="big">{pct(r.shares.time_share)}</span><small>of the time</small></div>
<div><span class="big">{r.moving_min:.0f} min</span><small>moving</small></div></div>
<div class="card"><h2>Profile</h2>{_svg_profile(d_km, prof.elev, poss)}
<div class="legend"><span><i style="background:var(--green)"></i>Zone 2 possible that day</span><span><i style="background:var(--red)"></i>impossible</span></div></div>
{gear_html}"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(_page(f"z2analyzer · ride of {when:%Y-%m-%d}", body))
