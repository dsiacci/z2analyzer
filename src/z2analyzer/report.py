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




def _profile_svg(dist_km, elev, colour, marks=(), strip=None, width=1000, height=260) -> str:
    """Elevation profile filled by verdict colour, optional labelled marks, optional strip of causes below.

    `colour[i]` is the fill of the stretch from point i to i+1; `marks` are
    (km, label) pairs; `strip` is a list of colours (or None) per point.
    """
    left, right, top = 46, 12, 14
    base = height - (58 if strip is not None else 30)
    x0, x1 = float(dist_km[0]), float(dist_km[-1])
    lo, hi = float(np.min(elev)), float(np.max(elev))
    pad = max(10.0, 0.06 * (hi - lo))
    lo, hi = lo - pad, hi + pad
    X = lambda d: left + (d - x0) / max(x1 - x0, 1e-9) * (width - left - right)  # noqa: E731
    Y = lambda z: base - (z - lo) / max(hi - lo, 1e-9) * (base - top)  # noqa: E731
    parts = []
    # Merge consecutive points of the same colour into one polygon.
    i, n = 0, len(dist_km)
    while i < n - 1:
        j = i
        while j + 1 < n - 1 and colour[j + 1] == colour[i]:
            j += 1
        pts = [f"{X(dist_km[i]):.1f},{base:.1f}"] + [f"{X(dist_km[k]):.1f},{Y(elev[k]):.1f}" for k in range(i, j + 2)]
        pts.append(f"{X(dist_km[j + 1]):.1f},{base:.1f}")
        parts.append(f'<polygon points="{" ".join(pts)}" fill="{colour[i]}" />')
        i = j + 1
    line = " ".join(f"{X(d):.1f},{Y(z):.1f}" for d, z in zip(dist_km, elev))
    step = max(1, int(round((x1 - x0) / 10)))
    ticks = "".join(f'<text x="{X(k):.1f}" y="{base + 16:.1f}" class="tick" text-anchor="middle">{k:.0f}</text>'
                    for k in np.arange(np.ceil(x0), x1 + 1e-9, step))
    zt = "".join(f'<text x="{left - 6}" y="{Y(z) + 4:.1f}" class="tick" text-anchor="end">{z:.0f} m</text>'
                 for z in np.linspace(lo + pad, hi - pad, 3))
    grid = "".join(f'<line x1="{left}" x2="{width - right}" y1="{Y(z):.1f}" y2="{Y(z):.1f}" class="grid" />'
                   for z in np.linspace(lo + pad, hi - pad, 3))
    tags = []
    placed: list[float] = []
    for km, label in marks:
        x = X(km)
        if any(abs(x - q) < 110 for q in placed):
            continue  # too close to a label already drawn
        placed.append(x)
        z = float(np.interp(km, dist_km, elev))
        y = Y(z) - 10
        tags.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y + 2:.1f}" y2="{Y(z):.1f}" class="pin" />'
                    f'<text x="{x:.1f}" y="{y - 4:.1f}" class="tag" text-anchor="middle">{html.escape(label)}</text>')
    strip_svg = ""
    if strip is not None:
        sy = height - 30
        cells = []
        for k in range(n - 1):
            if strip[k]:
                cells.append(f'<rect x="{X(dist_km[k]):.1f}" y="{sy}" width="{max(X(dist_km[k + 1]) - X(dist_km[k]), 0.6):.1f}" '
                             f'height="14" fill="{strip[k]}" />')
        strip_svg = f'<rect x="{left}" y="{sy}" width="{width - left - right}" height="14" class="lane" />{"".join(cells)}'
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Elevation profile">{grid}{"".join(parts)}'
            f'<polyline points="{line}" fill="none" class="ridge" />{ticks}{zt}'
            f'<text x="{width - right}" y="{base + 16:.1f}" class="tick" text-anchor="end">km</text>{"".join(tags)}{strip_svg}</svg>')


def _shape_svg(lat, lon, colour, keep, width=420, height=360) -> str:
    """The route's shape, without its first and last 2 km and without any map underneath."""
    if keep.sum() < 2:
        return ""
    la, lo_ = lat[keep], lon[keep]
    k = np.cos(np.radians(np.mean(la)))
    x, y = lo_ * k, la
    sx = min((width - 40) / max(x.max() - x.min(), 1e-9), (height - 40) / max(y.max() - y.min(), 1e-9))
    X = 20 + (x - x.min()) * sx + (width - 40 - (x.max() - x.min()) * sx) / 2
    Y = height - 20 - (y - y.min()) * sx - (height - 40 - (y.max() - y.min()) * sx) / 2
    c = np.asarray(colour, dtype=object)[keep]
    segs = []
    i, n = 0, len(X)
    while i < n - 1:
        j = i
        while j + 1 < n - 1 and c[j + 1] == c[i]:
            j += 1
        pts = " ".join(f"{X[m]:.1f},{Y[m]:.1f}" for m in range(i, j + 2))
        segs.append(f'<polyline points="{pts}" fill="none" stroke="{c[i]}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />')
        i = j + 1
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Route shape, ends left out">{"".join(segs)}</svg>'


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<title>{title}</title>
<style>
/* A survey map's paper: cool grey-green ground, white cards, navy ink; green and coral only carry verdicts. */
:root {{ --bg:#e9eeec; --card:#ffffff; --ink:#15202a; --muted:#566370; --line:rgba(21,32,42,.12);
  --green:{green}; --red:{red}; --amber:{amber}; --grey:{grey}; --ridge:#15202a; color-scheme:light; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0e1418; --card:#162027; --ink:#e8eef1; --muted:#97a6b2;
  --line:rgba(232,238,241,.13); --ridge:#e8eef1; color-scheme:dark; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:17px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:28px 16px 56px; }}
.kicker {{ text-transform:uppercase; letter-spacing:.14em; font-size:12px; font-weight:600; color:var(--muted); }}
.kicker b {{ color:var(--ink); }}
h1 {{ font-size:clamp(28px,4.8vw,50px); line-height:1.08; letter-spacing:-.015em; margin:.25em 0 .35em; font-weight:800; }}
h1 .g {{ color:var(--green); }} h1 .r {{ color:var(--red); }} h1 .a {{ color:var(--amber); }} h1 .m {{ color:var(--grey); }}
h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin:0 0 10px; }}
.lede {{ font-size:19px; color:var(--muted); margin:0 0 22px; max-width:60em; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:18px 20px; margin:14px 0; }}
.row {{ display:grid; grid-template-columns:1fr; gap:14px; }}
@media (min-width:880px) {{ .row {{ grid-template-columns:1fr 1.35fr; }} }}
.row .card {{ margin:0; }}
svg {{ width:100%; height:auto; display:block; }}
.tick {{ font-size:12px; fill:var(--muted); font-variant-numeric:tabular-nums; }}
.tag {{ font-size:12px; font-weight:700; fill:var(--red); paint-order:stroke; stroke:var(--card); stroke-width:4px; stroke-linejoin:round; }}
.pin {{ stroke:var(--red); stroke-width:1.2; }}
.grid {{ stroke:var(--line); stroke-width:1; }}
.ridge {{ stroke:var(--ridge); stroke-width:1.4; }}
.lane {{ fill:var(--line); }}
table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:15px; }}
th,td {{ text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); }}
th {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); font-weight:600; }}
td.x {{ font-weight:700; color:var(--red); }}
.bar {{ display:flex; height:44px; border-radius:10px; overflow:hidden; margin:6px 0 10px; }}
.bar div {{ display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; font-size:16px; min-width:0; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 18px; font-size:15px; margin-top:4px; }}
.legend i {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px 16px; }}
.stat b {{ display:block; font-size:30px; line-height:1.1; font-variant-numeric:tabular-nums; }}
.stat small {{ color:var(--muted); font-size:14px; }}
.note {{ color:var(--muted); font-size:14px; margin:.4em 0 0; }}
footer {{ color:var(--muted); font-size:13px; margin-top:26px; border-top:1px solid var(--line); padding-top:12px; }}
</style></head><body><main>
{body}
<footer>{credits}<br>A finding about a route or a ride, not training or medical advice. Wind is not counted.</footer>
</main></body></html>
"""


def _page(title: str, body: str) -> str:
    return PAGE.format(title=html.escape(title), body=body, credits=html.escape(CREDITS), green=GREEN, red=RED,
                       amber=AMBER, grey=GREY)


def _stat(big: str, small: str) -> str:
    return f'<div class="stat"><b>{big}</b><small>{small}</small></div>'


def write_plan_html(path, p, name: str, cond: str, prof, watts: bool = False) -> None:
    sh, lo, hi = p.shares, p.shares_low, p.shares_high
    st = p.stretches
    ends = np.array([s.end_m for s in st])
    idx = np.clip(np.searchsorted(ends, prof.dist), 0, len(st) - 1)
    ok = np.array([st[i].possible for i in idx]) if st else np.ones(prof.dist.size, bool)
    colour = [GREEN if o else RED for o in ok]
    marks = [(x.start_m / 1000 + x.length_m / 2000, f"{100 * x.mean_grade:.0f} % · {times(x.max_ratio)}")
             for x in sorted(p.passages, key=lambda x: -x.length_m)[:4]]
    keep = trim_mask(prof.dist, 2000.0) & np.isfinite(prof.lat)
    rows = "".join(
        f"<tr><td>km {x.start_m / 1000:.1f}</td><td>{x.length_m / 1000:.1f} km</td><td>{100 * x.mean_grade:.1f} %</td>"
        f"<td>{100 * x.max_grade:.1f} %</td><td class='x'>{times(x.max_ratio)}</td><td>{hm(x.time_s)}</td></tr>" for x in p.passages)
    table = (f'<div class="card"><h2>Where Zone 2 is impossible</h2><table><tr><th>From</th><th>Length</th><th>Grade</th>'
             f'<th>Steepest</th><th>Needs</th><th>At your slowest</th></tr>{rows}</table>'
             f'<p class="note">Needs: the power of your slowest pace there, as a multiple of your Zone 2 ceiling.</p></div>'
             if p.passages else '<div class="card"><h2>Where Zone 2 is impossible</h2><p>Nowhere on this route, in these conditions.</p></div>')
    imp_km = sum(x.length_m for x in p.passages) / 1000
    body = f"""
<header><div class="kicker">z2analyzer · before the ride · <b>{html.escape(name)}</b></div>
<h1><span class="g">{pct(sh.km_share)}</span> of the kilometres, but <span class="r">{pct(sh.time_share)}</span> of the time.</h1>
<p class="lede">That is how much of this route you can ride in Zone 2, even in your smallest gear at the slowest cadence you really hold. {html.escape(cond)}. Depending on the day: {pct(lo.km_share)} to {pct(hi.km_share)} of the kilometres, {pct(lo.time_share)} to {pct(hi.time_share)} of the time.</p></header>
<div class="stats">{_stat(f"{sh.km_total:.1f} km", "route")}{_stat(hm(sh.time_total_s), "riding easy")}
{_stat(str(len(p.passages)), "passages where Zone 2 is impossible")}{_stat(f"{imp_km:.1f} km", f"of them, {hm(sum(x.time_s for x in p.passages))} at your slowest")}</div>
<div class="card"><h2>Profile</h2>{_profile_svg(prof.dist / 1000.0, prof.elev, colour, marks)}
<div class="legend"><span><i style="background:var(--green)"></i>Zone 2 possible</span><span><i style="background:var(--red)"></i>impossible, even at your slowest</span></div></div>
<div class="row"><div class="card"><h2>Route</h2>{_shape_svg(prof.lat, prof.lon, colour, keep)}<p class="note">First and last 2 km left out.</p></div>
{table}</div>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(_page(f"z2analyzer · {name}", body))


def write_review_html(path, r, rd, z2_top: float, cool: float) -> None:
    tot = max(r.above_min, 1e-9)
    parts = [("the road", r.road_min, RED), ("the heat", r.heat_min, AMBER), ("me", r.me_min, GREY)]
    bar = "".join(f'<div style="width:{100 * m / tot:.2f}%;background:{c}">{m:.0f}</div>' for _, m, c in parts if m >= 0.5)
    legend = "".join(f'<span><i style="background:{c}"></i>{lab}: {m:.0f} min</span>' for lab, m, c in parts)
    prof = rd.profile
    ps = r.per_second
    d = rd.dist
    # Colour of the profile: what the road allowed that day; strip: the cause of the seconds above Zone 2 there.
    poss = np.interp(prof.dist, d, ps["possible"].astype(float)) >= 0.5
    colour = [GREEN if o else RED for o in poss]
    bins = np.searchsorted(prof.dist, d)
    strip = [None] * prof.dist.size
    for cause, c in (("me", GREY), ("heat", AMBER), ("road", RED)):
        hit = np.bincount(np.clip(bins[ps[cause]], 0, prof.dist.size - 1), minlength=prof.dist.size)
        for k in np.flatnonzero(hit):
            strip[k] = c
    when = datetime.fromtimestamp(rd.start_epoch, timezone.utc).astimezone()
    g = r.gear
    gear_html = ""
    if g.get("impossible_s") or g.get("possible_above_s"):
        a = g.get("impossible_smallest_gear_share", float("nan"))
        b = g.get("possible_above_easier_gear_left_share", float("nan"))
        gear_html = (f'<div class="card"><h2>The check that needs no model</h2><p>Where Zone 2 was impossible, the smallest '
                     f'gear was already in <b>{pct(a) if a == a else "–"}</b> of the time. Where it was possible and the heart '
                     f'was above it anyway, an easier gear was still there <b>{pct(b) if b == b else "–"}</b> of the time.</p>'
                     f'<p class="note">The gear comes from speed and cadence alone.</p></div>')
    level = ""
    if abs(r.day_offset_bpm) >= 0.5:
        level = (f' That day the heart ran {abs(r.day_offset_bpm):.0f} bpm {"above" if r.day_offset_bpm > 0 else "below"} '
                 f'its usual level at the same power, and the ceilings follow it.')
    body = f"""
<header><div class="kicker">z2analyzer · after the ride · <b>{when:%A %d %B %Y}</b></div>
<h1>{r.above_min:.0f} min above Zone 2: <span class="r">{r.road_min:.0f}</span> imposed by the road, <span class="a">{r.heat_min:.0f}</span> by the heat, <span class="m">{r.me_min:.0f}</span> mine.</h1>
<p class="lede">Zone 2 up to {z2_top:.0f} bpm. A minute above it goes to the road when even the slowest pace was above the ceiling in the cool ({cool:.0f} °C), to the heat when it was possible in the cool but not that day, and to me when it was possible.{level}</p></header>
<div class="card"><div class="bar">{bar}</div><div class="legend">{legend}</div>
<p class="note">Depending on the ceiling's uncertainty: road {r.road_range[0]:.0f}–{r.road_range[1]:.0f} min, heat {r.heat_range[0]:.0f}–{r.heat_range[1]:.0f}, me {r.me_range[0]:.0f}–{r.me_range[1]:.0f}.</p></div>
<div class="stats">{_stat(pct(r.shares.km_share), "of the kilometres could stay in Zone 2")}{_stat(pct(r.shares.time_share), "of the time")}
{_stat(f"{r.moving_min:.0f} min", "moving")}{_stat(f"{r.shares.km_total:.1f} km", "ridden")}</div>
<div class="card"><h2>Profile, and the minutes above Zone 2 along it</h2>{_profile_svg(prof.dist / 1000.0, prof.elev, colour, (), strip)}
<div class="legend"><span><b>Profile</b></span><span><i style="background:var(--green)"></i>Zone 2 possible that day</span><span><i style="background:var(--red)"></i>impossible</span></div>
<div class="legend"><span><b>Band: minutes above Zone 2</b></span><span><i style="background:var(--red)"></i>the road</span><span><i style="background:var(--amber)"></i>the heat</span><span><i style="background:var(--grey)"></i>me</span></div></div>
{gear_html}"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(_page(f"z2analyzer · ride of {when:%Y-%m-%d}", body))
