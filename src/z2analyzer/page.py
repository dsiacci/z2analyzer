"""The local pages: the route as a skyline at dawn, big numbers, the passages like the climbs of a roadbook.

One HTML file each, and nothing loaded from anywhere: the fonts (Saira, SIL
Open Font License) travel inside the page, the charts are inline SVG, and
their labels are HTML so they stay readable on a phone. The route's shape
leaves out its first and last 2 km.
"""

from __future__ import annotations

import base64
import html
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources

import numpy as np

from .geo import trim_mask

OK, NO, HEAT, ME = "#19a974", "#ff5a4f", "#ffae2b", "#8c9bb8"
CREDITS = ("Built with PriorLabs-TabPFN · terrain: IGN RGE ALTI® (Licence Ouverte 2.0) · bridges and tunnels: "
           "© OpenStreetMap contributors (ODbL) · weather: Open-Meteo, Météo-France models (CC BY 4.0) · "
           "fonts: Saira (SIL Open Font License)")
FACES = (("Saira Extra Condensed", 600, "SairaExtraCondensed-600.woff2"), ("Saira Extra Condensed", 800, "SairaExtraCondensed-800.woff2"),
         ("Saira Semi Condensed", 500, "SairaSemiCondensed-500.woff2"), ("Saira Semi Condensed", 700, "SairaSemiCondensed-700.woff2"))
e = html.escape


@lru_cache(maxsize=1)
def fonts_css() -> str:
    out = []
    for family, weight, name in FACES:
        data = resources.files("z2analyzer").joinpath("static", "fonts", name).read_bytes()
        out.append(f"@font-face{{font-family:'{family}';font-weight:{weight};font-style:normal;font-display:swap;"
                   f"src:url(data:font/woff2;base64,{base64.b64encode(data).decode()}) format('woff2')}}")
    return "\n".join(out)


CSS = """
/* Dawn over the island: night-blue sky to peach horizon, ink-dark ridges; green, coral, amber and slate carry the verdicts. */
:root { --paper:#eef1f6; --card:#ffffff; --ink:#121828; --muted:#5d6680; --line:rgba(18,24,40,.10); --soft:rgba(18,24,40,.05);
  --ok:#19a974; --no:#ff5a4f; --heat:#ffae2b; --me:#8c9bb8;
  --sky1:#0c1230; --sky2:#272255; --sky3:#6f3a66; --sky4:#ef8a5c; --sky5:#ffc98e; --ridge:#080c1b; --ridge2:#221c46;
  --display:"Saira Extra Condensed","Arial Narrow",system-ui,sans-serif;
  --ui:"Saira Semi Condensed",system-ui,-apple-system,"Segoe UI",sans-serif;
  --body:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; color-scheme:light; }
@media (prefers-color-scheme: dark) { :root { --paper:#080c18; --card:#111a2c; --ink:#eaeef7; --muted:#97a2bd;
  --line:rgba(234,238,247,.10); --soft:rgba(234,238,247,.05); color-scheme:dark; } }
* { box-sizing:border-box; }
html { -webkit-text-size-adjust:100%; }
body { margin:0; background:var(--paper); color:var(--ink); font:16px/1.5 var(--body); }
.hero { position:relative; height:clamp(300px, 62vw, 460px); overflow:hidden; color:#fff;
  background:radial-gradient(circle at 74% 84%, rgba(255,233,190,.95) 0 2.6%, rgba(255,196,130,.55) 5%, rgba(255,160,110,0) 16%),
    linear-gradient(180deg, var(--sky1) 0%, var(--sky2) 34%, var(--sky3) 62%, var(--sky4) 84%, var(--sky5) 100%); }
.hero::before { content:""; position:absolute; inset:0; opacity:.55;
  background-image:radial-gradient(1px 1px at 12% 18%, #fff 50%, transparent 52%), radial-gradient(1px 1px at 33% 9%, #fff 50%, transparent 52%),
    radial-gradient(1.4px 1.4px at 58% 15%, #fff 50%, transparent 52%), radial-gradient(1px 1px at 81% 7%, #fff 50%, transparent 52%),
    radial-gradient(1px 1px at 91% 24%, #fff 50%, transparent 52%), radial-gradient(1px 1px at 46% 26%, #fff 50%, transparent 52%); }
.hero svg { position:absolute; left:0; right:0; bottom:0; width:100%; height:50%; }
.hero .glow { filter:drop-shadow(0 0 5px rgba(255,120,90,.95)); }
.hero-text { position:relative; z-index:1; padding:max(22px, env(safe-area-inset-top)) max(18px, env(safe-area-inset-right)) 0 max(18px, env(safe-area-inset-left));
  max-width:1020px; margin:0 auto; }
.kicker { font:700 12.5px/1.3 var(--ui); letter-spacing:.16em; text-transform:uppercase; color:rgba(255,255,255,.72); }
.hero h1 { font:800 clamp(38px, 9vw, 72px)/.98 var(--display); margin:.18em 0 .16em; letter-spacing:.005em; text-wrap:balance; text-transform:uppercase; }
.hero .sub { font:500 clamp(15px, 3.6vw, 18px)/1.35 var(--ui); color:rgba(255,255,255,.82); }
main { position:relative; z-index:2; max-width:1020px; margin:-58px auto 0; padding:0 max(14px, env(safe-area-inset-right)) 40px max(14px, env(safe-area-inset-left)); }
.card { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:18px; margin:0 0 14px; box-shadow:0 10px 30px -18px rgba(8,12,27,.45); }
h2 { font:700 13px/1.2 var(--ui); letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:2px 0 14px;
  display:flex; align-items:center; gap:10px; }
h2 .count { font:800 15px/1 var(--display); color:#fff; background:var(--no); border-radius:999px; padding:3px 9px 2px; letter-spacing:.02em; }
.score { display:grid; grid-template-columns:1fr 1fr; gap:14px 18px; }
.metric .big { font:800 clamp(58px, 15vw, 96px)/.86 var(--display); letter-spacing:-.01em; font-variant-numeric:tabular-nums; }
.metric .big small { font-size:.5em; font-weight:600; margin-left:2px; color:var(--muted); }
.metric .lbl { font:700 15px/1.25 var(--ui); margin-top:4px; }
.track { height:10px; border-radius:999px; background:color-mix(in srgb, var(--no) 78%, transparent); overflow:hidden; margin-top:10px; }
.track i { display:block; height:100%; background:var(--ok); border-radius:999px 0 0 999px; }
.rng { font:500 13px/1.3 var(--ui); color:var(--muted); margin-top:6px; }
.lede { grid-column:1 / -1; margin:2px 0 0; color:var(--muted); font-size:15px; }
.lede b { color:var(--ink); }
.facts { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; grid-column:1 / -1; }
.chip { font:600 13.5px/1.25 var(--ui); padding:7px 12px 6px; border-radius:14px; background:var(--soft); border:1px solid var(--line); }
.chip.no { color:#fff; background:var(--no); border-color:transparent; }
.climbs { list-style:none; margin:0; padding:0; display:grid; gap:10px; grid-template-columns:repeat(auto-fill, minmax(min(100%, 290px), 1fr)); }
.climb { display:grid; grid-template-columns:auto 1fr auto; gap:4px 12px; align-items:center; padding:12px 14px; border-radius:14px;
  background:var(--soft); border:1px solid var(--line); min-width:0; }
.climb .n { width:30px; height:30px; border-radius:50%; display:grid; place-items:center; font:800 17px/1 var(--display);
  background:var(--ink); color:var(--card); }
.climb .where { font:700 16px/1.2 var(--ui); }
.climb .grade { font:500 13.5px/1.3 var(--ui); color:var(--muted); }
.climb .need { text-align:right; }
.climb .need b { display:block; font:800 30px/.9 var(--display); color:var(--no); font-variant-numeric:tabular-nums; }
.climb .need small { font:600 11.5px/1.2 var(--ui); color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
.climb .spark { grid-column:2 / 4; height:30px; width:100%; }
.climb .slow { grid-column:2 / 4; font:500 13px/1.2 var(--ui); color:var(--muted); }
.chart { position:relative; height:clamp(170px, 42vw, 250px); margin:6px 0 0 34px; }
.chart svg { position:absolute; inset:0; width:100%; height:100%; overflow:visible; }
.ylab { position:absolute; left:-34px; transform:translateY(-50%); font:600 11.5px/1 var(--ui); color:var(--muted); }
.xaxis { position:relative; height:18px; margin:4px 0 0 34px; }
.xaxis span { position:absolute; transform:translateX(-50%); font:600 11.5px/1 var(--ui); color:var(--muted); font-variant-numeric:tabular-nums; }
.xaxis span:last-child { transform:translateX(-100%); }
.flag { position:absolute; transform:translate(-50%, -125%); min-width:22px; height:22px; padding:0 6px; border-radius:999px; display:grid; place-items:center;
  font:800 14px/1 var(--display); color:#fff; background:var(--no); box-shadow:0 0 0 3px var(--card); }
.hline { position:absolute; left:0; right:0; border-top:2px dashed color-mix(in srgb, var(--ink) 55%, transparent); }
.hline span { position:absolute; right:0; top:-21px; font:700 12px/1 var(--ui); color:var(--ink); background:var(--card); padding:3px 6px; border-radius:6px; }
.band { position:relative; height:12px; margin:8px 0 0 34px; border-radius:4px; background:var(--soft); overflow:hidden; }
.band i { position:absolute; top:0; bottom:0; }
.legend { display:flex; flex-wrap:wrap; gap:6px 16px; font:600 13.5px/1.3 var(--ui); margin-top:12px; color:var(--muted); }
.legend i { display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.split { display:flex; height:66px; border-radius:14px; overflow:hidden; gap:3px; }
.split div { display:flex; flex-direction:column; justify-content:center; padding:0 12px; color:#fff; min-width:58px; }
.split b { font:800 34px/.9 var(--display); font-variant-numeric:tabular-nums; }
.split span { font:700 11.5px/1.2 var(--ui); text-transform:uppercase; letter-spacing:.08em; opacity:.92; }
.rings { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.ring { display:grid; grid-template-columns:76px 1fr; gap:12px; align-items:center; }
.ring svg { width:76px; height:76px; }
.ring p { margin:0; font:500 13.5px/1.35 var(--ui); color:var(--muted); }
.ring p b { display:block; font:700 15px/1.2 var(--ui); color:var(--ink); margin-bottom:2px; }
.two { display:grid; grid-template-columns:1fr; gap:14px; }
@media (min-width:860px) { .two { grid-template-columns:1fr 1fr; } .two .card { margin:0; } }
.shape svg { width:100%; max-width:560px; height:auto; display:block; margin:0 auto; }
.note { font:500 13px/1.4 var(--ui); color:var(--muted); margin:10px 0 0; }
footer { max-width:1020px; margin:0 auto; padding:6px max(16px, env(safe-area-inset-right)) max(28px, env(safe-area-inset-bottom)) max(16px, env(safe-area-inset-left));
  font:500 12.5px/1.5 var(--ui); color:var(--muted); }
footer b { font:800 18px/1 var(--display); color:var(--ink); letter-spacing:.02em; display:block; margin-bottom:6px; }
@media (max-width:520px) { .rings { grid-template-columns:1fr; } }
@media (max-width:420px) { .card { padding:15px; border-radius:16px; } .score { gap:12px; } .ring { grid-template-columns:64px 1fr; } .ring svg { width:64px; height:64px; } }
"""


def document(title: str, body: str) -> str:
    return (f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">"
            f"<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:\">"
            f"<meta name=\"theme-color\" content=\"#0c1230\"><title>{e(title)}</title>"
            f"<style>{fonts_css()}{CSS}</style></head><body>{body}"
            f"<footer><b>z2analyzer</b>{e(CREDITS)}<br>A finding about a route or a ride, not training or medical advice. "
            f"Wind is not counted.</footer></body></html>\n")


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def hm(seconds: float) -> str:
    m = int(round(seconds / 60))
    return f"{m // 60} h {m % 60:02d}" if m >= 60 else f"{m} min"


def times(r: float) -> str:
    return f"{r:.2f}×" if r < 1.1 else f"{r:.1f}×"


# ------------------------------------------------------------------ drawing


def _runs(values):
    """(start, end, value) for each run of equal consecutive values."""
    out, i, n = [], 0, len(values)
    while i < n:
        j = i
        while j + 1 < n and values[j + 1] == values[i]:
            j += 1
        out.append((i, j, values[i]))
        i = j + 1
    return out


def _smooth(z, k):
    if k <= 1 or len(z) < 3:
        return np.asarray(z, float)
    pad = np.pad(np.asarray(z, float), k, mode="edge")
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]


def skyline(dist_km, elev, ridge_colours) -> str:
    """The route's profile as mountains at dawn; `ridge_colours[i]` lights the ridge from point i to i+1 (None: dark)."""
    d = np.asarray(dist_km, float)
    z = np.asarray(elev, float)
    n = len(d)
    step = max(1, n // 900)
    idx = np.arange(0, n, step)
    W, H = 1000.0, 300.0
    lo, hi = float(z.min()), float(z.max())
    span = max(hi - lo, 60.0)
    X = (d[idx] - d[0]) / max(d[-1] - d[0], 1e-9) * W
    # Sixty kilometres on a phone's width turn every climb into a needle: soften the ridge, keep it low.
    Y = H - 10 - (_smooth(z[idx], max(2, len(idx) // 70)) - lo) / span * (H * 0.62)
    back = H - 30 - (_smooth(z[idx], max(3, len(idx) // 12)) - lo) / span * (H * 0.70)
    front = "M0," + f"{H:.0f} " + " ".join(f"L{x:.1f},{y:.1f}" for x, y in zip(X, Y)) + f" L{W:.0f},{H:.0f} Z"
    far = "M0," + f"{H:.0f} " + " ".join(f"L{x:.1f},{y:.1f}" for x, y in zip(X, back)) + f" L{W:.0f},{H:.0f} Z"
    cols = [ridge_colours[i] for i in idx]
    lit = []
    for a, b, c in _runs(cols):
        if c is None:
            continue
        b = min(b + 1, len(idx) - 1)
        pts = " ".join(f"{X[k]:.1f},{Y[k]:.1f}" for k in range(a, b + 1))
        lit.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="3.2" stroke-linecap="round" '
                   f'stroke-linejoin="round" vector-effect="non-scaling-stroke"/>')
    return (f'<svg viewBox="0 0 {W:.0f} {H:.0f}" preserveAspectRatio="none" aria-hidden="true">'
            f'<path d="{far}" fill="var(--ridge2)" opacity=".9"/><path d="{front}" fill="var(--ridge)"/></svg>'
            f'<svg class="glow" viewBox="0 0 {W:.0f} {H:.0f}" preserveAspectRatio="none" aria-hidden="true">{"".join(lit)}</svg>')


def hero(kicker: str, title: str, sub: str, dist_km, elev, ridge_colours) -> str:
    return (f'<header class="hero">{skyline(dist_km, elev, ridge_colours)}<div class="hero-text">'
            f'<div class="kicker">{kicker}</div><h1>{e(title)}</h1><div class="sub">{sub}</div></div></header>')


def roadbook(dist_km, elev, fills, flags=(), strip=None) -> str:
    """Elevation profile with HTML labels: `fills[i]` colours the stretch i..i+1; `flags` are (km, label)."""
    d = np.asarray(dist_km, float)
    z = np.asarray(elev, float)
    step = max(1, len(d) // 1200)
    idx = np.arange(0, len(d), step)
    W, H = 1000.0, 200.0
    lo, hi = float(z.min()), float(z.max())
    pad = max(8.0, 0.08 * (hi - lo))
    lo2, hi2 = lo - pad, hi + pad
    X = (d[idx] - d[0]) / max(d[-1] - d[0], 1e-9) * W
    Y = H - (z[idx] - lo2) / (hi2 - lo2) * H
    cols = [fills[i] for i in idx]
    polys = []
    for a, b, c in _runs(cols):
        b = min(b + 1, len(idx) - 1)
        pts = [f"{X[a]:.1f},{H:.0f}"] + [f"{X[k]:.1f},{Y[k]:.1f}" for k in range(a, b + 1)] + [f"{X[b]:.1f},{H:.0f}"]
        polys.append(f'<polygon points="{" ".join(pts)}" fill="{c}"/>')
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(X, Y))
    svg = (f'<svg viewBox="0 0 {W:.0f} {H:.0f}" preserveAspectRatio="none" aria-hidden="true">{"".join(polys)}'
           f'<polyline points="{line}" fill="none" stroke="var(--ink)" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>')
    ylabs = "".join(f'<div class="ylab" style="top:{100 * (1 - (v - lo2) / (hi2 - lo2)):.1f}%">{v:.0f} m</div>'
                    for v in (lo, (lo + hi) / 2, hi))
    fl = []
    placed: list[float] = []
    for km, label in flags:
        frac = (km - d[0]) / max(d[-1] - d[0], 1e-9)
        if any(abs(frac - q) < 0.075 for q in placed):
            continue  # too close to a flag already drawn: the cards below list every passage
        placed.append(frac)
        zk = float(np.interp(km, d, z))
        fl.append(f'<div class="flag" style="left:{100 * (km - d[0]) / max(d[-1] - d[0], 1e-9):.2f}%;'
                  f'top:{100 * (1 - (zk - lo2) / (hi2 - lo2)):.1f}%">{e(label)}</div>')
    total = d[-1] - d[0]
    tick = next(t for t in (1, 2, 5, 10, 20, 25, 50) if total / t <= 7)
    ticks = [k for k in np.arange(0, total + 1e-9, tick)]
    xs = "".join(f'<span style="left:{100 * k / max(total, 1e-9):.2f}%">{k:.0f}</span>' for k in ticks[:-1] if total - k > tick * 0.4)
    xs += f'<span style="left:100%">{total:.1f} km</span>'
    strip_html = ""
    if strip is not None:
        cells = []
        for a, b, c in _runs([strip[i] for i in idx]):
            if c is None:
                continue
            b = min(b + 1, len(idx) - 1)
            cells.append(f'<i style="left:{100 * X[a] / W:.2f}%;width:{max(100 * (X[b] - X[a]) / W, 0.25):.2f}%;background:{c}"></i>')
        strip_html = f'<div class="band">{"".join(cells)}</div>'
    return f'<div class="chart">{svg}{ylabs}{"".join(fl)}</div>{strip_html}<div class="xaxis">{xs}</div>'


def shape(lat, lon, colours, keep, marks=()) -> str:
    """The route's shape without its first and last 2 km, no map underneath; `marks` are (index, label)."""
    if keep.sum() < 2:
        return ""
    la, lo_ = np.asarray(lat)[keep], np.asarray(lon)[keep]
    k = np.cos(np.radians(np.mean(la)))
    x, y = lo_ * k, la
    W, H = 400.0, 300.0
    s = min((W - 40) / max(np.ptp(x), 1e-9), (H - 40) / max(np.ptp(y), 1e-9))
    X = 20 + (x - x.min()) * s + (W - 40 - np.ptp(x) * s) / 2
    Y = H - 20 - (y - y.min()) * s - (H - 40 - np.ptp(y) * s) / 2
    c = [colours[i] for i in np.flatnonzero(keep)]
    lines = []
    for a, b, col in _runs(c):
        b = min(b + 1, len(X) - 1)
        pts = " ".join(f"{X[m]:.1f},{Y[m]:.1f}" for m in range(a, b + 1))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
    kept = np.flatnonzero(keep)
    pins = []
    spots: list[tuple[float, float]] = []
    for i, label in marks:
        j = int(np.searchsorted(kept, i))
        if 0 <= j < len(kept) and abs(kept[min(j, len(kept) - 1)] - i) < 20:
            if any((X[j] - a) ** 2 + (Y[j] - b) ** 2 < 26 ** 2 for a, b in spots):
                continue  # would cover another pin
            spots.append((X[j], Y[j]))
            pins.append(f'<g><circle cx="{X[j]:.1f}" cy="{Y[j]:.1f}" r="11" fill="{NO}" stroke="var(--card)" stroke-width="3"/>'
                        f'<text x="{X[j]:.1f}" y="{Y[j] + 4.5:.1f}" text-anchor="middle" font-family="Saira Extra Condensed, sans-serif" '
                        f'font-weight="800" font-size="13" fill="#fff">{e(label)}</text></g>')
    return f'<svg viewBox="0 0 {W:.0f} {H:.0f}" role="img" aria-label="Route shape, ends left out">{"".join(lines)}{"".join(pins)}</svg>'


def sparkline(dist_km, elev, a_km, b_km, margin_km=0.3) -> str:
    d = np.asarray(dist_km, float)
    z = np.asarray(elev, float)
    m = (d >= a_km - margin_km) & (d <= b_km + margin_km)
    if m.sum() < 3:
        return ""
    dd, zz = d[m], z[m]
    W, H = 300.0, 40.0
    X = (dd - dd[0]) / max(dd[-1] - dd[0], 1e-9) * W
    Y = H - 3 - (zz - zz.min()) / max(np.ptp(zz), 1.0) * (H - 6)
    inside = (dd >= a_km) & (dd <= b_km)
    area = f"M{X[0]:.1f},{H:.0f} " + " ".join(f"L{x:.1f},{y:.1f}" for x, y in zip(X, Y)) + f" L{X[-1]:.1f},{H:.0f} Z"
    hot = " ".join(f"{x:.1f},{y:.1f}" for x, y, k in zip(X, Y, inside) if k)
    return (f'<svg class="spark" viewBox="0 0 {W:.0f} {H:.0f}" preserveAspectRatio="none" aria-hidden="true">'
            f'<path d="{area}" fill="var(--line)"/><polyline points="{hot}" fill="none" stroke="{NO}" stroke-width="3" '
            f'stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>')


def ring(share: float, colour: str) -> str:
    r, c = 30.0, 2 * np.pi * 30.0
    v = 0.0 if share != share else max(0.0, min(1.0, share))
    return (f'<svg viewBox="0 0 76 76" aria-hidden="true"><circle cx="38" cy="38" r="{r}" fill="none" stroke="var(--soft)" stroke-width="9"/>'
            f'<circle cx="38" cy="38" r="{r}" fill="none" stroke="{colour}" stroke-width="9" stroke-linecap="round" '
            f'stroke-dasharray="{c * v:.1f} {c:.1f}" transform="rotate(-90 38 38)"/>'
            f'<text x="38" y="44" text-anchor="middle" font-family="Saira Extra Condensed, sans-serif" font-weight="800" font-size="20" '
            f'fill="currentColor">{"–" if share != share else pct(share) + "%"}</text></svg>')


# ------------------------------------------------------------------ pages


def plan_page(p, name: str, cond: str, prof) -> str:
    sh, lo, hi = p.shares, p.shares_low, p.shares_high
    st = p.stretches
    ends = np.array([s.end_m for s in st])
    idx = np.clip(np.searchsorted(ends, prof.dist), 0, len(st) - 1)
    ok = np.array([st[i].possible for i in idx]) if st else np.ones(prof.dist.size, bool)
    passages = sorted(p.passages, key=lambda x: x.start_m)
    up = float(np.sum(np.clip(np.diff(prof.elev), 0, None)))
    d_km = prof.dist / 1000.0
    head = hero(f"Before the ride · {e(cond.split(',')[0]) if cond else ''}", name,
                f"{sh.km_total:.1f} km · {up:,.0f} m up · about {hm(sh.time_total_s)} riding easy".replace(",", " "),
                d_km, prof.elev, [None if o else NO for o in ok])
    score = f"""<section class="card score" aria-label="Share of the route in Zone 2">
<div class="metric"><div class="big">{pct(sh.km_share)}<small>%</small></div><div class="lbl">of the kilometres</div>
<div class="track"><i style="width:{100 * sh.km_share:.1f}%"></i></div><div class="rng">{pct(lo.km_share)}–{pct(hi.km_share)} % depending on the day</div></div>
<div class="metric"><div class="big">{pct(sh.time_share)}<small>%</small></div><div class="lbl">of the time</div>
<div class="track"><i style="width:{100 * sh.time_share:.1f}%"></i></div><div class="rng">{pct(lo.time_share)}–{pct(hi.time_share)} % depending on the day</div></div>
<p class="lede">That much of the route can stay in <b>Zone 2</b>, even in your smallest gear at the slowest cadence you really hold. Climbs are slow: they weigh more in time than in distance.</p>
<div class="facts"><span class="chip">{e(cond)}</span>{f'<span class="chip no">{len(passages)} passage{"s" if len(passages) != 1 else ""} impossible · {sum(x.length_m for x in passages) / 1000:.1f} km</span>' if passages else ''}</div>
</section>"""
    flags = [((x.start_m + x.end_m) / 2000, str(i + 1)) for i, x in enumerate(passages)]
    book = (f'<section class="card"><h2>Roadbook</h2>{roadbook(d_km, prof.elev, [OK if o else NO for o in ok], flags)}'
            f'<div class="legend"><span><i style="background:{OK}"></i>Zone 2 possible</span>'
            f'<span><i style="background:{NO}"></i>impossible, even at your slowest</span></div></section>')
    items = []
    for i, x in enumerate(passages):
        items.append(f"""<li class="climb"><div class="n">{i + 1}</div><div><div class="where">km {x.start_m / 1000:.1f} · {x.length_m / 1000:.1f} km</div>
<div class="grade">{100 * x.mean_grade:.1f} % average, up to {100 * x.max_grade:.1f} %</div></div>
<div class="need"><b>{times(x.max_ratio)}</b><small>the ceiling</small></div>{sparkline(d_km, prof.elev, x.start_m / 1000, x.end_m / 1000)}
<div class="slow">{hm(x.time_s)} at your slowest pace</div></li>""")
    climbs = (f'<section class="card"><h2>Where Zone 2 is impossible <span class="count">{len(passages)}</span></h2>'
              f'<ol class="climbs">{"".join(items)}</ol><p class="note">The multiple is the power of your slowest pace there, '
              f'divided by your Zone 2 ceiling for the temperature and the hours ridden by then.</p></section>' if passages else
              '<section class="card"><h2>Where Zone 2 is impossible</h2><p>Nowhere on this route, in these conditions.</p></section>')
    keep = trim_mask(prof.dist, 2000.0) & np.isfinite(prof.lat)
    mid_idx = [(int(np.searchsorted(prof.dist, (x.start_m + x.end_m) / 2)), str(i + 1)) for i, x in enumerate(passages)]
    route = (f'<section class="card shape"><h2>The route</h2>{shape(prof.lat, prof.lon, [OK if o else NO for o in ok], keep, mid_idx)}'
             f'<p class="note">The first and last 2 km are left out, and no map is loaded.</p></section>')
    body = f"{head}<main>{score}{book}{climbs}{route}</main>"
    return document(f"z2analyzer · {name}", body)


def review_page(r, rd, z2_top: float, cool: float) -> str:
    prof = rd.profile
    ps = r.per_second
    d_km = prof.dist / 1000.0
    bins = np.clip(np.searchsorted(prof.dist, rd.dist), 0, prof.dist.size - 1)
    cause = [None] * prof.dist.size
    for name, col in (("me", ME), ("heat", HEAT), ("road", NO)):
        for k in np.flatnonzero(np.bincount(bins[ps[name]], minlength=prof.dist.size)):
            cause[k] = col
    poss = np.interp(prof.dist, rd.dist, ps["possible"].astype(float)) >= 0.5
    when = datetime.fromtimestamp(rd.start_epoch, timezone.utc).astimezone()
    air = float(np.nanmedian(rd.temp_air[rd.moving]))
    head = hero(f"After the ride · {when:%a %d %b %Y} · {when:%H:%M}", f"{r.above_min:.0f} min above Zone 2",
                f"{r.shares.km_total:.1f} km · {r.moving_min:.0f} min moving · air {air:.0f} °C · Zone 2 up to {z2_top:.0f} bpm",
                d_km, prof.elev, cause)
    tot = max(r.above_min, 1e-9)
    parts = [("the road", r.road_min, NO), ("the heat", r.heat_min, HEAT), ("me", r.me_min, ME)]
    split = "".join(f'<div style="flex:{max(m, 0.001):.3f};background:{c}"><b>{m:.0f}</b><span>{lab}</span></div>'
                    for lab, m, c in parts if m >= 0.5)
    level = ""
    if abs(r.day_offset_bpm) >= 0.5:
        level = (f'<span class="chip">heart {r.day_offset_bpm:+.0f} bpm vs usual that day</span>')
    score = f"""<section class="card" aria-label="Who made the minutes above Zone 2">
<h2>Who made the {r.above_min:.0f} minutes</h2><div class="split">{split}</div>
<p class="lede" style="margin-top:12px"><b>The road</b>: even the slowest pace was above the ceiling, even in the cool ({cool:.0f} °C).
<b>The heat</b>: possible in the cool, not that day. <b>Me</b>: Zone 2 was possible there.</p>
<div class="facts"><span class="chip">road {r.road_range[0]:.0f}–{r.road_range[1]:.0f} · heat {r.heat_range[0]:.0f}–{r.heat_range[1]:.0f} · me {r.me_range[0]:.0f}–{r.me_range[1]:.0f} min</span>{level}</div></section>"""
    heart = _heart_card(r, rd, z2_top)
    book = (f'<section class="card"><h2>Along the road</h2>{roadbook(d_km, prof.elev, [OK if o else NO for o in poss], (), cause)}'
            f'<div class="legend"><span><i style="background:{OK}"></i>Zone 2 possible that day</span><span><i style="background:{NO}"></i>impossible</span></div>'
            f'<div class="legend"><span>Band, above Zone 2:</span><span><i style="background:{NO}"></i>the road</span>'
            f'<span><i style="background:{HEAT}"></i>the heat</span><span><i style="background:{ME}"></i>me</span></div></section>')
    g = r.gear
    gear = (f"""<section class="card"><h2>The check that needs no model</h2><div class="rings">
<div class="ring">{ring(g.get("impossible_smallest_gear_share", float("nan")), NO)}<p><b>Smallest gear already in</b>where Zone 2 was impossible</p></div>
<div class="ring">{ring(g.get("possible_above_easier_gear_left_share", float("nan")), ME)}<p><b>An easier gear left</b>where it was possible and the heart above</p></div>
</div><p class="note">The gear comes from speed and cadence alone. Zone 2 was possible on {pct(r.shares.km_share)} % of the kilometres, {pct(r.shares.time_share)} % of the time.</p></section>""")
    body = f'{head}<main>{score}{heart}{book}{gear}</main>'
    return document(f"z2analyzer · ride of {when:%Y-%m-%d}", body)


def _heart_card(r, rd, z2_top: float) -> str:
    """Heart rate over the ride, the top of Zone 2 dashed, the minutes above it coloured by cause."""
    hr = np.asarray(rd.hr, float)
    n = hr.size
    if n < 120 or not np.isfinite(hr).any():
        return ""
    k = 30
    good = np.isfinite(hr)
    hr_f = np.interp(np.arange(n), np.flatnonzero(good), hr[good])
    sm = np.convolve(np.pad(hr_f, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    step = max(1, n // 600)
    idx = np.arange(0, n, step)
    lo = float(min(np.percentile(sm, 1), z2_top - 25))
    hi = float(max(np.percentile(sm, 99.5), z2_top + 15))
    W, H = 1000.0, 200.0
    X = idx / max(n - 1, 1) * W
    Y = H - (sm[idx] - lo) / (hi - lo) * H
    ytop = H - (z2_top - lo) / (hi - lo) * H
    ps = r.per_second
    col = np.full(n, None, dtype=object)
    for name, c in (("me", ME), ("heat", HEAT), ("road", NO)):
        col[np.asarray(ps[name], bool)] = c
    fills = []
    for a, b, c in _runs(list(col[idx])):
        if c is None:
            continue
        b = min(b + 1, len(idx) - 1)
        pts = [f"{X[a]:.1f},{ytop:.1f}"] + [f"{X[m]:.1f},{min(Y[m], ytop):.1f}" for m in range(a, b + 1)] + [f"{X[b]:.1f},{ytop:.1f}"]
        fills.append(f'<polygon points="{" ".join(pts)}" fill="{c}" opacity=".9"/>')
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(X, Y))
    svg = (f'<svg viewBox="0 0 {W:.0f} {H:.0f}" preserveAspectRatio="none" aria-hidden="true">{"".join(fills)}'
           f'<polyline points="{line}" fill="none" stroke="var(--ink)" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>')
    ylabs = "".join(f'<div class="ylab" style="top:{100 * (1 - (v - lo) / (hi - lo)):.1f}%">{v:.0f}</div>'
                    for v in (round(lo / 10) * 10 + 10, round(hi / 10) * 10 - 10) if lo < v < hi)
    hline = f'<div class="hline" style="top:{100 * ytop / H:.1f}%"><span>Zone 2 top · {z2_top:.0f} bpm</span></div>'
    minutes = n / 60.0
    tick = next(t for t in (15, 30, 60, 90) if minutes / t <= 6)
    xs = "".join(f'<span style="left:{100 * m / minutes:.2f}%">{m:.0f}</span>' for m in np.arange(0, minutes - tick * 0.4, tick))
    xs += f'<span style="left:100%">{minutes:.0f} min</span>'
    return (f'<section class="card"><h2>Heart rate, and why it was above</h2><div class="chart">{svg}{ylabs}{hline}</div>'
            f'<div class="xaxis">{xs}</div><div class="legend"><span><i style="background:{NO}"></i>the road</span>'
            f'<span><i style="background:{HEAT}"></i>the heat</span><span><i style="background:{ME}"></i>me</span>'
            f'<span>bpm, smoothed over a minute</span></div></section>')


GALLERY_CSS = """
.list { display:grid; gap:10px; }
@media (min-width:760px) { .list { grid-template-columns:1fr 1fr; } }
a.tile { display:block; color:inherit; text-decoration:none; background:var(--card); border:1px solid var(--line); border-radius:16px; padding:14px 16px;
  box-shadow:0 10px 26px -20px rgba(8,12,27,.5); }
a.tile:focus-visible, a.tile:hover { border-color:color-mix(in srgb, var(--ink) 35%, transparent); }
.tile .k { font:700 12px/1.2 var(--ui); letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }
.tile .t { font:800 27px/1 var(--display); margin:6px 0 4px; text-transform:uppercase; letter-spacing:.01em; }
.tile .b { font:500 14.5px/1.35 var(--ui); color:var(--muted); }
.tile .b em { font-style:normal; font-weight:700; color:var(--ink); }
.mini { display:flex; height:8px; border-radius:999px; overflow:hidden; margin-top:10px; gap:2px; }
section.group h2 { margin:20px 2px 10px; }
"""


def gallery_page(title: str, kicker: str, sub: str, groups, skyline_of=None) -> str:
    """An index: `groups` are (heading, tiles); a tile is (href, kicker, title, body_html, bar) with bar [(value, colour)]."""
    if skyline_of is not None:
        d_km, elev = skyline_of
        head = hero(e(kicker), title, sub, d_km, elev, [None] * len(d_km))
    else:
        head = f'<header class="hero" style="height:220px"><div class="hero-text"><div class="kicker">{e(kicker)}</div><h1>{e(title)}</h1><div class="sub">{sub}</div></div></header>'
    parts = []
    for heading, tiles in groups:
        items = []
        for href, k, t, b, bar in tiles:
            mini = "".join(f'<i style="flex:{max(v, 0.001):.3f};background:{c}"></i>' for v, c in (bar or []) if v > 0)
            items.append(f'<a class="tile" href="{e(href)}"><div class="k">{e(k)}</div><div class="t">{e(t)}</div>'
                         f'<div class="b">{b}</div>{f"<div class=mini>{mini}</div>" if mini else ""}</a>')
        parts.append(f'<section class="group"><h2>{e(heading)}</h2><div class="list">{"".join(items)}</div></section>')
    body = f'{head}<main style="margin-top:6px">{"".join(parts)}</main>'
    return document(title, body).replace("</style>", GALLERY_CSS + "</style>", 1)
