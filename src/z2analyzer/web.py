"""The web app: start a new analysis, read its result a few seconds later, find every earlier one.

    z2analyzer serve --limits limits.json --terrain dem/ --structures bridges.json --z2-top 145
    # then open http://127.0.0.1:8765/
    z2analyzer serve ... --watch ~/rides   # and every new ride that lands there is judged on its own

A web service you run yourself: the pages come from your own machine, the
files you drop stay in its data folder (`--data`), and no page loads anything
from the internet. As with the command line, the only thing about a route or a
ride that goes out is one rounded weather point (`weather.py`). Each analysis
keeps a folder of its own: the file dropped (`in/`), `meta.json`, the verdict
as `result.json` and its page, the same page `plan --html` and `review --html`
write. The analyses run one at a time, in a background thread.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import queue
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .geo import trim_mask
from .page import CREDITS, CSS, ENDS_M, GALLERY_CSS, HEAT, ME, NO, OK, fonts_css, skyline

e = html.escape
ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")
MAX_UPLOAD = 32 * 1024 * 1024
KIND = {"plan": "Before a ride", "review": "After a ride"}
STEPS = {
    "plan": ("Reading the route", "Terrain profile (IGN RGE ALTI)", "Temperature along the route",
             "Zone 2, stretch by stretch"),
    "review": ("Reading the ride", "Air temperature during the ride", "Terrain profile (IGN RGE ALTI)",
               "Who made the minutes above Zone 2"),
}

# Shows the name of the file chosen or dropped (the input itself covers the drop zone, so dropping works without it).
SCRIPT = ("document.querySelectorAll('.drop input').forEach(function(i){i.addEventListener('change',function(){"
          "var z=i.closest('.drop');z.querySelector('.name').textContent=i.files.length?i.files[0].name:'';"
          "z.classList.toggle('has',i.files.length>0);});});")
SCRIPT_HASH = base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()

APP_CSS = """
.appbar { display:flex; align-items:center; gap:10px; padding:10px max(16px, env(safe-area-inset-right)) 10px max(16px, env(safe-area-inset-left));
  background:var(--sky1); color:#fff; position:relative; z-index:3; }
.appbar .brand { font:800 23px/1 var(--display); letter-spacing:.02em; color:#fff; text-decoration:none; margin-right:auto; }
.btn { display:inline-flex; align-items:center; justify-content:center; gap:8px; font:700 15px/1 var(--ui); letter-spacing:.02em;
  color:#121828; background:var(--sky5); border:0; border-radius:999px; padding:11px 17px; text-decoration:none; cursor:pointer; white-space:nowrap; }
.btn.ghost { background:rgba(255,255,255,.13); color:#fff; }
.btn.big { font-size:18px; padding:15px 26px; }
.btn.line { background:transparent; color:var(--ink); box-shadow:inset 0 0 0 2px var(--line); }
.hero.short { height:auto; min-height:220px; padding-bottom:78px; }
.hero .btn.big { margin-top:20px; }
.home .sub { max-width:640px; }
.form { display:flex; flex-direction:column; gap:14px; }
.form h3 { font:800 34px/1 var(--display); text-transform:uppercase; margin:2px 0 0; letter-spacing:.01em; }
.form p { margin:0; color:var(--muted); font:500 15.5px/1.4 var(--ui); }
.form .k { font:700 12.5px/1.2 var(--ui); letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
.drop { position:relative; display:grid; place-items:center; gap:4px; text-align:center; min-height:132px; padding:18px; border-radius:16px;
  border:2px dashed color-mix(in srgb, var(--ink) 28%, transparent); background:var(--soft); cursor:pointer; }
.drop input { position:absolute; inset:0; width:100%; height:100%; opacity:0; cursor:pointer; }
.drop .what { font:700 16px/1.3 var(--ui); }
.drop .hint { font:500 13px/1.3 var(--ui); color:var(--muted); }
.drop .name { font:800 24px/1.1 var(--display); color:var(--ok); word-break:break-all; }
.drop.has { border-style:solid; border-color:var(--ok); background:color-mix(in srgb, var(--ok) 8%, var(--card)); }
.drop.has .what, .drop.has .hint { display:none; }
.field { display:flex; flex-direction:column; gap:5px; font:700 13px/1.2 var(--ui); letter-spacing:.06em; text-transform:uppercase; color:var(--muted); flex:1; }
.field input { font:600 17px/1.2 var(--ui); color:var(--ink); background:var(--card); border:1px solid var(--line); border-radius:12px; padding:11px 12px;
  letter-spacing:0; text-transform:none; width:100%; }
.row { display:flex; gap:12px; flex-wrap:wrap; }
.conds { border:0; margin:0; padding:0; display:flex; flex-direction:column; gap:8px; font:600 15px/1.3 var(--ui); }
.conds legend { font:700 13px/1.2 var(--ui); letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin-bottom:6px; padding:0; }
.conds label { display:flex; gap:9px; align-items:center; }
.form .btn.big { align-self:flex-start; margin-top:4px; }
.steps { list-style:none; margin:0; padding:0; display:grid; gap:12px; font:600 18px/1.3 var(--ui); }
.steps li { display:flex; gap:12px; align-items:center; color:var(--muted); opacity:.55; }
.steps li::before { content:""; width:22px; height:22px; border-radius:50%; border:3px solid var(--line); flex:none; }
.steps li.done { color:var(--ink); opacity:1; } .steps li.done::before { background:var(--ok); border-color:var(--ok); content:"✓"; color:#fff; display:grid; place-items:center; font:800 13px/1 var(--ui); }
.steps li.now { color:var(--ink); opacity:1; } .steps li.now::before { border-color:var(--ok) var(--ok) var(--line) var(--line); animation:spin .8s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
.tile.running .t, .tile.failed .t { color:var(--muted); }
.tile .src { font:700 12.5px/1.3 var(--ui); color:var(--ok); margin-top:9px; letter-spacing:.02em; }
h2 .count.n { background:var(--ink); color:var(--card); }
.total .sum { font:800 clamp(30px, 6vw, 44px)/1 var(--display); text-transform:uppercase; margin:0 0 14px; }
.total .sum b { font-size:1.35em; }
.about { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.about form { margin-left:auto; }
.error { font:600 17px/1.4 var(--ui); }
"""

FOOTER = (f"<footer><b>z2analyzer</b>Runs on this machine: the files you drop stay in its data folder. {e(CREDITS)}<br>"
          f"A finding about a route or a ride, not training or medical advice. Wind is not counted.</footer>")


# ------------------------------------------------------------------ the analyses on disk


class Store:
    """One folder per analysis under `root/analyses/<id>/`."""

    def __init__(self, root: str | Path):
        self.root = Path(root) / "analyses"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def folder(self, aid: str) -> Path:
        if not ID.fullmatch(aid or ""):
            raise KeyError(aid)
        return self.root / aid

    def create(self, kind: str, filename: str, data: bytes, params: dict) -> dict:
        aid = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        f = self.folder(aid)
        (f / "in").mkdir(parents=True)
        name = safe_name(filename)
        (f / "in" / name).write_bytes(data)
        meta = {"id": aid, "kind": kind, "file": name, "params": params, "status": "queued", "step": 0,
                "created": datetime.now(timezone.utc).isoformat(timespec="microseconds")}
        self.save(meta)
        return meta

    def save(self, meta: dict) -> None:
        f = self.folder(meta["id"])
        tmp = f / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        tmp.replace(f / "meta.json")

    def get(self, aid: str) -> dict | None:
        try:
            return json.loads((self.folder(aid) / "meta.json").read_text(encoding="utf-8"))
        except (KeyError, OSError, ValueError):
            return None

    def update(self, aid: str, **fields) -> dict:
        with self.lock:
            meta = self.get(aid)
            if meta is None:
                raise KeyError(aid)
            meta.update(fields)
            self.save(meta)
            return meta

    def all(self) -> list[dict]:
        out = [m for m in (self.get(p.name) for p in self.root.iterdir() if p.is_dir()) if m]
        return sorted(out, key=lambda m: (m["created"], m["id"]), reverse=True)

    def delete(self, aid: str) -> None:
        shutil.rmtree(self.folder(aid), ignore_errors=True)


def safe_name(filename: str) -> str:
    base = Path((filename or "").replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip(".-") or "upload"
    return base[-80:]


def looks_like(data: bytes) -> str | None:
    """'fit', 'gpx' or None, from the first bytes."""
    if len(data) >= 12 and data[8:12] == b".FIT":
        return "fit"
    if b"<gpx" in data[:4096].lower():
        return "gpx"
    return None


# ------------------------------------------------------------------ one analysis


@dataclass
class Settings:
    limits: str
    terrain: str
    structures: str | None
    z2_top: float
    tz: str = "Europe/Paris"
    edge_temp: bool = False
    watch: str | None = None  # a folder where FIT files land: each new ride is judged on its own
    every_min: float = 5.0


def run(store: Store, aid: str, settings: Settings, rider, terrain, structures, limits_path: str | None = None) -> dict:
    """One analysis, from the file dropped to its page. `limits_path` judges with other limits than the settings'."""
    from .limits import Limits

    meta = store.update(aid, status="running", step=0, error=None)
    lim = Limits.load(limits_path or settings.limits)
    src = store.folder(aid) / "in" / meta["file"]
    try:
        if meta["kind"] == "plan":
            summary, page, result, sky = _plan(store, aid, meta, src, lim, settings, rider, terrain, structures)
        else:
            summary, page, result, sky = _review(store, aid, src, lim, settings, rider, terrain, structures)
    except Exception as ex:  # the page says what went wrong, in words
        return store.update(aid, status="failed", error=friendly(ex))
    f = store.folder(aid)
    (f / "page.html").write_text(page, encoding="utf-8")
    (f / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (f / "skyline.json").write_text(json.dumps(sky), encoding="utf-8")
    n = lim.notes or {}
    limits = {"source": lim.source, "rides": len(n.get("rides", [])), "learned": n.get("learned")}
    return store.update(aid, status="done", step=len(STEPS[meta["kind"]]), summary=summary, limits=limits,
                        finished=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _plan(store, aid, meta, src, lim, settings, rider, terrain, structures):
    from .cli import _plan_json, plan_conditions
    from .page import plan_page
    from .profile import build_profile
    from .report import plan_text
    from .verdict import plan

    with open(src, "rb") as f:
        first = f.read(4096)
    if looks_like(first) == "fit":  # a ride done before, planned again
        from .fitfile import read_fit

        ride = read_fit(src)
        ok = np.isfinite(ride.lat) & np.isfinite(ride.lon)
        lat, lon, name = ride.lat[ok], ride.lon[ok], ride.name
    else:
        from .gpx import read_gpx

        r = read_gpx(src)
        lat, lon, name = r.lat, r.lon, r.name
    if lat.size < 2:
        raise ValueError("this file has no positions")
    pr = meta["params"]
    name = pr.get("name") or name
    store.update(aid, step=1, name=name)
    covered(terrain, lat, lon)
    prof = build_profile(lat, lon, terrain, structures)
    store.update(aid, step=2)
    cond_kind = pr.get("conditions", "forecast")
    temp, cond = plan_conditions(prof, lim, pr.get("when") if cond_kind == "forecast" else None, settings.tz,
                                 None, cond_kind == "cool", cond_kind == "hot")
    store.update(aid, step=3)
    p = plan(prof, rider, lim, temp, top_bpm=settings.z2_top)
    sh = p.shares
    summary = {"title": name, "lines": plan_text(p, name, cond, prof).splitlines()[:2], "conditions": cond,
               "km_share": sh.km_share, "time_share": sh.time_share, "km_total": sh.km_total,
               "passages": len(p.passages), "passages_km": sum(x.length_m for x in p.passages) / 1000}
    st = p.stretches
    ends = np.array([s.end_m for s in st])
    idx = np.clip(np.searchsorted(ends, prof.dist), 0, len(st) - 1)
    lit = [not st[i].possible for i in idx] if st else [False] * prof.dist.size
    return summary, plan_page(p, name, cond, prof), _plan_json(p, name, cond, False), sky_of(prof.dist, prof.elev, lit)


def _review(store, aid, src, lim, settings, rider, terrain, structures):
    from .cli import _review_json, outdoor_ride, ride_weather
    from .dataset import ride_data
    from .fitfile import read_fit
    from .page import review_page
    from .report import review_text
    from .verdict import review

    ride = outdoor_ride(read_fit(src))
    when = ride.start.astimezone()
    store.update(aid, step=1, name=f"Ride of {when:%a %d %b %Y}", ride_start=when.isoformat(timespec="minutes"))
    temp_fn = None if settings.edge_temp else ride_weather(ride)
    store.update(aid, step=2)
    ok = np.isfinite(ride.lat) & np.isfinite(ride.lon)
    covered(terrain, ride.lat[ok], ride.lon[ok])
    rd = ride_data(ride, terrain, structures, temp_fn)
    store.update(aid, step=3)
    cool = float((lim.notes or {}).get("cool_c", 15.0))
    r = review(dist_m=rd.dist, speed_ms=rd.speed, hr=rd.hr, cadence_rpm=rd.cadence, moving=rd.moving,
               temp_air_c=rd.temp_air, profile=rd.profile, rider=rider, limits=lim, z2_top_bpm=settings.z2_top,
               cool_temp_c=cool, power_w=rd.power)
    summary = {"title": f"{r.above_min:.0f} min above Zone 2", "lines": review_text(r, rd, settings.z2_top, cool).splitlines()[:2],
               "above": r.above_min, "road": r.road_min, "heat": r.heat_min, "me": r.me_min, "moving": r.moving_min,
               "km_share": r.shares.km_share, "time_share": r.shares.time_share, "km_total": r.shares.km_total}
    prof = rd.profile
    return (summary, review_page(r, rd, settings.z2_top, cool), _review_json(r, rd, settings.z2_top, cool),
            sky_of(prof.dist, prof.elev, [False] * prof.dist.size))


def covered(terrain, lat, lon) -> None:
    """Refuse a track that leaves the terrain tiles, rather than guess its slopes."""
    k = max(1, len(lat) // 400)
    z = np.asarray(terrain(np.asarray(lat)[::k], np.asarray(lon)[::k]), float)
    if np.mean(~np.isfinite(z)) > 0.02:
        raise ValueError("part of this track is outside your terrain tiles: add the IGN RGE ALTI tiles of that area "
                         "to the terrain folder")


def sky_of(dist_m, elev, lit) -> dict:
    """The profile for the home page's skyline, without its first and last 2 km."""
    keep = trim_mask(np.asarray(dist_m, float), ENDS_M)
    d, z = np.asarray(dist_m, float)[keep] / 1000.0, np.asarray(elev, float)[keep]
    lit = np.asarray(lit, bool)[keep]
    step = max(1, d.size // 600)
    return {"km": np.round(d[::step], 3).tolist(), "elev": np.round(z[::step], 1).tolist(), "lit": lit[::step].tolist()}


HORIZON = ("Météo-France has no forecast for that start yet (it reaches about four days ahead). "
           "Pick a closer day, or the cool of your rides, or a hot day.")


def friendly(ex: Exception) -> str:
    from urllib.error import HTTPError

    msg = str(ex).strip() or type(ex).__name__
    if "has no temperature" in msg or (isinstance(ex, HTTPError) and ex.code == 400):
        return HORIZON
    if isinstance(ex, OSError) and not isinstance(ex, FileNotFoundError):
        return f"The weather service could not be reached ({msg}). Try again, or pick the cool of your rides."
    return msg[:1].upper() + msg[1:] + ("" if msg.endswith(".") else ".")


def watch_folder(store: Store, folder: str | Path, seen_file: str | Path, settle_s: float = 10.0) -> list[dict]:
    """New analyses for the rides that landed in `folder` since the last look.

    Indoor rides and rides without power are left out. A file changed in the
    last `settle_s` seconds (still being copied) waits for the next look; a file
    that changes later is looked at again.
    """
    from .fitfile import read_fit

    seen_file = Path(seen_file)
    seen = json.loads(seen_file.read_text(encoding="utf-8")) if seen_file.exists() else {}
    out, now = [], time.time()
    for f in sorted(Path(folder).iterdir(), key=lambda p: p.stat().st_mtime):
        if not f.is_file() or f.suffix.lower() != ".fit":
            continue
        st = f.stat()
        stamp = f"{st.st_size}:{int(st.st_mtime)}"
        if seen.get(f.name) == stamp or now - st.st_mtime < settle_s:
            continue
        seen[f.name] = stamp
        try:
            ride = read_fit(f)
        except Exception:  # not a FIT file after all, or a broken one: nothing to judge
            continue
        if ride.is_virtual or not ride.has_power:
            continue
        out.append(store.create("review", f.name, f.read_bytes(), {"source": "folder"}))
    seen_file.write_text(json.dumps(seen, indent=1), encoding="utf-8")
    return out


class Analyst:
    """Runs the analyses one at a time in a background thread: they are short and CPU-bound."""

    def __init__(self, store: Store, settings: Settings, rider):
        self.store, self.settings, self.rider = store, settings, rider
        self._terrain = self._structures = None
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()
        for m in reversed(store.all()):  # interrupted by a restart: run them again
            if m["status"] in ("queued", "running"):
                self.q.put(m["id"])

    def submit(self, aid: str) -> None:
        self.q.put(aid)

    def stop(self) -> None:
        self.q.put(None)

    def _loop(self) -> None:
        while (aid := self.q.get()) is not None:
            try:
                run(self.store, aid, self.settings, self.rider, *self._tools())
            except Exception as ex:  # the terrain folder itself, for instance
                if self.store.get(aid):
                    self.store.update(aid, status="failed", error=friendly(ex))

    def _tools(self):
        if self._terrain is None:
            from .dem import Terrain
            from .structures import Structures

            self._terrain = Terrain(self.settings.terrain)
            self._structures = Structures(self.settings.structures) if self.settings.structures else None
        return self._terrain, self._structures


# ------------------------------------------------------------------ pages


def document(title: str, body: str, refresh: bool = False) -> str:
    csp = (f"default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:; "
           f"script-src 'sha256-{SCRIPT_HASH}'; form-action 'self'; base-uri 'none'")
    return (f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">"
            f"<meta http-equiv=\"Content-Security-Policy\" content=\"{csp}\">"
            + ("<meta http-equiv=\"refresh\" content=\"1\">" if refresh else "")
            + f"<meta name=\"theme-color\" content=\"#0c1230\"><title>{e(title)}</title>"
            f"<style>{fonts_css()}{CSS}{GALLERY_CSS}{APP_CSS}</style></head><body>{appbar()}{body}{FOOTER}"
            f"<script>{SCRIPT}</script></body></html>\n")


def appbar() -> str:
    return ('<nav class="appbar"><a class="brand" href="/">z2analyzer</a><a class="btn ghost" href="/">Your analyses</a>'
            '<a class="btn" href="/new">New analysis</a></nav>')


def short_hero(kicker: str, title: str, sub: str) -> str:
    return (f'<header class="hero short"><div class="hero-text"><div class="kicker">{e(kicker)}</div><h1>{e(title)}</h1>'
            f'<div class="sub">{sub}</div></div></header>')


def _when(meta: dict) -> str:
    """The kicker of an analysis: its kind and its date."""
    pr = meta.get("params", {})
    if meta["kind"] == "review":
        rs = meta.get("ride_start")
        return KIND["review"] + (f" · {datetime.fromisoformat(rs):%a %d %b %Y, %H:%M}" if rs else "")
    c = pr.get("conditions", "forecast")
    if c == "forecast" and pr.get("when"):
        return f"{KIND['plan']} · {datetime.strptime(pr['when'], '%Y-%m-%d %H:%M'):%a %d %b %Y, %H:%M}"
    return f"{KIND['plan']} · " + ("a hot day, 30 °C" if c == "hot" else "the cool of your rides")


def tile(meta: dict) -> str:
    href, k = f"/a/{meta['id']}", _when(meta)
    s = meta.get("summary") or {}
    if meta["status"] in ("queued", "running"):
        return (f'<a class="tile running" href="{href}"><div class="k">{e(k)}</div><div class="t">Analysing…</div>'
                f'<div class="b">{e(meta["file"])}</div></a>')
    if meta["status"] == "failed":
        return (f'<a class="tile failed" href="{href}"><div class="k">{e(k)}</div><div class="t">Not analysed</div>'
                f'<div class="b">{e(meta["file"])}: {e(meta.get("error") or "")}</div></a>')
    src = '<div class="src">From your rides folder, judged on its own</div>' if meta["params"].get("source") == "folder" else ""
    if meta["kind"] == "plan":
        body = (f'Zone 2 on <em>{100 * s["km_share"]:.0f} %</em> of the km, <em>{100 * s["time_share"]:.0f} %</em> of the time'
                f' · {s["passages"]} passage{"s" if s["passages"] != 1 else ""} where it is not')
        bar = [(s["km_share"], OK), (1 - s["km_share"], NO)]
    else:
        body = (f'the road <em>{s["road"]:.0f}</em> · the heat <em>{s["heat"]:.0f}</em> · me <em>{s["me"]:.0f}</em>'
                f' · {s["km_total"]:.0f} km')
        bar = [(s["road"], NO), (s["heat"], HEAT), (s["me"], ME)]
    mini = "".join(f'<i style="flex:{max(v, 0.001):.3f};background:{c}"></i>' for v, c in bar if v > 0)
    return (f'<a class="tile" href="{href}"><div class="k">{e(k)}</div><div class="t">{e(s.get("title") or meta.get("name", ""))}</div>'
            f'<div class="b">{body}</div><div class="mini">{mini}</div>{src}</a>')


def home_page(store: Store, settings: Settings | None = None, limits_notes: dict | None = None) -> str:
    items = store.all()
    done = [m for m in items if m["status"] == "done"]
    sky = None
    for m in done:
        try:
            sky = json.loads((store.folder(m["id"]) / "skyline.json").read_text(encoding="utf-8"))
            break
        except (OSError, ValueError):
            continue
    if sky and len(sky["km"]) >= 2:
        ridge = skyline(sky["km"], sky["elev"], [NO if x else None for x in sky["lit"]])
    else:
        x = np.linspace(0, 60, 400)
        ridge = skyline(x, 300 + 180 * np.sin(x / 7) + 90 * np.sin(x / 2.3 + 1), [None] * x.size)
    head = (f'<header class="hero home">{ridge}<div class="hero-text"><div class="kicker">Zone 2, before and after a ride</div>'
            f'<h1>Is it me, or the mountain?</h1><div class="sub">Drop a route and a start time, or the FIT file of a ride: '
            f'how much of it can stay in Zone 2, and who made the minutes above it, the road, the heat or you.</div>'
            f'<a class="btn big" href="/new">＋ New analysis</a></div></header>')
    rides = [m for m in done if m["kind"] == "review"]
    total = ""
    if len(rides) >= 2:
        tot = {k: sum(m["summary"][k] for m in rides) for k in ("above", "road", "heat", "me")}
        split = "".join(f'<div style="flex:{max(tot[k], 0.001):.3f};background:{c}"><b>{tot[k]:.0f}</b><span>{lab}</span></div>'
                        for k, c, lab in (("road", NO, "the road"), ("heat", HEAT, "the heat"), ("me", ME, "me")) if tot[k] >= 0.5)
        total = (f'<section class="card total"><h2>Your rides so far</h2><div class="sum"><b>{tot["above"]:.0f}</b> min above '
                 f'Zone 2 on {len(rides)} rides</div><div class="split">{split}</div><p class="note">Each minute goes to one '
                 f'cause: the road (Zone 2 impossible there, even in the cool), the heat, or me.</p></section>')
    if items:
        listing = (f'<section class="group"><h2>Your analyses <span class="count n">{len(items)}</span></h2>'
                   f'<div class="list">{"".join(tile(m) for m in items)}</div></section>')
    else:
        listing = ('<section class="card"><h2>Your analyses</h2><p>None yet. Start with a route you plan to ride, '
                   'or the FIT file of a ride you did.</p></section>')
    profile = ""
    if settings is not None:
        n = limits_notes or {}
        models = n.get("models") or {}
        named = {"tabpfn": "TabPFN v2", "linear": "a straight line", "constant": "a constant"}
        how = ", ".join(f"{lab} {named.get(models[k], models[k])}" for k, lab in
                        (("hr", "heart rate:"), ("cadence", "lowest cadence:"), ("speed", "speed:")) if k in models)
        rides_n = len(n.get("rides", []))
        text = f"Zone 2 up to {settings.z2_top:.0f} bpm."
        if rides_n:
            text = (f"Learned on {rides_n} of your rides" + (f" ({e(how)})" if how else "") + f", Zone 2 up to "
                    f"{settings.z2_top:.0f} bpm. To learn them again from new rides: <code>z2analyzer learn</code>.")
        if settings.watch:
            text += (f" New rides that land in your folder “{e(Path(settings.watch).name)}” are judged on their own, "
                     f"every {settings.every_min:g} min.")
        profile = f'<section class="card"><h2>Your limits</h2><p class="note" style="margin:0">{text}</p></section>'
    # The first card may sit over the hero; a bare heading may not.
    main = "<main>" if total else '<main style="margin-top:6px">'
    return document("z2analyzer", f"{head}{main}{total}{listing}{profile}</main>")


def new_page(token: str, chosen: dict | None = None, defaults: dict | None = None) -> str:
    """The two forms; `chosen` ({'plan': name} or {'review': name}) draws a file as chosen, as the page does once one is."""
    chosen = chosen or {}
    d = {"day": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"), "time": "08:30", "name": "", "conditions": "forecast"}
    d.update(defaults or {})

    def drop(kind: str, accept: str, what: str, hint: str) -> str:
        name = chosen.get(kind, "")
        return (f'<label class="drop{" has" if name else ""}"><input type="file" name="file" accept="{accept}" required>'
                f'<span class="what">{what}</span><span class="hint">{hint}</span><span class="name">{e(name)}</span></label>')

    def radio(value: str, label: str) -> str:
        return (f'<label><input type="radio" name="conditions" value="{value}"{" checked" if d["conditions"] == value else ""}>'
                f'{label}</label>')

    hidden = f'<input type="hidden" name="token" value="{e(token)}">'
    plan_form = f"""<form class="card form" method="post" action="/new" enctype="multipart/form-data">{hidden}
<input type="hidden" name="kind" value="plan"><div class="k">Before a ride</div><h3>A route and a start time</h3>
<p>How much of it can stay in Zone 2, in kilometres and in time, and where it can't.</p>
{drop("plan", ".gpx,.fit", "Drop a GPX here, or choose a file", "or the FIT file of a ride done before")}
<label class="field">Name <input type="text" name="name" maxlength="80" placeholder="from the file" value="{e(d["name"])}"></label>
<div class="row"><label class="field">Day <input type="date" name="day" value="{e(d["day"])}"></label>
<label class="field">Start <input type="time" name="time" value="{e(d["time"])}"></label></div>
<fieldset class="conds"><legend>Temperature</legend>{radio("forecast", "Météo-France forecast for that start")}
{radio("cool", "The cool of my rides")}{radio("hot", "A hot day, 30 °C")}</fieldset>
<button class="btn big" type="submit">Analyse the route</button></form>"""
    review_form = f"""<form class="card form" method="post" action="/new" enctype="multipart/form-data">{hidden}
<input type="hidden" name="kind" value="review"><div class="k">After a ride</div><h3>The FIT file of the ride</h3>
<p>Who made the minutes above Zone 2: the road, the heat, or you.</p>
{drop("review", ".fit", "Drop a FIT file here, or choose one", "from your bike computer, Garmin Connect or intervals.icu")}
<p class="note">Indoor rides and rides without power are left out: they have no real slope, or nothing to judge.</p>
<button class="btn big" type="submit">Analyse the ride</button></form>"""
    head = short_hero("New analysis", "Before or after a ride?",
                      "Your file stays on this machine. Only one rounded weather point goes to Open-Meteo.")
    return document("z2analyzer · new analysis", f'{head}<main><div class="two">{plan_form}{review_form}</div></main>')


def progress_page(meta: dict, refresh: bool = True) -> str:
    steps = STEPS[meta["kind"]]
    k = int(meta.get("step", 0))
    items = "".join(f'<li class="{"done" if i < k else "now" if i == k else ""}">{e(s)}</li>' for i, s in enumerate(steps))
    head = short_hero(f"{_when(meta)} · {meta['file']}", "Analysing…", e(meta.get("name") or meta["file"]))
    body = (f'{head}<main><section class="card"><h2>A few seconds</h2><ol class="steps">{items}</ol>'
            f'<p class="note">This page follows the analysis and opens the result by itself.</p></section></main>')
    return document("z2analyzer · analysing", body, refresh=refresh)


def failed_page(meta: dict, token: str) -> str:
    head = short_hero(f"{_when(meta)} · {meta['file']}", "Not analysed", "")
    body = (f'{head}<main><section class="card"><p class="error">{e(meta.get("error") or "Something went wrong.")}</p>'
            f'<div class="about" style="margin-top:14px"><a class="btn" href="/new">Try another file</a>{delete_form(meta, token)}</div>'
            f'</section></main>')
    return document("z2analyzer · not analysed", body)


def delete_form(meta: dict, token: str) -> str:
    return (f'<form method="post" action="/a/{meta["id"]}/delete"><input type="hidden" name="token" value="{e(token)}">'
            f'<button class="btn line" type="submit">Delete this analysis</button></form>')


def result_page(store: Store, meta: dict, token: str) -> str:
    """The analysis's own page, with the app's bar on top and what it was made from at the bottom."""
    page = (store.folder(meta["id"]) / "page.html").read_text(encoding="utf-8")
    lim = meta.get("limits") or {}
    made = (f"limits learned on {lim['rides']} rides" if lim.get("rides") else f"{lim.get('source', 'fixed')} limits")
    where = " · from your rides folder" if meta["params"].get("source") == "folder" else ""
    about = (f'<section class="card about"><span class="chip">{e(KIND[meta["kind"]])} · {e(meta["file"])}{where}</span>'
             f'<span class="chip">judged with your {e(made)}</span>{delete_form(meta, token)}</section>')
    page = page.replace("</style>", APP_CSS + "</style>", 1)
    page = page.replace("<body>", "<body>" + appbar(), 1)
    return page.replace("</main>", about + "</main>", 1)


# ------------------------------------------------------------------ the server


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """Fields and files of a multipart/form-data body, as browsers send it."""
    m = re.search(r'boundary="?([^";]+)"?', content_type or "")
    if not m or not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("expected a form with a file")
    sep = b"--" + m.group(1).encode("latin-1")
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for chunk in body.split(sep)[1:]:
        if chunk.startswith(b"--"):
            break  # the closing boundary
        head, _, data = chunk.partition(b"\r\n\r\n")
        if data.endswith(b"\r\n"):
            data = data[:-2]
        disp = next((ln for ln in head.decode("utf-8", "replace").split("\r\n")
                     if ln.lower().startswith("content-disposition:")), "")
        name = re.search(r';\s*name="([^"]*)"', disp)
        if not name:
            continue
        filename = re.search(r';\s*filename="([^"]*)"', disp)
        if filename is not None:
            files[name.group(1)] = (filename.group(1), data)
        else:
            fields[name.group(1)] = data.decode("utf-8", "replace")
    return fields, files


class App:
    def __init__(self, store: Store, settings: Settings, rider):
        from .limits import Limits

        self.store, self.settings = store, settings
        self.notes = Limits.load(settings.limits).notes or {}
        self.token = secrets.token_urlsafe(16)
        self.analyst = Analyst(store, settings, rider)
        self.stopped = threading.Event()
        if settings.watch:
            threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self) -> None:
        while not self.stopped.is_set():
            for meta in watch_folder(self.store, self.settings.watch, self.store.root.parent / "watched.json"):
                self.analyst.submit(meta["id"])
            self.stopped.wait(self.settings.every_min * 60)

    def stop(self) -> None:
        self.stopped.set()
        self.analyst.stop()

    def new(self, fields: dict, files: dict) -> dict:
        """Check the form and queue the analysis; a ValueError says what is wrong with it."""
        kind = fields.get("kind")
        if kind not in KIND:
            raise ValueError("unknown kind of analysis")
        filename, data = files.get("file", ("", b""))
        if not data:
            raise ValueError("no file came with the form")
        what = looks_like(data)
        if kind == "review" and what != "fit":
            raise ValueError(f"{filename} is not a FIT file")
        if kind == "plan" and what is None:
            raise ValueError(f"{filename} is neither a GPX nor a FIT file")
        params = {}
        if kind == "plan":
            params["name"] = fields.get("name", "").strip()[:80]
            params["conditions"] = fields.get("conditions", "forecast")
            if params["conditions"] not in ("forecast", "cool", "hot"):
                raise ValueError("unknown temperature choice")
            if params["conditions"] == "forecast":
                when = f"{fields.get('day', '').strip()} {fields.get('time', '').strip()}"
                try:
                    datetime.strptime(when, "%Y-%m-%d %H:%M")
                except ValueError:
                    raise ValueError("give the day and the time of the start") from None
                if datetime.strptime(when, "%Y-%m-%d %H:%M") < datetime.now() - timedelta(hours=12):
                    raise ValueError("that start is in the past: the forecast only looks ahead")
                params["when"] = when
        meta = self.store.create(kind, filename, data, params)
        self.analyst.submit(meta["id"])
        return meta


def make_server(app: App, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "z2analyzer"

        def log_message(self, fmt, *args):  # quiet: the analyses say what they do on their pages
            pass

        def _send(self, code: int, text: str = "", location: str | None = None) -> None:
            data = text.encode("utf-8")
            self.send_response(code)
            if location:
                self.send_header("Location", location)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(data)

        def _oops(self, code: int, title: str, msg: str) -> None:
            body = (f'{short_hero("z2analyzer", title, "")}<main><section class="card"><p class="error">{e(msg)}</p>'
                    f'<a class="btn" href="/new">New analysis</a> <a class="btn line" href="/">Your analyses</a></section></main>')
            self._send(code, document(f"z2analyzer · {title}", body))

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, home_page(app.store, app.settings, app.notes))
            if path == "/new":
                return self._send(200, new_page(app.token))
            m = re.fullmatch(r"/a/([0-9a-f-]+)", path)
            if m and (meta := app.store.get(m.group(1))):
                if meta["status"] == "done":
                    return self._send(200, result_page(app.store, meta, app.token))
                if meta["status"] == "failed":
                    return self._send(200, failed_page(meta, app.token))
                return self._send(200, progress_page(meta))
            if path == "/favicon.ico":
                self.send_response(204)
                return self.end_headers()
            return self._oops(404, "Not found", "There is no such page, or that analysis was deleted.")

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                size = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                size = -1
            if size < 0 or size > MAX_UPLOAD:
                self.close_connection = True
                return self._oops(413, "Too big", f"Files up to {MAX_UPLOAD // 2 ** 20} MB.")
            body = self.rfile.read(size)
            try:
                fields, files = parse_multipart(self.headers.get("Content-Type", ""), body)
            except ValueError:
                fields, files = {}, {}
                if self.headers.get("Content-Type", "").startswith("application/x-www-form-urlencoded"):
                    from urllib.parse import parse_qs

                    fields = {k: v[0] for k, v in parse_qs(body.decode("utf-8", "replace")).items()}
            if not secrets.compare_digest(fields.get("token", ""), app.token):
                return self._oops(403, "Expired form", "This form comes from an older page: open it again and retry.")
            if path == "/new":
                try:
                    meta = app.new(fields, files)
                except ValueError as ex:
                    return self._oops(400, "Not analysed", friendly(ex))
                return self._send(303, "", location=f"/a/{meta['id']}")
            m = re.fullmatch(r"/a/([0-9a-f-]+)/delete", path)
            if m and app.store.get(m.group(1)):
                app.store.delete(m.group(1))
                return self._send(303, "", location="/")
            return self._oops(404, "Not found", "There is no such page, or that analysis was deleted.")

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def serve(args) -> int:
    from .cli import log, rider_from_env
    from .limits import Limits

    rider = rider_from_env(args.mass_kg)
    top = args.z2_top if args.z2_top is not None else Limits.load(args.limits).top_bpm
    if top is None:
        raise SystemExit("set the top of your Zone 2 with --z2-top (bpm)")
    if args.watch and not Path(args.watch).is_dir():
        raise SystemExit(f"no such folder: {args.watch}")
    settings = Settings(limits=args.limits, terrain=args.terrain, structures=args.structures, z2_top=float(top),
                        tz=args.tz, edge_temp=args.edge_temp, watch=args.watch, every_min=args.every)
    app = App(Store(args.data), settings, rider)
    httpd = make_server(app, args.host, args.port)
    log(f"z2analyzer is running on http://{args.host}:{httpd.server_address[1]}/ "
        f"(analyses kept in {Path(args.data).resolve()}; Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        app.stop()
    return 0
