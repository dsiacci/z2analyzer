"""Command line: prepare your rides, learn your limits, then judge a route or a ride.

    z2analyzer structures --bbox 41.3 8.5 43.1 9.6 --out bridges.json
    z2analyzer prepare rides/*.fit --terrain dem/ --structures bridges.json --out prepared/
    z2analyzer validate prepared/ --z2-top 145
    z2analyzer learn prepared/ --z2-top 145 --out limits.json
    z2analyzer plan route.gpx --limits limits.json --terrain dem/ --when "2026-10-24 08:30"
    z2analyzer review ride.fit --limits limits.json --terrain dem/ --z2-top 145

The rider's mass is read from the Z2_MASS_KG environment variable (or
--mass-kg) and never written anywhere. Outputs give shares, minutes, grades
and multiples of the ceiling; watts only with --watts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import __version__


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="z2analyzer", description="How much of a ride can really stay in Zone 2?")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("structures", help="download the road bridges and tunnels of a region (OpenStreetMap)")
    s.add_argument("--bbox", nargs=4, type=float, metavar=("SOUTH", "WEST", "NORTH", "EAST"), required=True)
    s.add_argument("--out", required=True)

    s = sub.add_parser("prepare", help="turn FIT files into one-second series (no positions kept)")
    s.add_argument("fit", nargs="+")
    s.add_argument("--terrain", required=True, help="folder of terrain tiles (IGN RGE ALTI .asc)")
    s.add_argument("--structures", help="bridges and tunnels file from `z2analyzer structures`")
    s.add_argument("--out", required=True)
    s.add_argument("--edge-temp", action="store_true", help="use the computer's sensor instead of Open-Meteo")

    s = sub.add_parser("validate", help="leave each ride out in turn: TabPFN against simple baselines")
    s.add_argument("prepared")
    s.add_argument("--out", help="write the full report as JSON")
    s.add_argument("--n-estimators", type=int, default=4)

    s = sub.add_parser("learn", help="learn the three limits from all prepared rides")
    s.add_argument("prepared")
    s.add_argument("--z2-top", type=float, required=True, help="top of your Zone 2, bpm")
    s.add_argument("--out", required=True)
    s.add_argument("--exclude", nargs="*", default=[], help="ride names to leave out (to judge them fairly)")
    s.add_argument("--baseline", choices=("linear", "fixed"),
                   help="instead of TabPFN: linear regressions on the same columns, or a fixed ceiling and cadence")
    s.add_argument("--n-estimators", type=int, default=4)

    s = sub.add_parser("profile", help="rebuild a route's profile from the terrain (GPX in, .npz out)")
    s.add_argument("route")
    s.add_argument("--terrain", required=True)
    s.add_argument("--structures")
    s.add_argument("--out", required=True)

    s = sub.add_parser("plan", help="before a ride: how much of a route can stay in Zone 2")
    s.add_argument("route", help="GPX file, or a profile from `z2analyzer profile`")
    s.add_argument("--limits", required=True, help="limits.json from `z2analyzer learn` or `z2analyzer fixed`")
    s.add_argument("--terrain", help="terrain tiles (needed for a GPX)")
    s.add_argument("--structures")
    when = s.add_mutually_exclusive_group(required=True)
    when.add_argument("--when", help="local start time, YYYY-MM-DD HH:MM (temperature from the Météo-France forecast)")
    when.add_argument("--temp", type=float, help="one temperature for the whole ride, °C")
    when.add_argument("--cool", action="store_true", help="the coolest conditions of your rides")
    when.add_argument("--hot", action="store_true", help="30 °C")
    s.add_argument("--tz", default="Europe/Paris")
    s.add_argument("--mass-kg", type=float)
    s.add_argument("--watts", action="store_true", help="also print watts")
    s.add_argument("--json", help="write the full verdict as JSON")
    s.add_argument("--html", help="write the local page")

    s = sub.add_parser("review", help="after a ride: who made the minutes above Zone 2 (road, heat, you)")
    s.add_argument("ride", help="FIT file, or a ride from `z2analyzer prepare`")
    s.add_argument("--limits", required=True)
    s.add_argument("--z2-top", type=float, required=True, help="top of your Zone 2, bpm")
    s.add_argument("--terrain", help="terrain tiles (needed for a FIT)")
    s.add_argument("--structures")
    s.add_argument("--cool-temp", type=float, help="reference 'cool' temperature, °C (default: from the limits)")
    s.add_argument("--mass-kg", type=float)
    s.add_argument("--watts", action="store_true")
    s.add_argument("--json")
    s.add_argument("--html")

    s = sub.add_parser("fixed", help="limits from a few numbers, without any ride")
    s.add_argument("--ceiling-cool", type=float, required=True, help="Zone 2 ceiling in the cool, W")
    s.add_argument("--ceiling-hot", type=float, required=True, help="Zone 2 ceiling at 30 °C, W")
    s.add_argument("--cool-temp", type=float, default=15.0)
    s.add_argument("--cadence", type=float, default=50.0, help="lowest cadence you hold on a climb, rpm")
    s.add_argument("--mass-kg", type=float)
    s.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "profile":
        return cmd_profile(args)
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "review":
        return cmd_review(args)
    if args.cmd == "fixed":
        from .limits import fixed_limits

        lim = fixed_limits(rider_from_env(args.mass_kg), args.ceiling_cool, args.ceiling_hot, cool_c=args.cool_temp,
                           cadence_rpm=args.cadence)
        lim.notes["cool_c"] = args.cool_temp
        lim.save(args.out)
        log(f"limits written to {args.out}")
        return 0
    if args.cmd == "structures":
        from .structures import fetch

        n = fetch(tuple(args.bbox), args.out)
        log(f"{n} bridges and tunnels written to {args.out} (© OpenStreetMap contributors, ODbL)")
        return 0
    if args.cmd == "prepare":
        return cmd_prepare(args)
    if args.cmd == "validate":
        return cmd_validate(args)
    if args.cmd == "learn":
        return cmd_learn(args)
    return 1


def cmd_prepare(args) -> int:
    from .dataset import ride_data
    from .dem import Terrain
    from .fitfile import read_fit
    from .structures import Structures
    from .weather import get_hourly

    terrain = Terrain(args.terrain)
    structures = Structures(args.structures) if args.structures else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for f in args.fit:
        ride = read_fit(f)
        if ride.is_virtual:
            log(f"{ride.name}: indoor ride, skipped")
            continue
        if not ride.has_power:
            log(f"{ride.name}: no power, skipped")
            continue
        temp_fn = None
        if not args.edge_temp:
            ok = np.isfinite(ride.lat) & np.isfinite(ride.lon)
            day0 = ride.start.date()
            day1 = (ride.start + timedelta(seconds=float(ride.t[-1]))).date()
            hourly = get_hourly(float(np.mean(ride.lat[ok])), float(np.mean(ride.lon[ok])), day0, day1, archive=True)
            temp_fn = hourly.at
        rd = ride_data(ride, terrain, structures, temp_fn)
        rd.save(out / f"{ride.name}.npz")
        moving = rd.moving
        prof = rd.profile
        rise_dem = float(np.sum(np.clip(np.diff(prof.elev), 0, None)))
        baro = rd.alt_baro[np.isfinite(rd.alt_baro)]
        rise_baro = float(np.sum(np.clip(np.diff(baro), 0, None))) if baro.size else float("nan")
        row = {"ride": ride.name, "date": ride.start.astimezone().strftime("%Y-%m-%d"),
               "start_local": ride.start.astimezone().strftime("%H:%M"), "moving_min": round(float(moving.sum()) / 60, 1),
               "km": round(prof.length_m / 1000, 1), "ascent_dem_m": round(rise_dem), "ascent_baro_m": round(rise_baro),
               "on_structures_m": round(float(prof.on_structure.sum()) * 10),
               "temp_air_median": round(float(np.nanmedian(rd.temp_air[moving])), 1),
               "temp_edge_median": round(float(np.nanmedian(rd.temp_edge[moving])), 1),
               "smallest_gear_share": round(float(np.mean(rd.gear[moving & (rd.gear >= 0)] == 0)), 3)}
        summary.append(row)
        log(json.dumps(row))
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


def _load_tables(prepared: str, exclude=()):
    from .dataset import RideData, chunks, tables

    files = sorted(Path(prepared).glob("*.npz"))
    rides = [RideData.load(f) for f in files]
    rides = [r for r in rides if r.name not in set(exclude)]
    if not rides:
        raise SystemExit(f"no prepared rides in {prepared}")
    return rides, tables([chunks(r) for r in rides])


def cmd_validate(args) -> int:
    from .models import ConstantModel, LinearModel, TabPFNModel, leave_one_ride_out

    rides, tabs = _load_tables(args.prepared)
    report = {"rides": [r.name for r in rides], "rows": {k: int(len(v["y"])) for k, v in tabs.items()}}
    factories = {
        "tabpfn": lambda X, y: TabPFNModel(X, y, n_estimators=args.n_estimators),
        "linear": LinearModel,
        "constant": ConstantModel,
    }
    for name, tab in tabs.items():
        log(f"{name}: {len(tab['y'])} rows from {len(set(tab['ride'].tolist()))} rides, leave-one-ride-out...")
        report[name] = leave_one_ride_out(tab, factories)
        for m, res in report[name].items():
            a = res["all"]
            if a:
                log(f"  {m:9s} MAE {a['mae']:.2f}  bias {a['bias']:+.2f}  80% interval holds {a['coverage_80']:.0%}  (n={a['n']})")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


def cmd_learn(args) -> int:
    from .models import ColumnsModel, ConstantModel, LinearModel, TabPFNModel, learned_limits

    rides, tabs = _load_tables(args.prepared, args.exclude)
    if args.baseline == "linear":
        models = {k: LinearModel(tabs[k]["X"], tabs[k]["y"]) for k in ("hr", "cadence", "speed")}
    elif args.baseline == "fixed":
        # One ceiling for every temperature and hour (heart rate from power alone), one cadence.
        models = {"hr": ColumnsModel(LinearModel, [0], tabs["hr"]["X"], tabs["hr"]["y"]),
                  "cadence": ConstantModel(tabs["cadence"]["X"], tabs["cadence"]["y"]),
                  "speed": LinearModel(tabs["speed"]["X"], tabs["speed"]["y"])}
    else:
        models = {k: TabPFNModel(tabs[k]["X"], tabs[k]["y"], n_estimators=args.n_estimators) for k in ("hr", "cadence", "speed")}
    temp = tabs["hr"]["X"][:, 2]
    notes = {"rides": [r.name for r in rides], "z2_top_bpm": args.z2_top,
             "temp_range_c": [float(np.percentile(temp, 2)), float(np.percentile(temp, 98))],
             "cool_c": float(np.percentile(temp, 10)), "learned": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    lim = learned_limits(models, args.z2_top, notes)
    lim.source = args.baseline or "tabpfn"
    lim.save(args.out)
    log(f"limits written to {args.out} ({lim.source}, {len(rides)} rides)")
    return 0


def _profile_of(route: str, terrain: str | None, structures: str | None):
    """A Profile from a .npz (already built) or a GPX (built now from the terrain tiles)."""
    from .profile import Profile, build_profile

    if route.endswith(".npz"):
        z = np.load(route, allow_pickle=False)
        return Profile(dist=z["dist"], lat=z["lat"], lon=z["lon"], elev=z["elev"], grade=z["grade"],
                       on_structure=z["on_structure"]), str(z["name"])
    if not terrain:
        raise SystemExit("a GPX needs --terrain (folder of terrain tiles)")
    from .dem import Terrain
    from .gpx import read_gpx
    from .structures import Structures

    r = read_gpx(route)
    st = Structures(structures) if structures else None
    return build_profile(r.lat, r.lon, Terrain(terrain), st), r.name


def cmd_profile(args) -> int:
    prof, name = _profile_of(args.route, args.terrain, args.structures)
    np.savez_compressed(args.out, name=name, dist=prof.dist, lat=prof.lat, lon=prof.lon, elev=prof.elev,
                        grade=prof.grade, on_structure=prof.on_structure)
    log(f"profile of {name}: {prof.length_m / 1000:.1f} km, written to {args.out}")
    return 0


def cmd_plan(args) -> int:
    from zoneinfo import ZoneInfo

    from .limits import Limits
    from .report import plan_text, write_plan_html
    from .verdict import constant_temp, forecast_temp, plan

    rider = rider_from_env(args.mass_kg)
    lim = Limits.load(args.limits)
    prof, name = _profile_of(args.route, args.terrain, args.structures)
    if args.when:
        from .weather import get_hourly

        start = datetime.strptime(args.when, "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo(args.tz)).astimezone(timezone.utc)
        hourly = get_hourly(float(np.mean(prof.lat)), float(np.mean(prof.lon)), start.date(),
                            (start + timedelta(hours=12)).date(), archive=False)
        origin = start.replace(minute=0, second=0, microsecond=0)
        temp = forecast_temp(hourly.hours_since(origin), hourly.temp_c, hourly.elevation_m,
                             (start - origin).total_seconds() / 3600.0)
        cond = f"{args.when} ({args.tz}), Météo-France forecast through Open-Meteo"
    else:
        t = args.temp if args.temp is not None else (30.0 if args.hot else float((lim.notes or {}).get("cool_c", 15.0)))
        temp = constant_temp(t)
        cond = f"{t:.0f} °C all along" + (" (the coolest of your rides)" if args.cool else "")
    p = plan(prof, rider, lim, temp)
    print(plan_text(p, name, cond, prof, watts=args.watts))
    if args.json:
        Path(args.json).write_text(json.dumps(_plan_json(p, name, cond, args.watts), indent=1), encoding="utf-8")
    if args.html:
        write_plan_html(args.html, p, name, cond, prof, watts=args.watts)
        log(f"page written to {args.html}")
    return 0


def _plan_json(p, name, cond, watts: bool) -> dict:
    def stretch(s):
        d = {"start_km": round(s.start_m / 1000, 2), "end_km": round(s.end_m / 1000, 2), "grade": round(s.grade, 4),
             "temp_c": round(s.temp_c, 1), "ratio": round(s.ratio, 3), "possible": s.possible,
             "speed_kmh": round(s.speed_ms * 3.6, 1), "time_s": round(s.time_s, 1)}
        if watts:
            d.update(power_min_w=round(s.power_min_w), ceiling_w=round(s.ceiling_w))
        return d

    sh = p.shares
    return {"route": name, "conditions": cond, "km_total": round(sh.km_total, 2),
            "km_share": round(sh.km_share, 4), "time_share": round(sh.time_share, 4),
            "km_share_range": [round(p.shares_low.km_share, 4), round(p.shares_high.km_share, 4)],
            "time_share_range": [round(p.shares_low.time_share, 4), round(p.shares_high.time_share, 4)],
            "passages": [{"start_km": round(x.start_m / 1000, 2), "length_m": round(x.length_m), "mean_grade": round(x.mean_grade, 4),
                          "max_grade": round(x.max_grade, 4), "max_ratio": round(x.max_ratio, 2), "time_s": round(x.time_s)}
                         for x in p.passages],
            "stretches": [stretch(s) for s in p.stretches]}


def cmd_review(args) -> int:
    from .dataset import RideData
    from .limits import Limits
    from .report import review_text, write_review_html
    from .verdict import review

    rider = rider_from_env(args.mass_kg)
    lim = Limits.load(args.limits)
    if args.ride.endswith(".npz"):
        rd = RideData.load(args.ride)
    else:
        if not args.terrain:
            raise SystemExit("a FIT file needs --terrain (folder of terrain tiles)")
        from .dataset import ride_data
        from .dem import Terrain
        from .fitfile import read_fit
        from .structures import Structures
        from .weather import get_hourly

        ride = read_fit(args.ride)
        if ride.is_virtual:
            raise SystemExit("an indoor ride has no real slope")
        ok = np.isfinite(ride.lat) & np.isfinite(ride.lon)
        hourly = get_hourly(float(np.mean(ride.lat[ok])), float(np.mean(ride.lon[ok])), ride.start.date(),
                            (ride.start + timedelta(seconds=float(ride.t[-1]))).date(), archive=True)
        rd = ride_data(ride, Terrain(args.terrain), Structures(args.structures) if args.structures else None, hourly.at)
    cool = args.cool_temp if args.cool_temp is not None else float((lim.notes or {}).get("cool_c", 15.0))
    r = review(dist_m=rd.dist, speed_ms=rd.speed, hr=rd.hr, cadence_rpm=rd.cadence, moving=rd.moving,
               temp_air_c=rd.temp_air, profile=rd.profile, rider=rider, limits=lim, z2_top_bpm=args.z2_top,
               cool_temp_c=cool)
    print(review_text(r, rd, args.z2_top, cool))
    if args.json:
        Path(args.json).write_text(json.dumps(_review_json(r, rd, args.z2_top, cool), indent=1), encoding="utf-8")
    if args.html:
        write_review_html(args.html, r, rd, args.z2_top, cool)
        log(f"page written to {args.html}")
    return 0


def _review_json(r, rd, z2_top, cool) -> dict:
    return {"ride": rd.name, "start_utc": datetime.fromtimestamp(rd.start_epoch, timezone.utc).isoformat(timespec="minutes"),
            "z2_top_bpm": z2_top, "cool_c": cool, "moving_min": round(r.moving_min, 1), "above_min": round(r.above_min, 1),
            "road_min": round(r.road_min, 1), "heat_min": round(r.heat_min, 1), "me_min": round(r.me_min, 1),
            "road_range": [round(x, 1) for x in r.road_range], "heat_range": [round(x, 1) for x in r.heat_range],
            "me_range": [round(x, 1) for x in r.me_range],
            "km_share": round(r.shares.km_share, 4), "time_share": round(r.shares.time_share, 4),
            "temp_air_median": round(float(np.nanmedian(rd.temp_air[rd.moving])), 1),
            "passages": [{"start_km": round(x.start_m / 1000, 2), "length_m": round(x.length_m), "mean_grade": round(x.mean_grade, 4),
                          "max_ratio": round(x.max_ratio, 2), "time_s": round(x.time_s)} for x in r.passages],
            "gear": r.gear}


def rider_from_env(mass_kg: float | None = None):
    from .physics import Rider

    m = mass_kg if mass_kg is not None else os.environ.get("Z2_MASS_KG")
    if m is None:
        raise SystemExit("set the rider's mass with Z2_MASS_KG (kg) or --mass-kg")
    return Rider(mass_kg=float(m))


if __name__ == "__main__":
    raise SystemExit(main())
