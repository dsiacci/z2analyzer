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
    s.add_argument("--baseline", action="store_true", help="linear regression and constants instead of TabPFN")
    s.add_argument("--n-estimators", type=int, default=4)

    args = ap.parse_args(argv)
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
    from .models import ConstantModel, LinearModel, TabPFNModel, learned_limits

    rides, tabs = _load_tables(args.prepared, args.exclude)
    if args.baseline:
        models = {"hr": LinearModel(tabs["hr"]["X"], tabs["hr"]["y"]),
                  "cadence": ConstantModel(tabs["cadence"]["X"], tabs["cadence"]["y"]),
                  "speed": LinearModel(tabs["speed"]["X"], tabs["speed"]["y"])}
    else:
        models = {k: TabPFNModel(tabs[k]["X"], tabs[k]["y"], n_estimators=args.n_estimators) for k in ("hr", "cadence", "speed")}
    temp = tabs["hr"]["X"][:, 2]
    notes = {"rides": [r.name for r in rides], "z2_top_bpm": args.z2_top,
             "temp_range_c": [float(np.percentile(temp, 2)), float(np.percentile(temp, 98))],
             "cool_c": float(np.percentile(temp, 10)), "learned": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    lim = learned_limits(models, args.z2_top, notes)
    lim.source = "baseline" if args.baseline else "tabpfn"
    lim.save(args.out)
    log(f"limits written to {args.out} ({lim.source}, {len(rides)} rides)")
    return 0


def rider_from_env(mass_kg: float | None = None):
    from .physics import Rider

    m = mass_kg if mass_kg is not None else os.environ.get("Z2_MASS_KG")
    if m is None:
        raise SystemExit("set the rider's mass with Z2_MASS_KG (kg) or --mass-kg")
    return Rider(mass_kg=float(m))


if __name__ == "__main__":
    raise SystemExit(main())
