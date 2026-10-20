"""The proof, in two steps that can run on different machines.

    python tools/proof.py learn prepared/ proof/ --z2-top 145
        For every ride, limits learned from all the other rides: TabPFN, a
        linear regression on the same columns, and a fixed ceiling and cadence.
        Needs TabPFN, not the rider's mass.

    Z2_MASS_KG=... python tools/proof.py judge prepared/ proof/ --z2-top 145
        Every ride judged with the limits that never saw it, and the totals.
        Needs the mass (physics), not TabPFN.

Output: proof/judged.json and a Markdown table on stdout (minutes and shares
only: no watts, no mass).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from z2analyzer import cli  # noqa: E402
from z2analyzer.dataset import RideData  # noqa: E402
from z2analyzer.limits import Limits  # noqa: E402
from z2analyzer.verdict import review  # noqa: E402

KINDS = ("tabpfn", "linear", "fixed")


def cmd_learn(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in Path(a.prepared).glob("*.npz"))
    for kind in a.kinds:
        extra = [] if kind == "tabpfn" else ["--baseline", kind]
        target = out / f"all.{kind}.json"
        if not target.exists():
            cli.main(["learn", a.prepared, "--z2-top", str(a.z2_top), "--out", str(target), *extra])
        for n in names:
            target = out / f"without-{n}.{kind}.json"
            if target.exists():
                continue
            cli.main(["learn", a.prepared, "--z2-top", str(a.z2_top), "--out", str(target), "--exclude", n, *extra])
    return 0


def cmd_judge(a) -> int:
    rider = cli.rider_from_env(None)
    out = Path(a.out)
    rides = sorted(Path(a.prepared).glob("*.npz"))
    res = {"z2_top_bpm": a.z2_top, "rides": {}}
    for f in rides:
        rd = RideData.load(f)
        row = {"date": None}
        for kind in KINDS:
            lf = out / f"without-{rd.name}.{kind}.json"
            if not lf.exists():
                continue
            lim = Limits.load(lf)
            cool = float(lim.notes.get("cool_c", 15.0))
            r = review(dist_m=rd.dist, speed_ms=rd.speed, hr=rd.hr, cadence_rpm=rd.cadence, moving=rd.moving,
                       temp_air_c=rd.temp_air, profile=rd.profile, rider=rider, limits=lim, z2_top_bpm=a.z2_top,
                       cool_temp_c=cool)
            row[kind] = {"moving_min": r.moving_min, "above_min": r.above_min, "road_min": r.road_min,
                         "heat_min": r.heat_min, "me_min": r.me_min, "road_range": r.road_range,
                         "heat_range": r.heat_range, "me_range": r.me_range, "km_share": r.shares.km_share,
                         "time_share": r.shares.time_share, "cool_c": cool, "gear": r.gear,
                         "passages": len(r.passages)}
        from datetime import datetime, timezone

        row["date"] = datetime.fromtimestamp(rd.start_epoch, timezone.utc).astimezone().strftime("%Y-%m-%d")
        row["temp_air_median"] = float(np.nanmedian(rd.temp_air[rd.moving]))
        res["rides"][rd.name] = row
    (out / "judged.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(table(res))
    return 0


def table(res: dict) -> str:
    lines = ["| ride | air | moving | above Z2 | road | heat | me | Z2 possible (km / time) | smallest gear where impossible |",
             "|---|---|---|---|---|---|---|---|---|"]
    tot = {k: 0.0 for k in ("moving", "above", "road", "heat", "me")}
    for name, row in sorted(res["rides"].items(), key=lambda kv: kv[1]["date"]):
        t = row.get("tabpfn")
        if not t:
            continue
        for k in tot:
            tot[k] += t[f"{k}_min"]
        g = t["gear"].get("impossible_smallest_gear_share", float("nan"))
        lines.append(f"| {row['date']} | {row['temp_air_median']:.0f} °C | {t['moving_min']:.0f} min | {t['above_min']:.0f} | "
                     f"{t['road_min']:.0f} | {t['heat_min']:.0f} | {t['me_min']:.0f} | "
                     f"{100 * t['km_share']:.0f} % / {100 * t['time_share']:.0f} % | {100 * g:.0f} % |")
    lines.append(f"| all | | {tot['moving']:.0f} min | {tot['above']:.0f} | {tot['road']:.0f} | {tot['heat']:.0f} | {tot['me']:.0f} | | |")
    lines.append("")
    lines.append("| limits | road | heat | me |")
    lines.append("|---|---|---|---|")
    for kind in KINDS:
        s = {k: sum(r[kind][f"{k}_min"] for r in res["rides"].values() if kind in r) for k in ("road", "heat", "me")}
        lines.append(f"| {kind} | {s['road']:.0f} | {s['heat']:.0f} | {s['me']:.0f} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("learn")
    s.add_argument("prepared")
    s.add_argument("out")
    s.add_argument("--z2-top", type=float, required=True)
    s.add_argument("--kinds", nargs="+", default=list(KINDS))
    s = sub.add_parser("judge")
    s.add_argument("prepared")
    s.add_argument("out")
    s.add_argument("--z2-top", type=float, required=True)
    a = ap.parse_args()
    if a.cmd == "learn":
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
        return cmd_learn(a)
    return cmd_judge(a)


if __name__ == "__main__":
    raise SystemExit(main())
