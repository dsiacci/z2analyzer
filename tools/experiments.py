"""Variants of the heart-rate and cadence models, all scored leave-one-ride-out.

    python tools/experiments.py prepared/ --out experiments.json

Every variant is scored on rides it never saw: error of the median, coverage
of the 10-90 % interval, and the pinball loss of the quantiles that the
verdicts use (the median heart rate, the 10 % cadence). The rule, fixed
before looking: the variant with the lowest pinball loss on the quantile
that matters wins.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from z2analyzer.dataset import RideData, chunks  # noqa: E402
from z2analyzer.models import QUANTILES, LinearModel, TabPFNModel  # noqa: E402
from z2analyzer.verdict import ewma  # noqa: E402


def pinball(y, q, tau):
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


class Hybrid:
    """Linear regression for the trend, TabPFN for what the line misses (its residuals)."""

    name = "linear + TabPFN on residuals"

    def __init__(self, X, y):
        self.lin = LinearModel(X, y)
        resid = np.asarray(y, float) - self.lin.quantiles(X, (0.5,))[:, 0]
        self.tab = TabPFNModel(X, resid)

    def quantiles(self, X, qs=QUANTILES):
        return self.lin.quantiles(X, (0.5,))[:, :1] + self.tab.quantiles(X, qs)


def hr_table(rides, size_s=30, extra=False, drop_temp=False):
    rows = []
    for rd in rides:
        if extra:
            rd.p20 = ewma(rd.power, 20.0)
            rd.p900 = ewma(rd.power, 900.0)
        c = chunks(rd, size_s)
        if extra:
            n = len(c["hr"])
            cut = n * size_s
            c["p20"] = rd.p20[:cut].reshape(n, size_s).mean(axis=1)
            c["p900"] = rd.p900[:cut].reshape(n, size_s).mean(axis=1)
        rows.append(c)
    c = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    keep = c["full"] & np.isfinite(c["hr"])
    feats = ["p60", "p300"] + (["p20", "p900"] if extra else []) + ([] if drop_temp else ["temp"]) + ["elapsed_min"]
    return {"X": np.column_stack([c[f][keep] for f in feats]).astype(np.float32), "y": c["hr"][keep].astype(np.float32),
            "ride": c["ride"][keep], "features": feats}


def cadence_table(rides, size_s=30):
    from z2analyzer.dataset import tables

    t = tables([chunks(r, size_s) for r in rides])["cadence"]
    t["features"] = list(t["features"])
    return t


def loo(table, make, tau_main):
    rides = sorted(set(table["ride"].tolist()))
    ys, qs, per = [], [], {}
    t0 = time.time()
    for r in rides:
        test = table["ride"] == r
        if test.sum() == 0:
            continue
        m = make(table["X"][~test], table["y"][~test])
        q = m.quantiles(table["X"][test], QUANTILES)
        y = table["y"][test]
        ys.append(y)
        qs.append(q)
        per[r] = {"n": int(test.sum()), "mae": float(np.mean(np.abs(q[:, QUANTILES.index(0.5)] - y))),
                  "bias": float(np.mean(q[:, QUANTILES.index(0.5)] - y)),
                  "pinball_main": pinball(y, q[:, QUANTILES.index(tau_main)], tau_main)}
    y = np.concatenate(ys)
    q = np.concatenate(qs)
    lo, hi = q[:, QUANTILES.index(0.1)], q[:, QUANTILES.index(0.9)]
    return {"mae": float(np.mean(np.abs(q[:, QUANTILES.index(0.5)] - y))), "bias": float(np.mean(q[:, QUANTILES.index(0.5)] - y)),
            "coverage_80": float(np.mean((y >= lo) & (y <= hi))),
            "pinball_main": pinball(y, q[:, QUANTILES.index(tau_main)], tau_main),
            "pinball_mean": float(np.mean([pinball(y, q[:, i], t) for i, t in enumerate(QUANTILES)])),
            "n": int(len(y)), "seconds": round(time.time() - t0), "per_ride": per}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="names of the variants to run")
    a = ap.parse_args()
    rides = [RideData.load(f) for f in sorted(Path(a.prepared).glob("*.npz"))]
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    tab = lambda X, y: TabPFNModel(X, y)  # noqa: E731
    variants = {
        "hr30 linear": (lambda: hr_table(rides, 30), LinearModel, 0.5),
        "hr60 linear": (lambda: hr_table(rides, 60), LinearModel, 0.5),
        "hr60 tabpfn": (lambda: hr_table(rides, 60), tab, 0.5),
        "hr60+lags linear": (lambda: hr_table(rides, 60, extra=True), LinearModel, 0.5),
        "hr60+lags tabpfn": (lambda: hr_table(rides, 60, extra=True), tab, 0.5),
        "hr60 no-temp tabpfn": (lambda: hr_table(rides, 60, drop_temp=True), tab, 0.5),
        "hr60 hybrid": (lambda: hr_table(rides, 60), Hybrid, 0.5),
        "cadence linear": (lambda: cadence_table(rides), LinearModel, 0.1),
        "cadence tabpfn": (lambda: cadence_table(rides), tab, 0.1),
        "cadence hybrid": (lambda: cadence_table(rides), Hybrid, 0.1),
    }
    for name, (table_fn, make, tau) in variants.items():
        if a.only and name not in a.only:
            continue
        if name in out:
            continue
        t = table_fn()
        res = loo(t, make, tau)
        res["features"] = t["features"]
        out[name] = res
        print(f"{name:22s} MAE {res['mae']:.2f}  pinball@{tau} {res['pinball_main']:.3f}  80% holds {res['coverage_80']:.0%}"
              f"  n={res['n']}  {res['seconds']}s", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
