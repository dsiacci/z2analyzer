"""TabPFN v2 learns the rider from their own rides; simple baselines keep it honest.

TabPFN is a tabular foundation model (Prior Labs): pre-trained on synthetic
tables, it predicts on a new table in one forward pass, given the training
rows as context. Nothing is fine-tuned. It returns a distribution, not a
single number, which is what the verdicts need: the Zone 2 ceiling comes with
a range, the lowest cadence is a low quantile. Weights: TabPFN v2 regressor,
Prior Labs License 1.1. Built with PriorLabs-TabPFN.

Three models, one table each (see `dataset.py`):
- heart rate from smoothed power, air temperature and minutes ridden;
- cadence on climbs in the smallest gear, from grade and minutes climbing;
- speed from grade and power.
"""

from __future__ import annotations

import os

import numpy as np

from .limits import CLIMB_MIN, ELAPSED, GRADES, POWERS, TEMPS, HeartTable, Limits, Table2D

QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
MAX_CONTEXT = 3000  # rows given to TabPFN as context (a small CPU stays at a few seconds)


def make_regressor(n_estimators: int = 4, random_state: int = 0):
    """A TabPFN v2 regressor on CPU (the v2 weights, not the newer non-commercial ones)."""
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, n_estimators=n_estimators, device="cpu",
                                                      random_state=random_state)


class TabPFNModel:
    name = "TabPFN v2"

    def __init__(self, X, y, n_estimators: int = 4, max_rows: int = MAX_CONTEXT, seed: int = 0):
        X, y = _subsample(np.asarray(X, np.float32), np.asarray(y, np.float32), max_rows, seed)
        self.reg = make_regressor(n_estimators=n_estimators, random_state=seed)
        self.reg.fit(X, y)

    def quantiles(self, X, qs=QUANTILES) -> np.ndarray:
        out = self.reg.predict(np.asarray(X, np.float32), output_type="quantiles", quantiles=list(qs))
        return np.column_stack([np.asarray(o, float) for o in out])


class LinearModel:
    """Least squares on the same features; the spread comes from the training residuals."""

    name = "linear regression"

    def __init__(self, X, y):
        X = np.asarray(X, float)
        A = np.column_stack([X, np.ones(len(X))])
        self.coef, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
        self.resid = np.asarray(y, float) - A @ self.coef

    def quantiles(self, X, qs=QUANTILES) -> np.ndarray:
        X = np.asarray(X, float)
        pred = np.column_stack([X, np.ones(len(X))]) @ self.coef
        return pred[:, None] + np.quantile(self.resid, qs)[None, :]


class ConstantModel:
    """One number for everyone and everything (with the spread of the training values)."""

    name = "constant"

    def __init__(self, X, y):
        self.q = np.asarray(y, float)

    def quantiles(self, X, qs=QUANTILES) -> np.ndarray:
        return np.tile(np.quantile(self.q, qs), (len(X), 1))


class ColumnsModel:
    """A model that only looks at some of the columns (to build blunter baselines)."""

    def __init__(self, base, cols, X, y):
        self.cols = list(cols)
        self.m = base(np.asarray(X)[:, self.cols], y)
        self.name = f"{base.name} on {len(self.cols)} column(s)"

    def quantiles(self, X, qs=QUANTILES) -> np.ndarray:
        return self.m.quantiles(np.asarray(X)[:, self.cols], qs)


def _subsample(X, y, max_rows: int, seed: int):
    if len(y) <= max_rows:
        return X, y
    idx = np.random.default_rng(seed).choice(len(y), size=max_rows, replace=False)
    return X[idx], y[idx]


def score(y, q: np.ndarray, qs=QUANTILES) -> dict:
    """Error of the median, and how often the 10-90 % interval holds the truth."""
    qs = list(qs)
    med = q[:, qs.index(0.5)]
    lo, hi = q[:, qs.index(0.1)], q[:, qs.index(0.9)]
    err = med - np.asarray(y, float)
    return {"n": int(len(err)), "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err)),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "coverage_80": float(np.mean((y >= lo) & (y <= hi))) if len(err) else float("nan")}


def leave_one_ride_out(table: dict, factories: dict) -> dict:
    """Each ride predicted by models trained on all the others, for every model factory."""
    rides = sorted(set(table["ride"].tolist()))
    out = {name: {"per_ride": {}, "all": None} for name in factories}
    preds = {name: [] for name in factories}
    truth = []
    for r in rides:
        test = table["ride"] == r
        if test.sum() == 0 or (~test).sum() < 20:
            continue
        truth.append(table["y"][test])
        for name, make in factories.items():
            m = make(table["X"][~test], table["y"][~test])
            q = m.quantiles(table["X"][test])
            preds[name].append(q)
            out[name]["per_ride"][r] = score(table["y"][test], q)
    y_all = np.concatenate(truth) if truth else np.zeros(0)
    for name in factories:
        if preds[name]:
            out[name]["all"] = score(y_all, np.concatenate(preds[name]))
    return out


# ------------------------------------------------------------ the three tables


def ceiling_table(hr_model, z2_top_bpm: float, temps=TEMPS, elapsed=ELAPSED,
                  powers=np.arange(30.0, 351.0, 5.0)) -> Table2D:
    """Power at which the heart settles at the top of Zone 2, for each temperature and time ridden.

    Steady state: one-minute and five-minute powers equal. Low uses the 90 %
    quantile of the heart rate (the heart runs high: lower ceiling), high the
    10 % quantile.
    """
    T, E, P = np.meshgrid(temps, elapsed, powers, indexing="ij")
    X = np.column_stack([P.ravel(), P.ravel(), T.ravel(), E.ravel()])
    q = hr_model.quantiles(X, (0.1, 0.5, 0.9)).reshape(len(temps), len(elapsed), len(powers), 3)
    q = np.maximum.accumulate(q, axis=2)  # the heart never slows down when the legs push harder
    out = {}
    for k, name in ((2, "low"), (1, "mid"), (0, "high")):
        out[name] = _first_crossing(q[..., k], powers, z2_top_bpm)
    return Table2D(np.asarray(temps, float), np.asarray(elapsed, float), out["low"], out["mid"], out["high"])


def _first_crossing(hr: np.ndarray, powers: np.ndarray, top: float) -> np.ndarray:
    """Interpolated power where hr (last axis over `powers`) first reaches `top`."""
    res = np.empty(hr.shape[:-1])
    for idx in np.ndindex(res.shape):
        h = hr[idx]
        above = np.flatnonzero(h >= top)
        if above.size == 0:
            res[idx] = powers[-1]
        elif above[0] == 0:
            res[idx] = powers[0]
        else:
            j = above[0]
            f = (top - h[j - 1]) / max(h[j] - h[j - 1], 1e-9)
            res[idx] = powers[j - 1] + f * (powers[j] - powers[j - 1])
    return res


def cadence_table(cad_model, grades=GRADES, climb=CLIMB_MIN, q_mid: float = 0.1) -> Table2D:
    """Lowest cadence really held in the smallest gear: `q_mid` quantile (low: 5 %, high: 25 %)."""
    G, C = np.meshgrid(grades, climb, indexing="ij")
    X = np.column_stack([np.maximum(G.ravel(), 0.0), C.ravel()])
    q = cad_model.quantiles(X, (0.05, q_mid, 0.25)).reshape(len(grades), len(climb), 3)
    return Table2D(np.asarray(grades, float), np.asarray(climb, float), q[..., 0], q[..., 1], q[..., 2])


def speed_table(spd_model, grades=GRADES, powers=POWERS) -> Table2D:
    G, P = np.meshgrid(grades, powers, indexing="ij")
    q = spd_model.quantiles(np.column_stack([G.ravel(), P.ravel()]), (0.25, 0.5, 0.75)).reshape(len(grades), len(powers), 3)
    q = np.maximum(q, 0.5)
    return Table2D(np.asarray(grades, float), np.asarray(powers, float), q[..., 0], q[..., 1], q[..., 2])


# Coarser grids for learned tables: TabPFN predicts every grid point, and the
# verdicts interpolate between them.
LEARN_TEMPS = np.arange(8.0, 39.0, 2.0)
LEARN_ELAPSED = np.array([0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0])
LEARN_POWERS = np.arange(40.0, 301.0, 5.0)
LEARN_GRADES = np.round(np.arange(-0.15, 0.2501, 0.01), 4)
LEARN_CLIMB = np.array([0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0])
LEARN_SPEED_POWERS = np.arange(0.0, 401.0, 20.0)


def heart_table(hr_model, temps=LEARN_TEMPS, elapsed=LEARN_ELAPSED, powers=LEARN_POWERS) -> HeartTable:
    """The heart-rate model on a grid (10, 50, 90 % quantiles), never decreasing with power."""
    T, E, P = np.meshgrid(temps, elapsed, powers, indexing="ij")
    X = np.column_stack([P.ravel(), P.ravel(), T.ravel(), E.ravel()])
    q = hr_model.quantiles(X, (0.1, 0.5, 0.9)).reshape(len(temps), len(elapsed), len(powers), 3)
    q = np.maximum.accumulate(q, axis=2)
    q = np.sort(q, axis=3)  # quantiles stay in order
    return HeartTable(np.asarray(temps, float), np.asarray(elapsed, float), np.asarray(powers, float), q)


def domain(tabs: dict) -> dict:
    """Where the rides say something: 1st to 99th percentile of each input the verdicts use."""
    p = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    hx, cx, sx = tabs["hr"]["X"], tabs["cadence"]["X"], tabs["speed"]["X"]
    return {"temp": [p(hx[:, 2], 1), p(hx[:, 2], 99)], "elapsed": [0.0, p(hx[:, 3], 99)],
            "climb_grade": [p(cx[:, 0], 1), p(cx[:, 0], 99)], "climb_min": [0.0, p(cx[:, 1], 99)],
            "speed_grade": [p(sx[:, 0], 1), p(sx[:, 0], 99)], "speed_power": [0.0, p(sx[:, 1], 99)]}


def _grid(lo: float, hi: float, step: float) -> np.ndarray:
    n = max(2, int(np.ceil((hi - lo) / step)) + 1)
    return np.linspace(lo, hi, n)


def learned_limits(models: dict, z2_top_bpm: float, notes: dict | None = None, dom: dict | None = None) -> Limits:
    """The three tables, built only where the rides say something.

    Outside that domain (a colder day, a steeper grade, a longer ride than any
    ride learned from) the tables hold their edge value instead of
    extrapolating: a line keeps falling forever, the rides do not.
    """
    if dom is None:
        heart = heart_table(models["hr"])
        cad = cadence_table(models["cadence"], LEARN_GRADES, LEARN_CLIMB)
        spd = speed_table(models["speed"], LEARN_GRADES, LEARN_SPEED_POWERS)
    else:
        heart = heart_table(models["hr"], _grid(*dom["temp"], 2.0), _grid(*dom["elapsed"], 30.0), LEARN_POWERS)
        cad = cadence_table(models["cadence"], np.round(_grid(max(0.0, dom["climb_grade"][0]), dom["climb_grade"][1], 0.01), 4),
                            _grid(*dom["climb_min"], 5.0))
        spd = speed_table(models["speed"], np.round(_grid(*dom["speed_grade"], 0.01), 4), _grid(*dom["speed_power"], 20.0))
    T, E = np.meshgrid(heart.temps, heart.elapsed, indexing="ij")
    c = {w: heart.ceiling(T.ravel(), E.ravel(), z2_top_bpm, w).reshape(T.shape) for w in ("low", "mid", "high")}
    notes = dict(notes or {}, z2_top_bpm=z2_top_bpm, domain=dom)
    return Limits(ceiling=Table2D(heart.temps, heart.elapsed, c["low"], c["mid"], c["high"]), cadence=cad, speed=spd,
                  source=getattr(models["hr"], "name", "learned"), notes=notes, heart=heart)
