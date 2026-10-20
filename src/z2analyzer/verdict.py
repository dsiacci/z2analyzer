"""Zone 2 possible or impossible, stretch by stretch: before a ride (a route) and after it (a FIT).

Before. For every stretch of about 100 m: the slowest speed the rider can
really ride (lowest cadence held, smallest gear), the power that speed needs
on that grade, and the Zone 2 ceiling for the temperature and the time
already ridden at that point. Zone 2 is impossible where the power needed is
above the ceiling. Riding easy where it is possible and as slowly as possible
where it is not gives the time, so the share is told in kilometres and in
time.

After. Every second above the top of Zone 2 gets one of three causes, judged
on the minimum power the road asked for over the minute before (the heart
lags the legs by a minute or two):
- the road: even in the cool, the slowest possible pace was above the ceiling;
- the heat: possible in the cool, not at that day's temperature;
- me: Zone 2 was possible there that day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .gears import gear_index, ratio_from
from .limits import Limits
from .physics import Rider, air_density, power_needed
from .profile import Profile, Segment, segments

CLIMB_START = 0.02  # a climb starts at 2 % ...
CLIMB_END = 0.01  # ... and ends below 1 %
LAPSE_C_PER_M = -0.0065

TempFn = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (seconds since the start, elevation m) -> °C


@dataclass
class Stretch:
    start_m: float
    end_m: float
    grade: float
    elev_m: float
    temp_c: float
    elapsed_min: float
    cadence_rpm: float
    min_speed_ms: float
    power_min_w: float
    ceiling_w: float
    ceiling_low_w: float
    ceiling_high_w: float
    speed_ms: float
    time_s: float

    @property
    def ratio(self) -> float:
        """Power needed at the slowest pace, as a multiple of the Zone 2 ceiling."""
        return self.power_min_w / self.ceiling_w if self.ceiling_w > 0 else float("inf")

    @property
    def possible(self) -> bool:
        return self.power_min_w <= self.ceiling_w

    @property
    def certain(self) -> bool:
        """The verdict holds at both ends of the ceiling's range."""
        return (self.power_min_w <= self.ceiling_low_w) or (self.power_min_w > self.ceiling_high_w)

    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m


@dataclass
class Passage:
    """Consecutive stretches where Zone 2 is impossible."""

    start_m: float
    end_m: float
    mean_grade: float
    max_grade: float
    max_ratio: float
    time_s: float

    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m


@dataclass
class Shares:
    km_total: float
    km_possible: float
    time_total_s: float
    time_possible_s: float

    @property
    def km_share(self) -> float:
        return self.km_possible / self.km_total if self.km_total else float("nan")

    @property
    def time_share(self) -> float:
        return self.time_possible_s / self.time_total_s if self.time_total_s else float("nan")


@dataclass
class Plan:
    stretches: list[Stretch]
    shares: Shares
    passages: list[Passage]
    shares_low: Shares  # with the low end of the ceiling (more stretches impossible)
    shares_high: Shares


def constant_temp(temp_c: float) -> TempFn:
    return lambda t, z: np.full(np.shape(t), float(temp_c))


def forecast_temp(hours: np.ndarray, temps_c: np.ndarray, ref_elev_m: float, start_hour: float) -> TempFn:
    """Hourly temperatures at one reference point, moved to the road's elevation (−6.5 °C per km)."""
    def fn(t, z):
        h = start_hour + np.asarray(t, dtype=float) / 3600.0
        base = np.interp(h, hours, temps_c)
        return base + LAPSE_C_PER_M * (np.asarray(z, dtype=float) - ref_elev_m)
    return fn


def plan(profile: Profile, rider: Rider, limits: Limits, temp: TempFn, start_elapsed_min: float = 0.0,
         stretch_m: float = 100.0, top_bpm: float | None = None) -> Plan:
    """Walk the route once, keeping time, so heat and hours ridden apply where they happen.

    Before a ride nobody knows the day's level of the heart: the range of the
    ceiling is how far that level moved between the rides it was learned from.
    """
    segs = segments(profile, stretch_m)
    out: list[Stretch] = []
    t = 0.0
    climbing_s = 0.0
    in_climb = False
    day_lo, day_hi = limits.day_range()
    for s in segs:
        if s.grade >= CLIMB_START or (in_climb and s.grade >= CLIMB_END):
            in_climb = True
        else:
            in_climb, climbing_s = False, 0.0
        elapsed = start_elapsed_min + t / 60.0
        temp_c = float(temp(np.array([t]), np.array([s.elev_m]))[0])
        # A heart running high (day_hi) lowers the ceiling: low, middle, high.
        ceilings = tuple(float(np.ravel(limits.ceiling_w(temp_c, elapsed, "mid", off, top_bpm))[0])
                         for off in (day_hi, 0.0, day_lo))
        st = _judge(s, rider, limits, temp_c, elapsed, climbing_s / 60.0, ceilings)
        out.append(st)
        t += st.time_s
        if in_climb:
            climbing_s += st.time_s
    return Plan(stretches=out, shares=_shares(out, "mid"), passages=passages(out, gap_m=150.0),
                shares_low=_shares(out, "low"), shares_high=_shares(out, "high"))


def _judge(s: Segment, rider: Rider, limits: Limits, temp_c: float, elapsed_min: float, climb_min: float,
           ceilings: tuple[float, float, float]) -> Stretch:
    bike = rider.bike
    rho = float(air_density(temp_c, s.elev_m))
    cad = float(limits.cadence(s.grade, climb_min)) if s.grade > 0 else float(limits.cadence(0.0, 0.0))
    v_min = float(bike.speed_at_cadence(cad))
    p_min = float(power_needed(v_min, s.grade, rider, rho))
    lo, mid, hi = ceilings
    if p_min <= mid:
        v = float(limits.speed(s.grade, min(mid, _cruise_power(s.grade, mid))))
        v = max(v, v_min)
    else:
        v = v_min
    return Stretch(start_m=s.start_m, end_m=s.end_m, grade=s.grade, elev_m=s.elev_m, temp_c=temp_c,
                   elapsed_min=elapsed_min, cadence_rpm=cad, min_speed_ms=v_min, power_min_w=p_min,
                   ceiling_w=mid, ceiling_low_w=lo, ceiling_high_w=hi, speed_ms=v, time_s=s.length_m / v)


def _cruise_power(grade: float, ceiling_w: float) -> float:
    """Power when riding easy: the ceiling uphill and on the flat, fading to none on steep descents."""
    if grade >= -0.01:
        return ceiling_w
    return ceiling_w * max(0.0, 1.0 + (grade + 0.01) / 0.04)  # zero from −5 %


def _shares(st: list[Stretch], which: str) -> Shares:
    attr = {"low": "ceiling_low_w", "mid": "ceiling_w", "high": "ceiling_high_w"}[which]
    km_total = sum(s.length_m for s in st) / 1000.0
    ok = [s for s in st if s.power_min_w <= getattr(s, attr)]
    return Shares(km_total=km_total, km_possible=sum(s.length_m for s in ok) / 1000.0,
                  time_total_s=sum(s.time_s for s in st), time_possible_s=sum(s.time_s for s in ok))


def passages(st: list[Stretch], gap_m: float = 0.0) -> list[Passage]:
    """Merge impossible stretches that touch (or are within `gap_m` of each other)."""
    out: list[Passage] = []
    cur: list[Stretch] = []
    for s in st:
        if not s.possible:
            if cur and s.start_m - cur[-1].end_m > gap_m:
                out.append(_passage(cur))
                cur = []
            cur.append(s)
        elif cur and s.end_m - cur[-1].end_m > gap_m:
            out.append(_passage(cur))
            cur = []
    if cur:
        out.append(_passage(cur))
    return out


def _passage(cur: list[Stretch]) -> Passage:
    length = sum(s.length_m for s in cur)
    rise = sum(s.grade * s.length_m for s in cur)
    return Passage(start_m=cur[0].start_m, end_m=cur[-1].end_m, mean_grade=rise / length,
                   max_grade=max(s.grade for s in cur), max_ratio=max(s.ratio for s in cur),
                   time_s=sum(s.time_s for s in cur))


# ---------------------------------------------------------------- after the ride


@dataclass
class Review:
    """Minutes above Zone 2 and their causes, for one ride."""

    moving_min: float
    above_min: float
    road_min: float
    heat_min: float
    me_min: float
    road_range: tuple[float, float]  # (fewest, most) minutes over the ceiling's range
    heat_range: tuple[float, float]
    me_range: tuple[float, float]
    shares: Shares  # the ride's own kilometres and time where Zone 2 was possible that day
    passages: list[Passage]
    gear: dict = field(default_factory=dict)  # the model-free check on the smallest gear
    day_offset_bpm: float = 0.0  # how far above its usual level the heart ran that day
    per_second: dict = field(default_factory=dict)  # arrays for figures (cause, possible, ...)


def ewma(x, tau_s: float):
    """Exponentially weighted moving average on a one-second series (NaN counts as zero)."""
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    a = 1.0 - np.exp(-1.0 / tau_s)
    out = np.empty_like(x)
    acc = x[0] if x.size else 0.0
    for i, v in enumerate(x):
        acc += a * (v - acc)
        out[i] = acc
    return out


def climb_minutes(grade, moving):
    """Minutes already spent in the current climb, for every second."""
    g = np.asarray(grade, dtype=float)
    out = np.zeros(g.size)
    acc = 0.0
    climbing = False
    for i in range(g.size):
        if g[i] >= CLIMB_START or (climbing and g[i] >= CLIMB_END):
            climbing = True
            if moving[i]:
                acc += 1.0
        else:
            climbing, acc = False, 0.0
        out[i] = acc / 60.0
    return out


def review(*, dist_m, speed_ms, hr, cadence_rpm, moving, temp_air_c, profile: Profile, rider: Rider,
           limits: Limits, z2_top_bpm: float, cool_temp_c: float, lag_s: float = 60.0, power_w=None) -> Review:
    """Causes of the seconds above Zone 2 on a ride (one-second arrays, same length).

    With `power_w`, the day's level of the heart is measured on the ride's
    steady stretches first (the ceiling of a tired or a fresh day), then the
    ceilings use it.
    """
    moving = np.asarray(moving, dtype=bool)
    d = np.asarray(dist_m, dtype=float)
    grade = profile.grade_at(d)
    elev = profile.elev_at(d)
    temp = np.asarray(temp_air_c, dtype=float)
    elapsed_min = np.cumsum(moving) / 60.0
    climb_min = climb_minutes(grade, moving)

    cad = limits.cadence(np.maximum(grade, 0.0), np.where(grade > 0, climb_min, 0.0))
    v_min = rider.bike.speed_at_cadence(cad)
    p_min = np.where(moving, power_needed(v_min, grade, rider, air_density(temp, elev)), 0.0)
    p_lag = ewma(p_min, lag_s)

    offset = 0.0
    if power_w is not None:
        pw = np.nan_to_num(np.asarray(power_w, dtype=float))
        offset = limits.day_offset(hr, ewma(pw, 60.0), ewma(pw, 300.0), temp, elapsed_min, moving, z2_top_bpm)
    # Ceilings are evaluated on a coarse time grid (one value every 10 s) and interpolated.
    grid = np.arange(0, temp.size, 10)

    def ceilings(t_air):
        out = {}
        for w in ("low", "mid", "high"):
            c = np.ravel(limits.ceiling_w(t_air[grid], elapsed_min[grid], w, offset, z2_top_bpm))
            out[w] = np.interp(np.arange(temp.size), grid, c)
        return out

    ceil = ceilings(temp)
    cool = ceilings(np.full(temp.shape, cool_temp_c))
    above = moving & (np.nan_to_num(np.asarray(hr, dtype=float)) > z2_top_bpm)

    def split(c_day, c_cool):
        road = above & (p_lag > c_cool)
        heat = above & ~road & (p_lag > c_day)
        me = above & ~road & ~heat
        return road, heat, me

    road, heat, me = split(ceil["mid"], cool["mid"])
    # Fewest minutes for the road and the heat with the high ceilings, most with the low ones.
    r_hi, h_hi, m_hi = split(ceil["high"], cool["high"])
    r_lo, h_lo, m_lo = split(ceil["low"], cool["low"])
    to_min = lambda m: float(np.sum(m)) / 60.0  # noqa: E731

    # Kilometres and time of the ride where Zone 2 was possible that day (the instantaneous road).
    possible_now = p_min <= ceil["mid"]
    dd = np.diff(d, prepend=d[0])
    km_total = float(np.sum(np.where(moving, dd, 0.0))) / 1000.0
    km_ok = float(np.sum(np.where(moving & possible_now, dd, 0.0))) / 1000.0
    shares = Shares(km_total=km_total, km_possible=km_ok, time_total_s=float(np.sum(moving)),
                    time_possible_s=float(np.sum(moving & possible_now)))

    ratios = ratio_from(speed_ms, cadence_rpm, rider.bike.wheel_m)
    gidx = gear_index(ratios, rider.bike)
    gear = gear_check(gidx, above, possible_now, moving)
    return Review(moving_min=float(np.sum(moving)) / 60.0, above_min=to_min(above),
                  road_min=to_min(road), heat_min=to_min(heat), me_min=to_min(me),
                  road_range=_span(to_min(r_hi), to_min(r_lo)), heat_range=_span(to_min(h_hi), to_min(h_lo)),
                  me_range=_span(to_min(m_lo), to_min(m_hi)), shares=shares,
                  passages=_ride_passages(d, possible_now, moving, grade, p_min, ceil["mid"]),
                  gear=gear, day_offset_bpm=offset,
                  per_second={"grade": grade, "p_min_ratio": p_lag / np.maximum(ceil["mid"], 1.0),
                              "above": above, "road": road, "heat": heat, "me": me, "gear": gidx,
                              "possible": possible_now})


def _span(a: float, b: float) -> tuple[float, float]:
    """(fewest, most): a lower ceiling can move minutes from the heat to the road, so either end can be larger."""
    return (min(a, b), max(a, b))


def gear_check(gidx, above, possible, moving) -> dict:
    """The check that needs no model: which gear was in when the heart was above Zone 2?

    Where Zone 2 was declared impossible, the smallest gear should already be
    engaged; where it was possible and the heart was above it anyway, an
    easier gear was often still there.
    """
    g = np.asarray(gidx)
    known = moving & (g >= 0)
    imp = known & ~possible
    imp_above = imp & above
    pos_above = known & possible & above
    share = lambda num, den: float(num.sum() / den.sum()) if den.sum() else float("nan")  # noqa: E731
    return {
        "impossible_s": int(imp.sum()),
        "impossible_smallest_gear_share": share(imp & (g == 0), imp),
        "impossible_above_s": int(imp_above.sum()),
        "impossible_above_smallest_gear_share": share(imp_above & (g == 0), imp_above),
        "possible_above_s": int(pos_above.sum()),
        "possible_above_easier_gear_left_share": share(pos_above & (g > 0), pos_above),
    }


def _ride_passages(d, possible, moving, grade, p_min, ceiling, min_len_m: float = 50.0) -> list[Passage]:
    """Runs of seconds where Zone 2 was impossible (stops inside a run do not break it)."""
    out = []
    bad = moving & ~possible
    dd = np.diff(d, prepend=d[0])
    i = 0
    n = len(d)
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (bad[j + 1] or not moving[j + 1]):
            j += 1
        span = slice(i, j + 1)
        length = float(d[j] - d[i])
        if length >= min_len_m:
            sel = bad[span]
            ratio = (p_min[span] / np.maximum(ceiling[span], 1.0))[sel]
            out.append(Passage(start_m=float(d[i]), end_m=float(d[j]),
                               mean_grade=float(np.average(grade[span][sel], weights=np.maximum(dd[span][sel], 1e-6))),
                               max_grade=float(np.max(grade[span][sel])), max_ratio=float(np.max(ratio)),
                               time_s=float(np.sum(moving[span]))))
        i = j + 1
    return out
