"""Which gear was engaged, recovered from speed and cadence alone.

A FIT file has no gear channel unless the bike has electronic shifting, but
speed and cadence give it away: speed / (cadence / 60 × wheel) is the
chainring/cog ratio. On a 50/34 with an 11-34 cassette the two easiest gears
are 34×34 (ratio 1.0) and 34×30 (1.13), far enough apart to tell whether the
smallest gear was already in.
"""

from __future__ import annotations

import numpy as np

from .physics import Bike


def ratio_from(speed_ms, cadence_rpm, wheel_m: float, min_cadence: float = 30.0, min_speed_ms: float = 1.0):
    """Gear ratio at each sample; NaN while coasting, stopped or soft-pedalling."""
    v = np.asarray(speed_ms, dtype=float)
    c = np.asarray(cadence_rpm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = v / (c / 60.0 * wheel_m)
    return np.where((c >= min_cadence) & (v >= min_speed_ms), r, np.nan)


def gear_index(ratio, bike: Bike):
    """Index of the nearest distinct ratio (0 = easiest), or -1 when too far from all of them.

    "Too far" is more than half the gap to the neighbouring ratio, which keeps
    the two easiest gears apart and leaves ambiguous crossovers unassigned.
    """
    table = np.array(sorted(set(round(x, 6) for x in bike.ratios)))
    r = np.atleast_1d(np.asarray(ratio, dtype=float))
    out = np.full(r.shape, -1, dtype=int)
    ok = np.isfinite(r)
    if not ok.any():
        return out if np.ndim(ratio) else int(out[0])
    idx = np.abs(r[ok, None] - table[None, :]).argmin(axis=1)
    lower = np.where(idx > 0, table[np.maximum(idx - 1, 0)], -np.inf)
    upper = np.where(idx < len(table) - 1, table[np.minimum(idx + 1, len(table) - 1)], np.inf)
    centre = table[idx]
    half_down = np.where(np.isfinite(lower), (centre - lower) / 2, centre * 0.06)
    half_up = np.where(np.isfinite(upper), (upper - centre) / 2, centre * 0.06)
    rr = r[ok]
    inside = ((rr <= centre) & (centre - rr <= half_down)) | ((rr > centre) & (rr - centre <= half_up))
    out[np.flatnonzero(ok)[inside]] = idx[inside]
    return out if np.ndim(ratio) else int(out[0])


def in_lowest_gear(ratio, bike: Bike):
    """True where the easiest gear was engaged (NaN ratios give False)."""
    return gear_index(ratio, bike) == 0


def easier_gears_left(ratio, bike: Bike):
    """How many easier ratios were still available (-1 where the gear is unknown)."""
    return gear_index(ratio, bike)


def dominant_gear(ratio, bike: Bike) -> int:
    """The gear index seen most often in a stretch of samples (-1 if none is clear)."""
    idx = np.asarray(gear_index(ratio, bike))
    idx = idx[idx >= 0]
    if idx.size == 0:
        return -1
    values, counts = np.unique(idx, return_counts=True)
    return int(values[counts.argmax()])
