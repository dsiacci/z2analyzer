"""The road profile: elevation from a terrain model, bridges and tunnels bridged over, grade per stretch.

GPS and barometric altitudes drift and jump, and so do the elevations of a
planned GPX. The profile is rebuilt instead: the track is resampled every 10 m,
each point takes the elevation of a terrain model read from local files, the
stretches on bridges and in tunnels are replaced by a straight line between
their ends (the terrain model has the valley or the mountain there, not the
road), then the profile is smoothed and cut into stretches of about 100 m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import cumulative_m, resample

STEP_M = 10.0
SEGMENT_M = 100.0


@dataclass
class Profile:
    """A track every `STEP_M` metres, with its smoothed elevation and local grade."""

    dist: np.ndarray  # metres from the start
    lat: np.ndarray
    lon: np.ndarray
    elev: np.ndarray  # smoothed, metres
    grade: np.ndarray  # rise over run around each point (0.06 = 6 %)
    on_structure: np.ndarray  # bridge or tunnel

    @property
    def length_m(self) -> float:
        return float(self.dist[-1]) if self.dist.size else 0.0

    def grade_at(self, dist_m):
        """Grade at any distance along the track (linear between grid points)."""
        return np.interp(dist_m, self.dist, self.grade)

    def elev_at(self, dist_m):
        return np.interp(dist_m, self.dist, self.elev)


@dataclass
class Segment:
    start_m: float
    end_m: float
    grade: float
    elev_m: float  # mean elevation of the stretch

    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m


def build_profile(lat, lon, elevation_fn, structure_fn=None, step_m: float = STEP_M,
                  smooth_m: float = 50.0) -> Profile:
    """Resample, look up the terrain, bridge the structures, smooth.

    `elevation_fn(lat, lon)` returns metres (NaN where it has no data);
    `structure_fn(lat, lon)` returns True on bridges and in tunnels.
    """
    d0 = cumulative_m(lat, lon)
    dist, v = resample(d0, {"lat": lat, "lon": lon}, step_m)
    raw = np.asarray(elevation_fn(v["lat"], v["lon"]), dtype=float)
    if not np.isfinite(raw).any():
        raise ValueError("the terrain model has no data under this track")
    raw = fill_nan(raw)
    mask = np.zeros(dist.size, dtype=bool) if structure_fn is None else np.asarray(structure_fn(v["lat"], v["lon"]), dtype=bool)
    flat = bridge_over(dist, raw, mask)
    elev = smooth(flat, max(1, int(round(smooth_m / step_m))))
    grade = local_grade(dist, elev, SEGMENT_M)
    return Profile(dist=dist, lat=v["lat"], lon=v["lon"], elev=elev, grade=grade, on_structure=mask)


def fill_nan(x):
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.any() and (~bad).any():
        idx = np.arange(x.size)
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def bridge_over(dist, elev, mask):
    """Straight line across every run of masked points, from the last good point to the next."""
    e = np.asarray(elev, dtype=float).copy()
    m = np.asarray(mask, dtype=bool)
    if not m.any() or m.all():
        return e
    good = ~m
    e[m] = np.interp(np.asarray(dist)[m], np.asarray(dist)[good], e[good])
    return e


def smooth(x, window: int):
    """Running median (kills spikes) then a centred running mean, both over `window` points.

    The ends are padded by odd reflection, which keeps a steady slope steady
    up to the first and last point.
    """
    x = np.asarray(x, dtype=float)
    if window <= 1 or x.size < 3:
        return x.copy()
    half = min(window // 2, x.size - 1)
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    padded = np.pad(x, half, mode="reflect", reflect_type="odd")
    med = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=1)
    padded = np.pad(med, half, mode="reflect", reflect_type="odd")
    return np.convolve(padded, kernel, mode="valid")


def local_grade(dist, elev, span_m: float = SEGMENT_M):
    """Grade over `span_m` centred on each point (shorter at the ends)."""
    d = np.asarray(dist, dtype=float)
    e = np.asarray(elev, dtype=float)
    lo = np.interp(d - span_m / 2, d, e, left=e[0])
    hi = np.interp(d + span_m / 2, d, e, right=e[-1])
    run = np.minimum(d + span_m / 2, d[-1]) - np.maximum(d - span_m / 2, d[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (hi - lo) / run
    return np.nan_to_num(g)


def segments(profile: Profile, length_m: float = SEGMENT_M) -> list[Segment]:
    """Consecutive stretches of `length_m` (the last one takes the remainder if it is long enough)."""
    total = profile.length_m
    if total <= 0:
        return []
    edges = list(np.arange(0.0, total, length_m))
    if total - edges[-1] < length_m / 2 and len(edges) > 1:
        edges.pop()
    edges.append(total)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        ea, eb = profile.elev_at(a), profile.elev_at(b)
        mid = np.linspace(a, b, 5)
        out.append(Segment(start_m=float(a), end_m=float(b), grade=float((eb - ea) / (b - a)),
                           elev_m=float(np.mean(profile.elev_at(mid)))))
    return out


def track_distance(lat, lon):
    """Distance along the track for each sample (seconds of a ride), for lookups in a profile."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    out = np.full(lat.size, np.nan)
    if ok.sum() >= 2:
        out[ok] = cumulative_m(lat[ok], lon[ok])
        idx = np.arange(lat.size)
        out = np.interp(idx, idx[ok], out[ok])
    return out
