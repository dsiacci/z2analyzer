"""Distances along a track, resampling, and the Lambert-93 projection of the IGN tiles."""

from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (arrays welcome)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def cumulative_m(lat, lon):
    """Distance from the first point, in metres, for every point."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.size == 0:
        return np.zeros(0)
    steps = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return np.concatenate([[0.0], np.cumsum(steps)])


def resample(dist_m, values: dict[str, np.ndarray], step_m: float = 10.0):
    """Values interpolated every `step_m` metres along the track.

    Returns (new distances, dict of interpolated arrays). Points that do not
    move (repeated positions) are dropped first, so the interpolation sees a
    strictly increasing distance.
    """
    d = np.asarray(dist_m, dtype=float)
    keep = np.concatenate([[True], np.diff(d) > 1e-6])
    d = d[keep]
    grid = np.arange(0.0, d[-1] + 1e-9, step_m) if d.size else np.zeros(0)
    if d.size and grid[-1] < d[-1] - 1e-6:
        grid = np.append(grid, d[-1])
    out = {k: np.interp(grid, d, np.asarray(v, dtype=float)[keep]) for k, v in values.items()}
    return grid, out


def trim_mask(dist_m, trim_m: float = 2000.0):
    """True for the points at least `trim_m` from both ends (for maps and figures)."""
    d = np.asarray(dist_m, dtype=float)
    if d.size == 0:
        return np.zeros(0, dtype=bool)
    return (d >= d[0] + trim_m) & (d <= d[-1] - trim_m)


# Lambert-93 (RGF93, GRS80 ellipsoid): the projection of the IGN terrain tiles.
# Conic conformal with two standard parallels (IGN, note NT/G 71, algorithm ALG0003).
_A = 6378137.0
_F = 1 / 298.257222101
_E = math.sqrt(2 * _F - _F * _F)
_LAT0, _LON0 = math.radians(46.5), math.radians(3.0)
_LAT1, _LAT2 = math.radians(44.0), math.radians(49.0)
_X0, _Y0 = 700000.0, 6600000.0


def _m(phi):
    return math.cos(phi) / math.sqrt(1 - (_E * math.sin(phi)) ** 2)


def _t(phi):
    s = np.sin(phi)
    return np.tan(math.pi / 4 - phi / 2) / ((1 - _E * s) / (1 + _E * s)) ** (_E / 2)


_N = (math.log(_m(_LAT1)) - math.log(_m(_LAT2))) / (math.log(float(_t(_LAT1))) - math.log(float(_t(_LAT2))))
_C = _A * _m(_LAT1) / (_N * float(_t(_LAT1)) ** _N)  # IGN constant "c"
_YS = _Y0 + _C * float(_t(_LAT0)) ** _N  # IGN constant "Ys"


def lambert93(lat, lon):
    """(x, y) in metres, Lambert-93, from WGS84/RGF93 degrees (the two datums agree to well under a metre)."""
    phi = np.radians(np.asarray(lat, dtype=float))
    lam = np.radians(np.asarray(lon, dtype=float))
    r = _C * _t(phi) ** _N
    gamma = _N * (lam - _LON0)
    return _X0 + r * np.sin(gamma), _YS - r * np.cos(gamma)


def lambert93_inverse(x, y):
    """(lat, lon) in degrees from Lambert-93 metres."""
    dx = np.asarray(x, dtype=float) - _X0
    dy = _YS - np.asarray(y, dtype=float)
    r = np.hypot(dx, dy)
    gamma = np.arctan2(dx, dy)
    lam = _LON0 + gamma / _N
    lat_iso = -np.log(np.abs(r / _C)) / _N
    phi = 2 * np.arctan(np.exp(lat_iso)) - math.pi / 2
    for _ in range(12):
        s = np.sin(phi)
        phi = 2 * np.arctan(((1 + _E * s) / (1 - _E * s)) ** (_E / 2) * np.exp(lat_iso)) - math.pi / 2
    return np.degrees(phi), np.degrees(lam)
