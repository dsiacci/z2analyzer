import numpy as np
import pytest

from z2analyzer import geo


def test_haversine_one_degree_of_latitude():
    assert geo.haversine_m(45.0, 5.0, 46.0, 5.0) == pytest.approx(111_195, rel=1e-3)


def test_cumulative_and_resample():
    lat = np.array([45.0, 45.001, 45.002])
    lon = np.array([5.0, 5.0, 5.0])
    d = geo.cumulative_m(lat, lon)
    assert d[0] == 0 and d[-1] == pytest.approx(222.4, rel=1e-3)
    grid, v = geo.resample(d, {"lat": lat}, 10.0)
    assert np.allclose(np.diff(grid)[:-1], 10.0)
    assert grid[-1] == pytest.approx(d[-1])
    assert v["lat"][0] == 45.0 and v["lat"][-1] == pytest.approx(45.002)


def test_resample_ignores_repeated_points():
    d = np.array([0.0, 0.0, 10.0, 10.0, 20.0])
    grid, v = geo.resample(d, {"x": np.array([1.0, 1.0, 2.0, 2.0, 3.0])}, 5.0)
    assert np.allclose(v["x"], [1.0, 1.5, 2.0, 2.5, 3.0])


def test_trim_mask_keeps_the_middle():
    d = np.arange(0, 10_001, 100.0)
    m = geo.trim_mask(d, 2000.0)
    assert not m[d < 2000].any() and not m[d > 8000].any() and m[(d >= 2000) & (d <= 8000)].all()


def test_lambert93_constants_match_ign():
    # Published by IGN for Lambert-93: n, c and Ys.
    assert geo._N == pytest.approx(0.7256077650532670, abs=1e-12)
    assert geo._C == pytest.approx(11754255.426096, abs=1e-3)
    assert geo._YS == pytest.approx(12655612.049876, abs=1e-3)


def test_lambert93_origin_and_round_trip():
    x, y = geo.lambert93(46.5, 3.0)
    assert x == pytest.approx(700000.0, abs=1e-6) and y == pytest.approx(6600000.0, abs=1e-6)
    lat = np.array([41.39, 42.30, 42.70, 43.00])
    lon = np.array([9.16, 9.15, 9.45, 9.35])
    x, y = geo.lambert93(lat, lon)
    lat2, lon2 = geo.lambert93_inverse(x, y)
    assert np.allclose(lat, lat2, atol=1e-9) and np.allclose(lon, lon2, atol=1e-9)
    # Corsica sits around x 1.15-1.25 million, y 6.03-6.25 million.
    assert ((x > 1.1e6) & (x < 1.3e6)).all() and ((y > 6.0e6) & (y < 6.3e6)).all()
