import numpy as np
import pytest

from z2analyzer.dem import Terrain, read_header
from z2analyzer.geo import cumulative_m, lambert93
from z2analyzer.profile import bridge_over, build_profile, local_grade, segments, smooth


def straight_north(n=401, lat0=41.95, lon0=8.80, step_deg=0.0001):
    """A track heading north, about 11 m between points."""
    return lat0 + np.arange(n) * step_deg, np.full(n, lon0)


def test_constant_grade_is_recovered():
    lat, lon = straight_north()
    d0 = cumulative_m(lat, lon)

    def elevation(la, lo):
        return 100.0 + 0.06 * np.interp(la, lat, d0)

    p = build_profile(lat, lon, elevation)
    inner = (p.dist > 100) & (p.dist < p.dist[-1] - 100)
    assert np.allclose(p.grade[inner], 0.06, atol=1e-3)
    segs = segments(p)
    assert all(s.grade == pytest.approx(0.06, abs=2e-3) for s in segs)
    assert sum(s.length_m for s in segs) == pytest.approx(p.length_m)


def test_spike_is_removed_by_the_median():
    x = np.zeros(50)
    x[25] = 40.0
    assert np.max(np.abs(smooth(x, 5))) < 1e-9


def test_bridge_is_drawn_straight():
    d = np.arange(0, 200, 10.0)
    e = np.full(d.size, 50.0)
    e[8:12] = 10.0  # the valley under a bridge
    mask = np.zeros(d.size, dtype=bool)
    mask[8:12] = True
    assert np.allclose(bridge_over(d, e, mask), 50.0)


def test_local_grade_at_the_ends():
    d = np.arange(0, 1000, 10.0)
    e = 0.05 * d
    assert np.allclose(local_grade(d, e, 100.0), 0.05)


def write_tile(path, x_ll, y_ll, ncols, nrows, cell, fn, corner=True):
    """An ESRI ASCII grid whose cell centres follow fn(x, y)."""
    xs = x_ll + cell / 2 + np.arange(ncols) * cell
    ys = y_ll + cell / 2 + np.arange(nrows)[::-1] * cell  # row 0 is the top
    X, Y = np.meshgrid(xs, ys)
    Z = fn(X, Y)
    head = (f"ncols {ncols}\nnrows {nrows}\n"
            + (f"xllcorner {x_ll}\nyllcorner {y_ll}\n" if corner else f"xllcenter {x_ll + cell / 2}\nyllcenter {y_ll + cell / 2}\n")
            + f"cellsize {cell}\nNODATA_value -99999.00\n")
    body = "\n".join(" ".join(f"{z:.3f}" for z in row) for row in Z)
    path.write_text(head + body + "\n")


def test_terrain_is_bilinear_and_exact_on_a_plane(tmp_path):
    lat, lon = 41.95, 8.80
    x, y = lambert93(lat, lon)
    x_ll, y_ll = float(np.floor(x / 1000) * 1000 - 500), float(np.floor(y / 1000) * 1000 - 500)

    def plane(X, Y):
        return 200.0 + 0.03 * (X - x_ll) - 0.01 * (Y - y_ll)

    write_tile(tmp_path / "a.asc", x_ll, y_ll, 400, 400, 5.0, plane)
    write_tile(tmp_path / "b.asc", x_ll + 2000, y_ll, 400, 400, 5.0, plane, corner=False)
    t = Terrain(tmp_path)
    assert len(t.tiles) == 2
    h = read_header(tmp_path / "a.asc")
    assert h.x0 == pytest.approx(x_ll + 2.5) and h.ncols == 400
    z = t.elevation(np.array([lat]), np.array([lon]))
    assert z[0] == pytest.approx(float(plane(x, y)), abs=1e-3)
    zz = t.elevation_xy(np.array([x_ll + 2600.0, x_ll + 9000.0]), np.array([y_ll + 300.0, y_ll + 300.0]))
    assert zz[0] == pytest.approx(float(plane(x_ll + 2600.0, y_ll + 300.0)), abs=1e-3)
    assert np.isnan(zz[1])
    # Between the last cell centre of one tile and the first of the next: no hole.
    seam = t.elevation_xy(np.array([x_ll + 1999.0, x_ll + 2001.0]), np.array([y_ll + 300.0, y_ll + 300.0]))
    assert np.allclose(seam, plane(np.array([x_ll + 1999.0, x_ll + 2001.0]), y_ll + 300.0), atol=0.1)
    assert (tmp_path / "a.npy").exists()  # parsed once, cached
    assert Terrain(tmp_path).elevation(np.array([lat]), np.array([lon]))[0] == pytest.approx(z[0])
