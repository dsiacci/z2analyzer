"""End to end on made-up data: a terrain tile, a GPX, a FIT, fixed limits, plan and review."""

import json
from datetime import datetime, timezone

import numpy as np

from fitwriter import write_fit
from test_profile_dem import write_tile
from z2analyzer import cli
from z2analyzer.geo import cumulative_m, lambert93

LAT0, LON0 = 42.30, 9.15


def terrain(tmp_path):
    """A valley floor, then an 11 % ramp heading north, then a plateau."""
    x, y = lambert93(LAT0, LON0)
    x_ll, y_ll = float(np.floor(x) - 2000), float(np.floor(y) - 1000)

    def z(X, Y):
        u = Y - y
        return 100 + 0.11 * np.clip(u - 2000, 0, 1500)

    d = tmp_path / "dem"
    d.mkdir()
    write_tile(d / "t.asc", x_ll, y_ll, 400, 1400, 10.0, z)
    return d


def gpx(tmp_path, n=600):
    lat = LAT0 + np.arange(n) * 0.0001
    pts = "".join(f'<trkpt lat="{a:.6f}" lon="{LON0:.6f}"><ele>0</ele></trkpt>' for a in lat)
    p = tmp_path / "route.gpx"
    p.write_text(f'<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><name>Test climb</name><trkseg>{pts}</trkseg></trk></gpx>')
    return p, lat


def test_plan_and_review_end_to_end(tmp_path, capsys):
    dem = terrain(tmp_path)
    route, lat = gpx(tmp_path)
    lim = tmp_path / "limits.json"
    assert cli.main(["fixed", "--ceiling-cool", "150", "--ceiling-hot", "120", "--cadence", "50", "--mass-kg", "75", "--out", str(lim)]) == 0
    page = tmp_path / "plan.html"
    js = tmp_path / "plan.json"
    assert cli.main(["plan", str(route), "--limits", str(lim), "--terrain", str(dem), "--temp", "15", "--mass-kg", "75",
                     "--html", str(page), "--json", str(js)]) == 0
    out = capsys.readouterr().out
    assert "Zone 2 is possible on" in out and "W" not in out.split("Where")[0]
    v = json.loads(js.read_text())
    assert len(v["passages"]) == 1 and 0.09 < v["passages"][0]["mean_grade"] < 0.12
    assert v["time_share"] < v["km_share"] < 1
    assert "power_min_w" not in v["stretches"][0]  # no watts unless asked
    text = page.read_text()
    assert "<svg" in text and "http://" not in text.replace("http://www.w3.org", "") and "Built with PriorLabs-TabPFN" in text

    # A ride up the same road at 52 rpm in 34x34 on the ramp, heart above 145 bpm there.
    d = cumulative_m(lat, np.full(lat.size, LON0))
    t, x, samples = 0, 0.0, []
    while x < d[-1] - 5:
        on_ramp = 2000 <= x <= 3500
        v = 52 / 60 * 2.105 if on_ramp else 7.0
        la = float(np.interp(x, d, lat))
        samples.append({"t": t, "lat": la, "lon": LON0, "hr": 155 if (on_ramp and x > 2300) else 130, "cadence": 52 if on_ramp else 80,
                        "dist": x, "power": 180 if on_ramp else 120, "temp": 15, "speed": v, "alt": 100.0})
        x += v
        t += 1
    fit = tmp_path / "ride.fit"
    write_fit(fit, datetime(2026, 9, 27, 6, 0, tzinfo=timezone.utc), samples)
    rd = tmp_path / "prepared"
    # Prepare with the computer's sensor (no network in the tests), then review from the prepared ride.
    assert cli.main(["prepare", str(fit), "--terrain", str(dem), "--out", str(rd), "--edge-temp"]) == 0
    rj = tmp_path / "review.json"
    assert cli.main(["review", str(rd / "ride.npz"), "--limits", str(lim), "--z2-top", "145", "--cool-temp", "15",
                     "--mass-kg", "75", "--json", str(rj), "--html", str(tmp_path / "review.html")]) == 0
    r = json.loads(rj.read_text())
    assert r["above_min"] > 0 and r["road_min"] > 0.8 * r["above_min"]
    assert r["gear"]["impossible_smallest_gear_share"] > 0.9
    assert "imposed by the road" in capsys.readouterr().out
