from datetime import datetime, timezone

import numpy as np
import pytest

from fitwriter import write_fit
from z2analyzer.fitfile import fill_short_gaps, moving_mask, read_fit

START = datetime(2026, 9, 27, 5, 18, tzinfo=timezone.utc)


def samples(n=120, skip=()):
    out = []
    for i in range(n):
        if i in skip:
            continue
        out.append({"t": i, "lat": 41.95 + i * 1e-5, "lon": 8.80, "hr": 120 + i // 10, "cadence": 60,
                    "dist": i * 5.0, "power": 150, "temp": 21, "speed": 5.0, "alt": 100.0 + i * 0.2})
    return out


def test_read_fit_round_trip(tmp_path):
    p = tmp_path / "ride.fit"
    write_fit(p, START, samples())
    r = read_fit(p)
    assert r.start == START
    assert r.t.size == 120 and r.has_power and not r.is_virtual
    assert r.lat[0] == pytest.approx(41.95, abs=1e-6) and r.lon[5] == pytest.approx(8.80, abs=1e-6)
    assert r.hr[59] == 125 and r.cadence[0] == 60 and r.power[10] == 150
    assert r.speed[3] == pytest.approx(5.0) and r.dist[10] == pytest.approx(50.0)
    assert r.alt[10] == pytest.approx(102.0, abs=0.21) and r.temp[0] == 21
    assert moving_mask(r).all()
    assert r.time_at(60) == datetime(2026, 9, 27, 5, 19, tzinfo=timezone.utc)


def test_gaps_become_nan_and_short_ones_can_be_filled(tmp_path):
    p = tmp_path / "gap.fit"
    write_fit(p, START, samples(skip={40, 41, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79}))
    r = read_fit(p)
    assert r.t.size == 120
    assert np.isnan(r.hr[40]) and np.isnan(r.hr[75])
    hr = fill_short_gaps(r.hr, max_gap=5)
    assert np.isfinite(hr[40]) and np.isfinite(hr[41]) and np.isnan(hr[75])


def test_zwift_files_are_virtual(tmp_path):
    p = tmp_path / "zwift.fit"
    write_fit(p, START, samples(30), manufacturer=260, sub_sport=58)
    assert read_fit(p).is_virtual
