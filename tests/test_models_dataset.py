import json

import numpy as np
import pytest

from z2analyzer.dataset import RideData, chunks, tables
from z2analyzer.models import ConstantModel, LinearModel, ceiling_table, leave_one_ride_out, score
from z2analyzer.profile import Profile
from z2analyzer.structures import Structures


class FakeHeart:
    """Heart rate = 90 + 0.4 W + 0.5 bpm per °C above 15 + 0.03 bpm per minute ridden, ±5 bpm."""

    name = "fake"

    def quantiles(self, X, qs):
        X = np.asarray(X, float)
        mid = 90 + 0.4 * X[:, 0] + 0.5 * (X[:, 2] - 15) + 0.03 * X[:, 3]
        z = {0.1: -5.0, 0.5: 0.0, 0.9: 5.0}
        return np.column_stack([mid + z[q] for q in qs])


def test_ceiling_inverts_the_heart_model():
    t = ceiling_table(FakeHeart(), 145.0, temps=np.array([15.0, 25.0]), elapsed=np.array([0.0, 120.0]),
                      powers=np.arange(30.0, 351.0, 1.0))
    # 90 + 0.4 P = 145 → P = 137.5 W at 15 °C, fresh.
    assert t(15.0, 0.0) == pytest.approx(137.5, abs=0.3)
    # 10 °C warmer: 5 bpm more, so 12.5 W less.
    assert t(25.0, 0.0) == pytest.approx(125.0, abs=0.3)
    # Two hours in: 3.6 bpm more, 9 W less.
    assert t(15.0, 120.0) == pytest.approx(128.5, abs=0.3)
    # The heart running 5 bpm high lowers the ceiling by 12.5 W.
    assert t(15.0, 0.0, "low") == pytest.approx(125.0, abs=0.3)
    assert t(15.0, 0.0, "high") == pytest.approx(150.0, abs=0.3)


def test_baselines_and_score():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (400, 2))
    y = 2 * X[:, 0] + rng.normal(0, 0.1, 400)
    lin = LinearModel(X, y)
    s = score(y, lin.quantiles(X))
    assert s["mae"] < 0.1 and 0.7 < s["coverage_80"] < 0.9
    const = ConstantModel(X, y)
    assert score(y, const.quantiles(X))["mae"] > s["mae"]


def test_leave_one_ride_out_never_sees_the_test_ride():
    X = np.arange(60, dtype=float)[:, None]
    y = np.r_[np.zeros(30), np.ones(30)]
    table = {"X": X, "y": y, "ride": np.array(["a"] * 30 + ["b"] * 30)}
    res = leave_one_ride_out(table, {"constant": ConstantModel})
    # Trained on the other ride only, the constant is exactly wrong by one.
    assert res["constant"]["per_ride"]["a"]["mae"] == pytest.approx(1.0)
    assert res["constant"]["all"]["n"] == 60


def fake_ride(name="r1", n=1800, grade=0.08):
    t = np.arange(n, dtype=float)
    speed = np.full(n, 2.0)
    speed[:60] = 0.0  # a stop at the start
    dist = np.cumsum(speed)
    prof = Profile(dist=np.arange(0, dist[-1] + 10, 10.0), lat=None, lon=None, elev=None, grade=None, on_structure=None)
    prof.elev = 100 + grade * prof.dist
    prof.grade = np.full(prof.dist.size, grade)
    prof.on_structure = np.zeros(prof.dist.size, dtype=bool)
    gear = np.zeros(n, dtype=int)
    return RideData(name=name, start_epoch=0.0, moving=speed >= 1.0, dist=dist, grade=np.full(n, grade),
                    elev=100 + grade * dist, speed=speed, power=np.full(n, 150.0), hr=np.full(n, 140.0),
                    cadence=np.full(n, 57.0), temp_air=np.full(n, 20.0), temp_edge=np.full(n, 24.0),
                    alt_baro=100 + grade * dist, elapsed_min=np.cumsum(speed >= 1.0) / 60.0, climb_min=t / 60.0,
                    p60=np.full(n, 150.0), p300=np.full(n, 150.0), gear=gear, profile=prof)


def test_chunks_and_tables():
    rd = fake_ride()
    c = chunks(rd)
    assert len(c["hr"]) == 60
    assert not c["full"][:2].any() and c["full"][2:].all()  # the stop spoils the first two windows
    assert np.allclose(c["grade"][3:], 0.08, atol=1e-6)
    tabs = tables([c, chunks(fake_ride("r2"))])
    assert tabs["hr"]["X"].shape == (116, 4) and list(tabs["hr"]["features"]) == ["p60", "p300", "temp", "elapsed_min"]
    assert tabs["cadence"]["X"].shape[1] == 2 and np.allclose(tabs["cadence"]["y"], 57.0)
    assert set(tabs["speed"]["ride"].tolist()) == {"r1", "r2"}


def test_ride_data_save_load_keeps_no_positions(tmp_path):
    rd = fake_ride()
    rd.save(tmp_path / "r.npz")
    z = np.load(tmp_path / "r.npz")
    assert not any(k in z.files for k in ("lat", "lon", "profile_lat", "profile_lon"))
    back = RideData.load(tmp_path / "r.npz")
    assert back.name == "r1" and np.allclose(back.profile.grade, 0.08) and back.moving.sum() == rd.moving.sum()


def test_structures_follow_the_bridge_not_the_road_below(tmp_path):
    # A bridge heading north; one track runs over it, another passes under it heading east.
    bridge = [[41.95, 8.80], [41.9509, 8.80]]
    (tmp_path / "s.json").write_text(json.dumps({"bbox": [41.9, 8.7, 42.0, 8.9], "ways": [{"kind": "bridge", "highway": "primary", "coords": bridge}]}))
    s = Structures(tmp_path / "s.json")
    over_lat = np.linspace(41.9495, 41.9515, 21)
    over = s(over_lat, np.full(21, 8.80))
    assert over[(over_lat > 41.9501) & (over_lat < 41.9508)].all() and not over[0] and not over[-1]
    under_lon = np.linspace(8.799, 8.801, 21)
    assert not s(np.full(21, 41.9504), under_lon).any()


def test_heart_table_inverts_to_the_same_ceiling_and_follows_the_day_level():
    from z2analyzer.limits import Limits
    from z2analyzer.models import heart_table, learned_limits

    heart = heart_table(FakeHeart(), temps=np.array([15.0, 25.0]), elapsed=np.array([0.0, 120.0]),
                        powers=np.arange(30.0, 351.0, 1.0))
    assert heart.ceiling(15.0, 0.0, 145.0)[0] == pytest.approx(137.5, abs=0.3)
    assert heart.hr(25.0, 120.0, 137.5)[0] == pytest.approx(90 + 0.4 * 137.5 + 5 + 3.6, abs=0.05)
    # A heart running 4 bpm high that day: the ceiling drops by 10 W.
    assert heart.ceiling(15.0, 0.0, 145.0 - 4.0)[0] == pytest.approx(127.5, abs=0.3)

    lim = learned_limits({"hr": FakeHeart(), "cadence": ConstantModel(np.zeros((5, 2)), np.full(5, 50.0)),
                          "speed": LinearModel(np.random.default_rng(0).uniform(0, 1, (50, 2)), np.ones(50))}, 145.0)
    assert lim.ceiling_w(15.0, 0.0, "mid", 0.0)[0] == pytest.approx(137.5, abs=0.6)
    assert lim.ceiling_w(15.0, 0.0, "mid", 4.0)[0] == pytest.approx(127.5, abs=0.6)
    # The day level is read back from a ride whose heart runs 6 bpm above the model.
    n = 3600
    p = np.full(n, 150.0)
    hr = 90 + 0.4 * p + 0.5 * (20.0 - 15) + 0.03 * (np.arange(n) / 60.0) + 6.0
    off = lim.day_offset(hr, p, p, np.full(n, 20.0), np.arange(n) / 60.0, np.ones(n, bool))
    assert off == pytest.approx(6.0, abs=0.7)
