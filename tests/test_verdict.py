import numpy as np
import pytest

from z2analyzer.geo import cumulative_m
from z2analyzer.limits import Limits, Table2D, fixed_limits
from z2analyzer.physics import Rider, power_needed
from z2analyzer.profile import build_profile
from z2analyzer.verdict import climb_minutes, constant_temp, ewma, forecast_temp, plan, review

RIDER = Rider(mass_kg=75.0)  # made-up numbers throughout


def route(flat1=2000.0, climb=2000.0, flat2=1000.0, grade=0.10):
    """North-bound route: flat, a steady climb, flat again (one point every ~5.6 m)."""
    total = flat1 + climb + flat2
    n = int(total / 5.56) + 1
    lat = 41.95 + np.arange(n) * 0.00005
    lon = np.full(n, 8.80)
    d = cumulative_m(lat, lon)
    d = d * (total / d[-1])

    def elevation(la, lo):
        x = np.interp(la, lat, d)
        return 50.0 + grade * np.clip(x - flat1, 0, climb)

    return build_profile(lat, lon, elevation), (flat1, climb, flat2)


def test_ewma_and_climb_minutes():
    x = np.r_[np.zeros(10), np.ones(600)]
    y = ewma(x, 60.0)
    assert y[9] == 0 and 0.6 < y[70] < 0.7 and y[-1] > 0.99
    g = np.r_[np.zeros(60), np.full(120, 0.05), np.full(30, 0.015), np.zeros(10)]
    cm = climb_minutes(g, np.ones(g.size, dtype=bool))
    assert cm[59] == 0 and cm[179] == pytest.approx(2.0) and cm[209] == pytest.approx(2.5) and cm[-1] == 0


def test_table_interpolation_is_exact_on_planes():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0])
    z = 3 * x[:, None] + 0.5 * y[None, :]
    t = Table2D(x, y, z, z, z)
    assert t(1.5, 5.0) == pytest.approx(3 * 1.5 + 2.5)
    assert t(9.0, -3.0) == pytest.approx(6.0)  # clamped at the edges


def test_limits_round_trip(tmp_path):
    lim = fixed_limits(RIDER, 150.0, 120.0)
    lim.save(tmp_path / "l.json")
    back = Limits.load(tmp_path / "l.json")
    assert back.ceiling(22.5, 60.0) == pytest.approx(lim.ceiling(22.5, 60.0))
    assert lim.ceiling(15.0, 0.0) == pytest.approx(150.0) and lim.ceiling(30.0, 0.0) == pytest.approx(120.0)


def test_plan_on_a_steep_climb():
    prof, (f1, climb, f2) = route(grade=0.10)
    lim = fixed_limits(RIDER, ceiling_cool_w=150.0, ceiling_hot_w=120.0, cadence_rpm=50.0)
    p = plan(prof, RIDER, lim, constant_temp(15.0))
    # At 50 rpm in 34x34 a 10 % climb asks about 160 W of this made-up rider: above 150 W.
    v = RIDER.bike.speed_at_cadence(50.0)
    assert power_needed(v, 0.10, RIDER) > 150.0
    assert len(p.passages) == 1
    ps = p.passages[0]
    assert ps.start_m == pytest.approx(f1, abs=150) and ps.length_m == pytest.approx(climb, abs=250)
    assert p.shares.km_share == pytest.approx((f1 + f2) / (f1 + climb + f2), abs=0.06)
    # The climb is slow, so it weighs more in time than in distance.
    assert p.shares.time_share < p.shares.km_share - 0.2
    assert ps.max_ratio > 1.0


def test_plan_heat_makes_a_moderate_climb_impossible():
    prof, _ = route(grade=0.08)
    lim = fixed_limits(RIDER, ceiling_cool_w=150.0, ceiling_hot_w=110.0, cadence_rpm=50.0)
    cool = plan(prof, RIDER, lim, constant_temp(15.0))
    hot = plan(prof, RIDER, lim, constant_temp(30.0))
    assert not cool.passages and hot.passages
    assert hot.shares.km_share < cool.shares.km_share


def test_forecast_temp_lapse_rate():
    fn = forecast_temp(np.array([8.0, 9.0, 10.0]), np.array([18.0, 20.0, 22.0]), ref_elev_m=0.0, start_hour=8.5)
    assert fn(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(19.0)
    assert fn(np.array([1800.0]), np.array([1000.0]))[0] == pytest.approx(20.0 - 6.5)


def ride_over(prof, speeds_ms, hr_fn, cadence_rpm=50.0):
    """One-second samples along the profile at the given speed per stretch."""
    t, d = 0, 0.0
    dist, spd = [], []
    while d < prof.length_m:
        v = speeds_ms(d)
        dist.append(d)
        spd.append(v)
        d += v
        t += 1
    dist = np.array(dist)
    spd = np.array(spd)
    hr = np.array([hr_fn(x) for x in dist])
    cad = np.full(dist.size, cadence_rpm)
    return dist, spd, hr, cad


def test_review_splits_minutes_between_road_heat_and_me():
    prof, (f1, climb, f2) = route(grade=0.10)
    lim = fixed_limits(RIDER, ceiling_cool_w=150.0, ceiling_hot_w=120.0, cadence_rpm=50.0)
    v_climb = RIDER.bike.speed_at_cadence(50.0)

    def speed(x):
        return v_climb if f1 <= x < f1 + climb else 8.0

    # Heart above Zone 2 on the climb once it has caught up (the heart lags by a few
    # minutes), and on the first flat, where the rider pushed.
    dist, spd, hr, cad = ride_over(prof, speed, lambda x: 155.0 if (x < 1000.0 or f1 + 400 <= x < f1 + climb) else 130.0)
    r = review(dist_m=dist, speed_ms=spd, hr=hr, cadence_rpm=cad, moving=spd > 1.0, temp_air_c=np.full(dist.size, 15.0),
               profile=prof, rider=RIDER, limits=lim, z2_top_bpm=145.0, cool_temp_c=15.0)
    climb_min = (climb - 400) / v_climb / 60.0
    flat_min = 1000.0 / 8.0 / 60.0
    assert r.above_min == pytest.approx(climb_min + flat_min, rel=0.03)
    assert r.road_min == pytest.approx(climb_min, rel=0.08)
    assert r.me_min == pytest.approx(flat_min, rel=0.15)
    assert r.heat_min == pytest.approx(0.0, abs=0.5)
    assert r.road_range[0] <= r.road_min <= r.road_range[1]
    # The made-up rider stayed in 34x34 on the climb: the gear check sees it.
    assert r.gear["impossible_smallest_gear_share"] > 0.95
    assert len(r.passages) >= 1


def test_review_blames_the_heat_when_only_the_heat_closes_the_road():
    prof, (f1, climb, f2) = route(grade=0.08)
    lim = fixed_limits(RIDER, ceiling_cool_w=150.0, ceiling_hot_w=110.0, cadence_rpm=50.0)
    v_climb = RIDER.bike.speed_at_cadence(50.0)

    def speed(x):
        return v_climb if f1 <= x < f1 + climb else 8.0

    dist, spd, hr, cad = ride_over(prof, speed, lambda x: 155.0 if f1 + 100 <= x < f1 + climb else 130.0)
    r = review(dist_m=dist, speed_ms=spd, hr=hr, cadence_rpm=cad, moving=spd > 1.0, temp_air_c=np.full(dist.size, 30.0),
               profile=prof, rider=RIDER, limits=lim, z2_top_bpm=145.0, cool_temp_c=15.0)
    assert r.heat_min > 0.8 * r.above_min
    assert r.road_min < 0.1 * r.above_min
