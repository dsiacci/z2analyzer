import numpy as np

from z2analyzer.gears import dominant_gear, gear_index, in_lowest_gear, ratio_from
from z2analyzer.physics import Bike

BIKE = Bike()


def test_ratio_from_speed_and_cadence():
    v = BIKE.speed_at_cadence(55.0, ratio=34 / 30)
    assert abs(float(ratio_from(v, 55.0, BIKE.wheel_m)) - 34 / 30) < 1e-12


def test_ratio_is_nan_when_coasting_or_stopped():
    r = ratio_from([5.0, 0.2, 5.0], [0.0, 60.0, 20.0], BIKE.wheel_m)
    assert np.isnan(r).all()


def test_two_easiest_gears_are_told_apart_with_noise():
    rng = np.random.default_rng(1)
    lowest = 1.0 * (1 + rng.normal(0, 0.02, 500))
    second = (34 / 30) * (1 + rng.normal(0, 0.02, 500))
    assert in_lowest_gear(lowest, BIKE).mean() > 0.97
    assert (gear_index(second, BIKE) == 1).mean() > 0.95


def test_far_from_any_gear_is_unknown():
    assert gear_index(0.7, BIKE) == -1
    assert gear_index(6.0, BIKE) == -1
    assert gear_index(np.nan, BIKE) == -1


def test_every_exact_ratio_maps_to_itself():
    table = sorted(set(round(x, 6) for x in BIKE.ratios))
    idx = gear_index(np.array(table), BIKE)
    assert list(idx) == list(range(len(table)))


def test_dominant_gear():
    r = np.array([1.0, 1.01, 0.99, 34 / 30, np.nan])
    assert dominant_gear(r, BIKE) == 0
    assert dominant_gear(np.array([np.nan]), BIKE) == -1
