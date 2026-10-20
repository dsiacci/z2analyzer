import math

import numpy as np
import pytest

from z2analyzer.physics import Bike, Rider, air_density, max_grade, power_needed, speed_at_power

RIDER = Rider(mass_kg=75.0)  # a made-up rider: 86.5 kg with the default bike


def test_air_density_standard():
    assert air_density(15.0, 0.0) == pytest.approx(1.225, abs=0.002)
    assert air_density(30.0, 0.0) < air_density(15.0, 0.0)
    assert air_density(15.0, 1000.0) < air_density(15.0, 0.0)


def test_power_on_a_climb_by_hand():
    v, grade = 1.754, 0.10
    theta = math.atan(grade)
    expected = (86.5 * 9.80665 * v * (math.sin(theta) + 0.005 * math.cos(theta)) + 0.5 * 1.225 * 0.40 * v**3) / 0.975
    assert power_needed(v, grade, RIDER) == pytest.approx(expected, rel=1e-9)
    assert 155 < expected < 165


def test_power_never_negative_and_increases_with_grade_and_speed():
    grades = np.linspace(-0.1, 0.15, 26)
    p = power_needed(3.0, grades, RIDER)
    assert np.all(p >= 0)
    assert np.all(np.diff(p[p > 0]) > 0)
    assert power_needed(4.0, 0.05, RIDER) > power_needed(3.0, 0.05, RIDER)


def test_speed_at_power_inverts_power_needed():
    for grade in (-0.02, 0.0, 0.04, 0.09):
        v = speed_at_power(150.0, grade, RIDER)
        assert power_needed(v, grade, RIDER) == pytest.approx(150.0, rel=1e-4)


def test_speed_at_power_caps_descents():
    assert speed_at_power(0.0, -0.08, RIDER, v_max_ms=11.0) == pytest.approx(11.0, abs=1e-3)


def test_max_grade_inverts_power_needed():
    v = Bike().speed_at_cadence(54.0)
    g = max_grade(130.0, v, RIDER)
    assert power_needed(v, g, RIDER) == pytest.approx(130.0, rel=1e-4)
    assert max_grade(160.0, v, RIDER) > g


def test_smallest_gear_speed():
    bike = Bike()
    assert bike.lowest_ratio == pytest.approx(1.0)
    assert bike.speed_at_cadence(60.0) == pytest.approx(2.105)
    assert bike.speed_at_cadence(60.0, ratio=50 / 11) == pytest.approx(2.105 * 50 / 11)
    assert bike.ratios[0] == pytest.approx(1.0) and bike.ratios[-1] == pytest.approx(50 / 11)
