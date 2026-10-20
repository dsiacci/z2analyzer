"""Steady-state cycling physics: the power a slope asks for at a given speed.

    P = (m·g·v·(sin θ + Crr·cos θ) + ½·ρ·CdA·v³) / η

θ is the road angle (tan θ = grade), m the rider plus the bike, Crr the
rolling resistance, CdA the drag area, ρ the air density and η the drivetrain
efficiency. No wind and no acceleration. On a climb at walking pace, gravity is
almost all of it: below 8 km/h, drag is under 2 % of the total.

Every function takes plain floats or numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.80665  # m/s²


@dataclass(frozen=True)
class Bike:
    """The bike side of the equation.

    Defaults: a road bike with 50/34 chainrings and an 11-34 eleven-speed
    cassette (smallest gear 34×34), 700×28c tyres.
    """

    chainrings: tuple[int, ...] = (50, 34)
    cogs: tuple[int, ...] = (11, 13, 15, 17, 19, 21, 23, 25, 27, 30, 34)
    wheel_m: float = 2.105  # distance rolled per wheel turn
    mass_kg: float = 11.5  # bike, bottles, tools, computer
    crr: float = 0.005
    cda_m2: float = 0.40
    efficiency: float = 0.975

    @property
    def ratios(self) -> list[float]:
        """Every chainring/cog ratio, smallest (easiest) first."""
        return sorted(c / s for c in self.chainrings for s in self.cogs)

    @property
    def lowest_ratio(self) -> float:
        return min(self.chainrings) / max(self.cogs)

    def speed_at_cadence(self, cadence_rpm, ratio: float | None = None):
        """Road speed (m/s) at this cadence, in the smallest gear unless `ratio` is given."""
        r = self.lowest_ratio if ratio is None else ratio
        return np.asarray(cadence_rpm, dtype=float) / 60.0 * r * self.wheel_m


@dataclass(frozen=True)
class Rider:
    """The rider's mass is a parameter: z2analyzer never stores or prints it."""

    mass_kg: float
    bike: Bike = field(default_factory=Bike)

    @property
    def system_mass_kg(self) -> float:
        return self.mass_kg + self.bike.mass_kg


def air_density(temp_c=15.0, altitude_m=0.0):
    """Air density (kg/m³) from the temperature and the altitude (standard atmosphere pressure)."""
    t = np.asarray(temp_c, dtype=float)
    z = np.asarray(altitude_m, dtype=float)
    pressure = 101325.0 * (1.0 - 2.25577e-5 * z) ** 5.25588
    return pressure / (287.05 * (t + 273.15))


def power_needed(speed_ms, grade, rider: Rider, rho=1.225):
    """Pedal power (W) to hold `speed_ms` on `grade` (0.06 = 6 %). Never below zero."""
    v = np.asarray(speed_ms, dtype=float)
    theta = np.arctan(np.asarray(grade, dtype=float))
    b = rider.bike
    wheel = rider.system_mass_kg * G * v * (np.sin(theta) + b.crr * np.cos(theta)) + 0.5 * rho * b.cda_m2 * v**3
    out = np.maximum(0.0, wheel / b.efficiency)
    return float(out) if out.ndim == 0 else out


def speed_at_power(power_w, grade, rider: Rider, rho=1.225, v_max_ms=25.0, tol=1e-6):
    """Speed (m/s) at which `power_w` balances the road, found by bisection.

    Downhill, the bike can roll faster than `v_max_ms` with no power at all:
    the result is capped there (descending speed is the rider's choice).
    """
    p = np.atleast_1d(np.asarray(power_w, dtype=float))
    g = np.atleast_1d(np.asarray(grade, dtype=float))
    p, g = np.broadcast_arrays(p, g)
    lo = np.zeros(p.shape)
    hi = np.full(p.shape, v_max_ms)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        too_fast = _raw_power(mid, g, rider, rho) > p
        hi = np.where(too_fast, mid, hi)
        lo = np.where(too_fast, lo, mid)
        if np.all(hi - lo < tol):
            break
    out = 0.5 * (lo + hi)
    return float(out[0]) if np.ndim(power_w) == 0 and np.ndim(grade) == 0 else out


def max_grade(power_w, speed_ms, rider: Rider, rho=1.225, tol=1e-7):
    """The steepest grade on which `power_w` still holds `speed_ms`."""
    p = np.atleast_1d(np.asarray(power_w, dtype=float))
    v = np.atleast_1d(np.asarray(speed_ms, dtype=float))
    p, v = np.broadcast_arrays(p, v)
    lo = np.full(p.shape, -0.30)
    hi = np.full(p.shape, 0.40)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        too_steep = _raw_power(v, mid, rider, rho) > p
        hi = np.where(too_steep, mid, hi)
        lo = np.where(too_steep, lo, mid)
        if np.all(hi - lo < tol):
            break
    out = 0.5 * (lo + hi)
    return float(out[0]) if np.ndim(power_w) == 0 and np.ndim(speed_ms) == 0 else out


def _raw_power(v, grade, rider: Rider, rho):
    """Signed pedal power (W); negative means the slope pushes the bike."""
    theta = np.arctan(grade)
    b = rider.bike
    wheel = rider.system_mass_kg * G * v * (np.sin(theta) + b.crr * np.cos(theta)) + 0.5 * rho * b.cda_m2 * v**3
    return wheel / b.efficiency
