"""What the rider can do, as three lookup tables.

1. The Zone 2 ceiling in watts, for a temperature and a time already ridden:
   the power at which the heart rate settles at the top of Zone 2. Heat and
   hours in the saddle both lower it.
2. The lowest cadence really held on a climb, for a grade and a time already
   spent climbing. In the smallest gear, it sets the slowest possible speed.
3. The speed when riding easy, for a grade and a power, to turn kilometres
   into time.

The tables come either from fixed numbers (`fixed_limits`, no data needed) or
from the rider's own rides through TabPFN (see `models.py`). Each table keeps a
low, middle and high value so the uncertainty travels to the verdicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .physics import Rider, air_density, speed_at_power


@dataclass
class Table2D:
    """Values on a regular (x, y) grid, read by bilinear interpolation, clamped at the edges."""

    x: np.ndarray
    y: np.ndarray
    low: np.ndarray  # shape (len(x), len(y))
    mid: np.ndarray
    high: np.ndarray

    def __call__(self, xq, yq, which: str = "mid"):
        z = getattr(self, which)
        xq = np.clip(np.asarray(xq, dtype=float), self.x[0], self.x[-1])
        yq = np.clip(np.asarray(yq, dtype=float), self.y[0], self.y[-1])
        i = np.clip(np.searchsorted(self.x, xq, side="right") - 1, 0, len(self.x) - 2)
        j = np.clip(np.searchsorted(self.y, yq, side="right") - 1, 0, len(self.y) - 2)
        fx = (xq - self.x[i]) / (self.x[i + 1] - self.x[i])
        fy = (yq - self.y[j]) / (self.y[j + 1] - self.y[j])
        return (z[i, j] * (1 - fx) * (1 - fy) + z[i + 1, j] * fx * (1 - fy)
                + z[i, j + 1] * (1 - fx) * fy + z[i + 1, j + 1] * fx * fy)

    def to_dict(self) -> dict:
        return {k: np.asarray(getattr(self, k)).tolist() for k in ("x", "y", "low", "mid", "high")}

    @classmethod
    def from_dict(cls, d: dict) -> "Table2D":
        return cls(**{k: np.asarray(d[k], dtype=float) for k in ("x", "y", "low", "mid", "high")})


@dataclass
class Limits:
    ceiling: Table2D  # x: air temperature °C, y: minutes already ridden → watts
    cadence: Table2D  # x: grade, y: minutes already climbing → lowest cadence held (rpm)
    speed: Table2D  # x: grade, y: pedal power (W) → speed when riding easy (m/s)
    source: str = "fixed"
    notes: dict | None = None

    def save(self, path: str | Path) -> None:
        data = {"source": self.source, "notes": self.notes or {},
                "ceiling": self.ceiling.to_dict(), "cadence": self.cadence.to_dict(), "speed": self.speed.to_dict()}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Limits":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(ceiling=Table2D.from_dict(d["ceiling"]), cadence=Table2D.from_dict(d["cadence"]),
                   speed=Table2D.from_dict(d["speed"]), source=d.get("source", "?"), notes=d.get("notes"))


TEMPS = np.arange(5.0, 41.0, 1.0)
ELAPSED = np.arange(0.0, 421.0, 30.0)
GRADES = np.round(np.arange(-0.15, 0.2501, 0.005), 4)
CLIMB_MIN = np.arange(0.0, 91.0, 10.0)
POWERS = np.arange(0.0, 401.0, 10.0)


def fixed_limits(rider: Rider, ceiling_cool_w: float, ceiling_hot_w: float, cool_c: float = 15.0,
                 hot_c: float = 30.0, cadence_rpm: float = 50.0, drift_w_per_hour: float = 0.0,
                 spread: float = 0.08, descent_kmh: float = 40.0) -> Limits:
    """Limits from a few numbers: no rides needed.

    The ceiling falls linearly from `ceiling_cool_w` at `cool_c` to
    `ceiling_hot_w` at `hot_c` (and keeps that slope beyond), minus
    `drift_w_per_hour` for every hour ridden; low and high are ±`spread`.
    The lowest cadence is constant. Speeds come from the physics, capped at
    `descent_kmh` downhill.
    """
    slope = (ceiling_hot_w - ceiling_cool_w) / (hot_c - cool_c)
    t, e = np.meshgrid(TEMPS, ELAPSED, indexing="ij")
    mid = ceiling_cool_w + slope * (t - cool_c) - drift_w_per_hour * e / 60.0
    mid = np.maximum(mid, 20.0)
    ceiling = Table2D(TEMPS, ELAPSED, mid * (1 - spread), mid, mid * (1 + spread))
    cad = np.full((GRADES.size, CLIMB_MIN.size), float(cadence_rpm))
    cadence = Table2D(GRADES, CLIMB_MIN, cad, cad, cad)
    speed = physics_speed_table(rider, descent_kmh)
    return Limits(ceiling=ceiling, cadence=cadence, speed=speed, source="fixed",
                  notes={"ceiling_cool_w": ceiling_cool_w, "ceiling_hot_w": ceiling_hot_w, "cool_c": cool_c,
                         "hot_c": hot_c, "cadence_rpm": cadence_rpm})


def physics_speed_table(rider: Rider, descent_kmh: float = 40.0) -> Table2D:
    g, p = np.meshgrid(GRADES, POWERS, indexing="ij")
    v = speed_at_power(p, g, rider, rho=float(air_density(20.0, 200.0)), v_max_ms=descent_kmh / 3.6)
    v = np.asarray(v).reshape(g.shape)
    return Table2D(GRADES, POWERS, v, v, v)
