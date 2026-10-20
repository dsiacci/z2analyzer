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
class HeartTable:
    """Heart rate (10, 50 and 90 % quantiles) at a steady power, for an air temperature and minutes ridden.

    `q` has shape (temps, elapsed, powers, 3) and never decreases along the
    powers. Inverted along the powers, it gives the Zone 2 ceiling for any
    top of Zone 2 and any level of the day.
    """

    temps: np.ndarray
    elapsed: np.ndarray
    powers: np.ndarray
    q: np.ndarray

    def curves(self, temp, elapsed):
        """Heart-rate curves over `powers` for each (temp, elapsed): shape (n, powers, 3)."""
        t = np.clip(np.atleast_1d(np.asarray(temp, float)), self.temps[0], self.temps[-1])
        e = np.clip(np.atleast_1d(np.asarray(elapsed, float)), self.elapsed[0], self.elapsed[-1])
        t, e = np.broadcast_arrays(t, e)
        i = np.clip(np.searchsorted(self.temps, t, side="right") - 1, 0, len(self.temps) - 2)
        j = np.clip(np.searchsorted(self.elapsed, e, side="right") - 1, 0, len(self.elapsed) - 2)
        fi = ((t - self.temps[i]) / (self.temps[i + 1] - self.temps[i]))[:, None, None]
        fj = ((e - self.elapsed[j]) / (self.elapsed[j + 1] - self.elapsed[j]))[:, None, None]
        q = self.q
        return (q[i, j] * (1 - fi) * (1 - fj) + q[i + 1, j] * fi * (1 - fj)
                + q[i, j + 1] * (1 - fi) * fj + q[i + 1, j + 1] * fi * fj)

    def hr(self, temp, elapsed, power, which: str = "mid"):
        k = {"q10": 0, "low": 0, "mid": 1, "q90": 2, "high": 2}[which]
        c = self.curves(temp, elapsed)[..., k]
        p = np.clip(np.atleast_1d(np.asarray(power, float)), self.powers[0], self.powers[-1])
        p = np.broadcast_to(p, (c.shape[0],))
        idx = np.clip(np.searchsorted(self.powers, p, side="right") - 1, 0, len(self.powers) - 2)
        f = (p - self.powers[idx]) / (self.powers[idx + 1] - self.powers[idx])
        rows = np.arange(c.shape[0])
        return c[rows, idx] * (1 - f) + c[rows, idx + 1] * f

    def ceiling(self, temp, elapsed, top_bpm, which: str = "mid"):
        """Power where the heart reaches `top_bpm` (low: with the 90 % heart rate, high: with the 10 %)."""
        k = {"low": 2, "mid": 1, "high": 0}[which]
        c = self.curves(temp, elapsed)[..., k]
        top = np.broadcast_to(np.atleast_1d(np.asarray(top_bpm, float)), (c.shape[0],))
        above = c >= top[:, None]
        first = np.where(above.any(axis=1), above.argmax(axis=1), len(self.powers) - 1)
        j = np.clip(first, 1, len(self.powers) - 1)
        rows = np.arange(c.shape[0])
        h0, h1 = c[rows, j - 1], c[rows, j]
        f = np.clip((top - h0) / np.maximum(h1 - h0, 1e-9), 0.0, 1.0)
        out = self.powers[j - 1] + f * (self.powers[j] - self.powers[j - 1])
        out = np.where(first == 0, self.powers[0], out)
        return np.where(above.any(axis=1), out, self.powers[-1])

    def to_dict(self) -> dict:
        return {"temps": self.temps.tolist(), "elapsed": self.elapsed.tolist(), "powers": self.powers.tolist(),
                "q": np.round(self.q, 2).tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "HeartTable":
        return cls(temps=np.asarray(d["temps"], float), elapsed=np.asarray(d["elapsed"], float),
                   powers=np.asarray(d["powers"], float), q=np.asarray(d["q"], float))


@dataclass
class Limits:
    ceiling: Table2D  # x: air temperature °C, y: minutes already ridden → watts (at the learned top of Zone 2)
    cadence: Table2D  # x: grade, y: minutes already climbing → lowest cadence held (rpm)
    speed: Table2D  # x: grade, y: pedal power (W) → speed when riding easy (m/s)
    source: str = "fixed"
    notes: dict | None = None
    heart: HeartTable | None = None

    @property
    def top_bpm(self) -> float | None:
        return (self.notes or {}).get("z2_top_bpm")

    def day_range(self) -> tuple[float, float]:
        """How far the heart's level moves from one day to the next (bpm, low and high), measured on rides left out."""
        n = self.notes or {}
        return float(n.get("day_offset_low_bpm", -8.0)), float(n.get("day_offset_high_bpm", 8.0))

    def ceiling_w(self, temp, elapsed, which: str = "mid", offset_bpm: float = 0.0, top_bpm: float | None = None):
        """Zone 2 ceiling in watts when the heart runs `offset_bpm` above its usual level."""
        top = top_bpm if top_bpm is not None else self.top_bpm
        if self.heart is not None and top is not None:
            return self.heart.ceiling(temp, elapsed, top - offset_bpm, which)
        # Fixed limits: the ceiling moves by 1/slope watts per bpm.
        slope = float((self.notes or {}).get("bpm_per_w", 0.4))
        moved_top = (top - self.top_bpm) if (top is not None and self.top_bpm is not None) else 0.0
        return self.ceiling(temp, elapsed, which) + (moved_top - offset_bpm) / slope

    def day_offset(self, hr, p60, p300, temp, elapsed, moving, top_bpm: float | None = None) -> float:
        """The day's level: median of (heart rate − usual heart rate at that power) on steady stretches."""
        hr = np.asarray(hr, float)
        steady = (np.asarray(moving, bool) & np.isfinite(hr) & (np.abs(np.asarray(p60) - np.asarray(p300)) < 15.0)
                  & (np.asarray(p60) >= 60.0) & (np.asarray(p60) <= 260.0))
        if steady.sum() < 300:
            return 0.0
        idx = np.flatnonzero(steady)[::10]  # one second in ten is plenty
        if self.heart is not None:
            usual = self.heart.hr(np.asarray(temp)[idx], np.asarray(elapsed)[idx], np.asarray(p60)[idx], "mid")
        else:
            slope = float((self.notes or {}).get("bpm_per_w", 0.4))
            top = top_bpm if top_bpm is not None else (self.top_bpm or 145.0)
            ceil = self.ceiling_w(np.asarray(temp)[idx], np.asarray(elapsed)[idx], "mid", 0.0, top)
            usual = top + slope * (np.asarray(p60)[idx] - ceil)
        return float(np.median(hr[idx] - usual))

    def save(self, path: str | Path) -> None:
        data = {"source": self.source, "notes": self.notes or {},
                "ceiling": self.ceiling.to_dict(), "cadence": self.cadence.to_dict(), "speed": self.speed.to_dict()}
        if self.heart is not None:
            data["heart"] = self.heart.to_dict()
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Limits":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(ceiling=Table2D.from_dict(d["ceiling"]), cadence=Table2D.from_dict(d["cadence"]),
                   speed=Table2D.from_dict(d["speed"]), source=d.get("source", "?"), notes=d.get("notes"),
                   heart=HeartTable.from_dict(d["heart"]) if "heart" in d else None)


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
