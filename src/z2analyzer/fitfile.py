"""Rides from the original FIT file of the bike computer, on a one-second grid.

Read with fitdecode, which handles developer fields and compressed timestamp
headers. Indoor rides are refused: a trainer app's file has no real slope
(Zwift writes manufacturer 260 in the file header).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SEMICIRCLE = 180.0 / 2**31
VIRTUAL_MANUFACTURERS = {"zwift", "260", "wahoo_fitness_virtual", "tacx", "rouvy"}
CHANNELS = ("lat", "lon", "alt", "dist", "speed", "power", "hr", "cadence", "temp")


@dataclass
class Ride:
    """One-second samples from the first record to the last; NaN where the file has a gap."""

    name: str
    start: datetime  # UTC
    t: np.ndarray  # seconds since start
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray  # barometric altitude of the computer
    dist: np.ndarray  # metres, from the computer
    speed: np.ndarray  # m/s
    power: np.ndarray  # W
    hr: np.ndarray  # bpm
    cadence: np.ndarray  # rpm
    temp: np.ndarray  # °C, the computer's own sensor (warmed by the sun)
    manufacturer: str = ""
    product: str = ""
    sport: str = ""
    sub_sport: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def has_power(self) -> bool:
        return bool(np.nanmax(np.nan_to_num(self.power, nan=0.0), initial=0.0) > 0)

    @property
    def is_virtual(self) -> bool:
        if self.manufacturer.lower() in VIRTUAL_MANUFACTURERS:
            return True
        if self.sub_sport.lower() in {"virtual_activity", "indoor_cycling", "spin", "trainer"}:
            return True
        return not np.isfinite(self.lat).any()

    def time_at(self, seconds: float) -> datetime:
        return self.start + timedelta(seconds=float(seconds))


def read_fit(path: str | Path) -> Ride:
    import fitdecode

    rows: dict[int, dict] = {}
    file_id: dict = {}
    session: dict = {}
    first = None
    with fitdecode.FitReader(str(path), check_crc=fitdecode.CrcCheck.WARN) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name == "file_id":
                file_id = {f.name: f.value for f in frame.fields}
            elif frame.name == "session" and not session:
                session = {f.name: f.value for f in frame.fields}
            elif frame.name == "record":
                ts = _get(frame, "timestamp")
                if ts is None:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if first is None:
                    first = ts
                sec = int(round((ts - first).total_seconds()))
                if sec < 0:
                    continue
                rows[sec] = {
                    "lat": _semi(_get(frame, "position_lat")),
                    "lon": _semi(_get(frame, "position_long")),
                    "alt": _first(frame, "enhanced_altitude", "altitude"),
                    "dist": _get(frame, "distance"),
                    "speed": _first(frame, "enhanced_speed", "speed"),
                    "power": _get(frame, "power"),
                    "hr": _get(frame, "heart_rate"),
                    "cadence": _get(frame, "cadence"),
                    "temp": _get(frame, "temperature"),
                }
    if first is None:
        raise ValueError(f"{path}: no records")
    n = max(rows) + 1
    arrays = {k: np.full(n, np.nan) for k in CHANNELS}
    for sec, r in rows.items():
        for k in CHANNELS:
            v = r[k]
            if v is not None:
                arrays[k][sec] = float(v)
    return Ride(
        name=Path(path).stem,
        start=first,
        t=np.arange(n, dtype=float),
        manufacturer=str(file_id.get("manufacturer", "")),
        product=str(file_id.get("garmin_product", file_id.get("product", ""))),
        sport=str(session.get("sport", "")),
        sub_sport=str(session.get("sub_sport", "")),
        meta={"samples": len(rows)},
        **arrays,
    )


def moving_mask(ride: Ride, min_speed_ms: float = 1.0):
    """Seconds spent rolling (stops and gaps excluded)."""
    return np.nan_to_num(ride.speed, nan=0.0) >= min_speed_ms


def fill_short_gaps(x, max_gap: int = 5):
    """Linear interpolation across gaps of at most `max_gap` seconds; longer gaps stay NaN."""
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if not bad.any() or bad.all():
        return x
    idx = np.arange(x.size)
    starts = np.flatnonzero(bad & ~np.concatenate([[False], bad[:-1]]))
    ends = np.flatnonzero(bad & ~np.concatenate([bad[1:], [False]]))
    good = ~bad
    filled = np.interp(idx, idx[good], x[good])
    for s, e in zip(starts, ends):
        if e - s + 1 <= max_gap and s > 0 and e < x.size - 1:
            x[s : e + 1] = filled[s : e + 1]
    return x


def _get(frame, name):
    try:
        return frame.get_value(name, fallback=None)
    except Exception:
        return None


def _first(frame, *names):
    for n in names:
        v = _get(frame, n)
        if v is not None:
            return v
    return None


def _semi(v):
    return None if v is None else v * SEMICIRCLE
