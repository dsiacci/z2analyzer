"""Air temperature from Open-Meteo (CC BY 4.0), Météo-France models, at one rounded point.

The only thing that leaves the machine for the weather is one point rounded to
0.1° (about 11 km) and a date. The temperature is then moved to the road's
elevation along the ride (−6.5 °C per 1,000 m).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np

FORECAST = "https://api.open-meteo.com/v1/forecast"
ARCHIVE = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MODEL = "meteofrance_seamless"
USER_AGENT = "z2analyzer (+https://github.com/dsiacci/z2analyzer)"


@dataclass
class Hourly:
    """Hourly 2 m temperature at one point (times in UTC)."""

    lat: float
    lon: float
    elevation_m: float
    times: list[datetime]
    temp_c: np.ndarray

    def hours_since(self, origin: datetime) -> np.ndarray:
        return np.array([(t - origin).total_seconds() / 3600.0 for t in self.times])

    def at(self, when, elev_m=None):
        """Temperature at UTC datetimes (or seconds since the epoch), optionally moved to `elev_m`."""
        if isinstance(when, datetime):
            when = np.array([when.timestamp()])
        x = np.asarray(when, dtype=float)
        tx = np.array([t.timestamp() for t in self.times])
        out = np.interp(x, tx, self.temp_c)
        if elev_m is not None:
            out = out - 0.0065 * (np.asarray(elev_m, dtype=float) - self.elevation_m)
        return out


def rounded(lat: float, lon: float, step: float = 0.1) -> tuple[float, float]:
    return (round(round(lat / step) * step, 4), round(round(lon / step) * step, 4))


def get_hourly(lat: float, lon: float, day_from: date, day_to: date, archive: bool) -> Hourly:
    lat, lon = rounded(lat, lon)
    params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "models": MODEL,
              "start_date": day_from.isoformat(), "end_date": day_to.isoformat(), "timezone": "GMT"}
    url = (ARCHIVE if archive else FORECAST) + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    h = d["hourly"]
    times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in h["time"]]
    temps = np.array([np.nan if v is None else float(v) for v in h["temperature_2m"]])
    ok = np.isfinite(temps)
    if not ok.any():
        raise RuntimeError(f"Open-Meteo has no temperature for {day_from}..{day_to}")
    return Hourly(lat=lat, lon=lon, elevation_m=float(d.get("elevation", 0.0)),
                  times=[t for t, k in zip(times, ok) if k], temp_c=temps[ok])
