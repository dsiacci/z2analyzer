"""From a FIT file to one-second series, then to the 30-second rows the models learn from.

Per second: distance along the track, grade and elevation from the rebuilt
profile, air temperature, minutes ridden, minutes into the current climb,
power smoothed over one and five minutes (the heart follows the legs with a
lag), and the gear recovered from speed and cadence.

Per 30 seconds, three tables:
- heart rate: smoothed power, air temperature, minutes ridden → heart rate;
- cadence: grade, minutes climbing → cadence, on climbs ridden in the smallest gear;
- speed: grade, power → speed.
None of them holds a position.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fitfile import Ride, fill_short_gaps
from .gears import gear_index, ratio_from
from .physics import Bike
from .profile import Profile, build_profile, track_distance
from .verdict import climb_minutes, ewma

CHUNK_S = 30
HR_FEATURES = ("p60", "p300", "temp", "elapsed_min")
CADENCE_FEATURES = ("grade", "climb_min")
SPEED_FEATURES = ("grade", "power")


@dataclass
class RideData:
    name: str
    start_epoch: float
    moving: np.ndarray
    dist: np.ndarray
    grade: np.ndarray
    elev: np.ndarray
    speed: np.ndarray
    power: np.ndarray
    hr: np.ndarray
    cadence: np.ndarray
    temp_air: np.ndarray
    temp_edge: np.ndarray
    alt_baro: np.ndarray
    elapsed_min: np.ndarray
    climb_min: np.ndarray
    p60: np.ndarray
    p300: np.ndarray
    gear: np.ndarray
    profile: Profile | None = None

    ARRAYS = ("moving", "dist", "grade", "elev", "speed", "power", "hr", "cadence", "temp_air", "temp_edge",
              "alt_baro", "elapsed_min", "climb_min", "p60", "p300", "gear")

    def save(self, path: str | Path) -> None:
        """Series without positions (the profile keeps its distances and grades only)."""
        extra = {}
        if self.profile is not None:
            extra = {"profile_dist": self.profile.dist, "profile_elev": self.profile.elev,
                     "profile_grade": self.profile.grade, "profile_structure": self.profile.on_structure}
        np.savez_compressed(path, name=self.name, start_epoch=self.start_epoch,
                            **{k: getattr(self, k) for k in self.ARRAYS}, **extra)

    @classmethod
    def load(cls, path: str | Path) -> "RideData":
        z = np.load(path, allow_pickle=False)
        prof = None
        if "profile_dist" in z:
            n = z["profile_dist"].size
            prof = Profile(dist=z["profile_dist"], lat=np.full(n, np.nan), lon=np.full(n, np.nan),
                           elev=z["profile_elev"], grade=z["profile_grade"], on_structure=z["profile_structure"])
        return cls(name=str(z["name"]), start_epoch=float(z["start_epoch"]), profile=prof,
                   **{k: z[k] for k in cls.ARRAYS})


def ride_data(ride: Ride, terrain, structures=None, temp_fn=None, bike: Bike | None = None) -> RideData:
    """Everything the models and the verdicts need from one ride.

    `temp_fn(epoch_seconds, elevation_m)` gives the air temperature; without
    it, the computer's own sensor is used.
    """
    bike = bike or Bike()
    lat = fill_short_gaps(ride.lat, 10)
    lon = fill_short_gaps(ride.lon, 10)
    ok = np.isfinite(lat) & np.isfinite(lon)
    profile = build_profile(lat[ok], lon[ok], terrain, structures)
    dist = track_distance(lat, lon)
    speed = np.nan_to_num(fill_short_gaps(ride.speed, 5), nan=0.0)
    moving = speed >= 1.0
    power = np.nan_to_num(fill_short_gaps(ride.power, 5), nan=0.0)
    cadence = np.nan_to_num(fill_short_gaps(ride.cadence, 5), nan=0.0)
    hr = fill_short_gaps(ride.hr, 10)
    grade = profile.grade_at(dist)
    elev = profile.elev_at(dist)
    epoch = ride.start.timestamp() + ride.t
    temp_edge = fill_short_gaps(ride.temp, 60)
    temp_air = temp_fn(epoch, elev) if temp_fn is not None else temp_edge
    elapsed = np.cumsum(moving) / 60.0
    gidx = gear_index(ratio_from(speed, cadence, bike.wheel_m), bike)
    return RideData(name=ride.name, start_epoch=ride.start.timestamp(), moving=moving, dist=dist, grade=grade,
                    elev=elev, speed=speed, power=power, hr=hr, cadence=cadence, temp_air=temp_air,
                    temp_edge=temp_edge, alt_baro=ride.alt, elapsed_min=elapsed,
                    climb_min=climb_minutes(grade, moving), p60=ewma(power, 60.0), p300=ewma(power, 300.0),
                    gear=np.asarray(gidx), profile=profile)


def chunks(rd: RideData, size_s: int = CHUNK_S) -> dict[str, np.ndarray]:
    """30-second means; `full` marks windows ridden without a stop and with a heart-rate reading."""
    n = rd.moving.size // size_s
    if n == 0:
        return {}
    cut = n * size_s

    def m(x):
        return np.nanmean(np.asarray(x[:cut], dtype=float).reshape(n, size_s), axis=1)

    moving_share = m(rd.moving.astype(float))
    hr_ok = m(np.isfinite(rd.hr).astype(float))
    pedalling = m((rd.cadence > 20).astype(float))
    d = rd.dist[:cut].reshape(n, size_s)
    run = d[:, -1] - d[:, 0]
    g = np.asarray(rd.gear[:cut]).reshape(n, size_s)
    lowest_share = (g == 0).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rise = rd.elev[:cut].reshape(n, size_s)
        grade = np.where(run > 5, (rise[:, -1] - rise[:, 0]) / run, m(rd.grade))
    return {
        "ride": np.full(n, rd.name),
        "start_s": np.arange(n) * size_s,
        "full": (moving_share >= 0.999) & (hr_ok >= 0.999),
        "pedalling": pedalling,
        "lowest_gear_share": lowest_share,
        "grade": grade,
        "power": m(rd.power),
        "speed": m(rd.speed),
        "cadence": m(np.where(rd.cadence > 20, rd.cadence, np.nan)),
        "hr": m(rd.hr),
        "p60": m(rd.p60),
        "p300": m(rd.p300),
        "temp": m(rd.temp_air),
        "temp_edge": m(rd.temp_edge),
        "elapsed_min": m(rd.elapsed_min),
        "climb_min": m(rd.climb_min),
        "dist_m": run,
    }


def tables(all_chunks: list[dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """The three training tables, filtered, with their ride labels for leave-one-ride-out."""
    c = {k: np.concatenate([ch[k] for ch in all_chunks if ch]) for k in all_chunks[0]}
    full = c["full"]
    hr_rows = full & np.isfinite(c["hr"])
    cad_rows = full & (c["grade"] >= 0.02) & (c["pedalling"] >= 0.8) & (c["lowest_gear_share"] >= 0.6) & np.isfinite(c["cadence"])
    spd_rows = full & np.isfinite(c["speed"]) & (c["speed"] > 1.0)
    pick = lambda rows, feats, target: {"X": np.column_stack([c[f][rows] for f in feats]).astype(np.float32),  # noqa: E731
                                        "y": c[target][rows].astype(np.float32), "ride": c["ride"][rows],
                                        "features": np.array(feats)}
    return {"hr": pick(hr_rows, HR_FEATURES, "hr"),
            "cadence": pick(cad_rows, CADENCE_FEATURES, "cadence"),
            "speed": pick(spd_rows, SPEED_FEATURES, "speed")}
