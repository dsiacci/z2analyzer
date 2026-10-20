"""GPX routes and tracks: positions only (elevations in a GPX are noisy and get replaced)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Route:
    name: str
    lat: np.ndarray
    lon: np.ndarray
    ele: np.ndarray  # NaN where the file has none


def read_gpx(path: str | Path) -> Route:
    """Track points if there are any, else route points. Several segments are joined in order."""
    root = ET.parse(path).getroot()
    pts = root.findall(".//{*}trkpt") or root.findall(".//{*}rtept")
    if not pts:
        raise ValueError(f"{path}: no track or route points")
    lat = np.array([float(p.get("lat")) for p in pts])
    lon = np.array([float(p.get("lon")) for p in pts])
    ele = np.array([_float(p.find("{*}ele")) for p in pts])
    name = ""
    for query in (".//{*}trk/{*}name", ".//{*}rte/{*}name", ".//{*}metadata/{*}name"):
        el = root.find(query)  # an element without children is falsy: test against None
        if el is not None and (el.text or "").strip():
            name = el.text.strip()
            break
    return Route(name=name or Path(path).stem, lat=lat, lon=lon, ele=ele)


def _float(el) -> float:
    try:
        return float(el.text)
    except (AttributeError, TypeError, ValueError):
        return float("nan")
