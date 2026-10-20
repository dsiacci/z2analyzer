"""Bridges and tunnels from OpenStreetMap (ODbL), so the profile does not dive into valleys.

The terrain model is the ground: on a bridge it has the river bed, in a tunnel
the mountain above. The road's own bridges and tunnels are fetched once for a
whole region (a box you choose, never the track) and kept in a local file.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter")
USER_AGENT = "z2analyzer (+https://github.com/dsiacci/z2analyzer)"


def fetch(bbox: tuple[float, float, float, float], out: str | Path, timeout_s: int = 180) -> int:
    """Download every road bridge and tunnel in (south, west, north, east) to a GeoJSON-like file."""
    s, w, n, e = bbox
    q = (f"[out:json][timeout:90][maxsize:268435456];"
         f'(way["highway"]["bridge"]["bridge"!="no"]({s},{w},{n},{e});'
         f'way["highway"]["tunnel"]["tunnel"!="no"]["tunnel"!="culvert"]({s},{w},{n},{e}););out geom;')
    last = None
    for url in OVERPASS:
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                data = json.load(r)
            break
        except Exception as exc:  # try the next server
            last = exc
    else:
        raise RuntimeError(f"Overpass did not answer: {last}")
    ways = []
    for el in data.get("elements", []):
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        tags = el.get("tags", {})
        kind = "tunnel" if tags.get("tunnel") not in (None, "no") else "bridge"
        ways.append({"kind": kind, "highway": tags.get("highway", ""),
                     "coords": [[round(p["lat"], 7), round(p["lon"], 7)] for p in geom]})
    Path(out).write_text(json.dumps({"bbox": bbox, "source": "© OpenStreetMap contributors, ODbL 1.0",
                                     "ways": ways}), encoding="utf-8")
    return len(ways)


class Structures:
    """Which track points run along a bridge or through a tunnel."""

    def __init__(self, path: str | Path, cell_m: float = 200.0):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.ways = data["ways"]
        lat0 = float(np.mean(data["bbox"][0::2]))
        self.k = np.cos(np.radians(lat0))
        segs = []
        for w in self.ways:
            c = np.array(w["coords"], dtype=float)
            xy = self._xy(c[:, 0], c[:, 1])
            segs.extend(zip(xy[:-1], xy[1:]))
        self.a = np.array([s[0] for s in segs]) if segs else np.zeros((0, 2))
        self.b = np.array([s[1] for s in segs]) if segs else np.zeros((0, 2))
        self.cell = cell_m
        self.grid: dict[tuple[int, int], list[int]] = {}
        for i, (p, q) in enumerate(zip(self.a, self.b)):
            for cx in range(int(min(p[0], q[0]) // cell_m), int(max(p[0], q[0]) // cell_m) + 1):
                for cy in range(int(min(p[1], q[1]) // cell_m), int(max(p[1], q[1]) // cell_m) + 1):
                    self.grid.setdefault((cx, cy), []).append(i)

    def _xy(self, lat, lon):
        r = 6_371_008.8
        return np.column_stack([np.radians(lon) * r * self.k, np.radians(lat) * r])

    def __call__(self, lat, lon, tol_m: float = 10.0, max_angle_deg: float = 35.0):
        """True where the track follows a structure (close to it and heading the same way)."""
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        out = np.zeros(lat.size, dtype=bool)
        if self.a.size == 0 or lat.size < 2:
            return out
        p = self._xy(lat, lon)
        heading = np.gradient(p, axis=0)
        cos_max = np.cos(np.radians(max_angle_deg))
        for i in range(lat.size):
            cell = (int(p[i, 0] // self.cell), int(p[i, 1] // self.cell))
            cand = {j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in self.grid.get((cell[0] + dx, cell[1] + dy), ())}
            for j in cand:
                a, b = self.a[j], self.b[j]
                ab = b - a
                l2 = float(ab @ ab)
                if l2 == 0:
                    continue
                u = min(1.0, max(0.0, float((p[i] - a) @ ab) / l2))
                if np.hypot(*(p[i] - (a + u * ab))) > tol_m:
                    continue
                h = heading[i]
                nh = np.hypot(*h)
                if nh == 0 or abs(float(h @ ab)) / (nh * np.sqrt(l2)) >= cos_max:
                    out[i] = True
                    break
        return out
