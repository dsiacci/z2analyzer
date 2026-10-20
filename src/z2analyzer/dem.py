"""Terrain elevation from local tiles: the IGN RGE ALTI® (ESRI ASCII grids in Lambert-93).

The tiles are downloaded once (one archive per French département, Licence
Ouverte 2.0) and read from disk: no track point is ever sent anywhere to get an
elevation. Each tile is parsed once and cached as a .npy file next to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geo import lambert93


@dataclass
class Tile:
    path: Path
    ncols: int
    nrows: int
    x0: float  # centre of the lower-left cell
    y0: float
    cell: float
    nodata: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of the cells' outer edges (neighbouring tiles meet there)."""
        h = self.cell / 2
        return (self.x0 - h, self.y0 - h, self.x0 + (self.ncols - 1) * self.cell + h,
                self.y0 + (self.nrows - 1) * self.cell + h)


def read_header(path: Path) -> Tile:
    keys: dict[str, float] = {}
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for _ in range(6):
            parts = f.readline().split()
            if len(parts) == 2:
                keys[parts[0].lower()] = float(parts[1])
    cell = keys["cellsize"]
    if "xllcenter" in keys:
        x0, y0 = keys["xllcenter"], keys["yllcenter"]
    else:
        x0, y0 = keys["xllcorner"] + cell / 2, keys["yllcorner"] + cell / 2
    return Tile(path=Path(path), ncols=int(keys["ncols"]), nrows=int(keys["nrows"]), x0=x0, y0=y0, cell=cell,
                nodata=keys.get("nodata_value", -99999.0))


def read_grid(tile: Tile) -> np.ndarray:
    """Elevations as float32, row 0 at the top (north), NaN for no data. Cached as .npy."""
    cache = tile.path.with_suffix(".npy")
    if cache.exists() and cache.stat().st_mtime >= tile.path.stat().st_mtime:
        return np.load(cache)
    with open(tile.path, "r", encoding="ascii", errors="replace") as f:
        lines = f.readlines()
    header = 0
    while header < len(lines) and lines[header].split() and lines[header].split()[0][0].isalpha():
        header += 1  # five or six header lines, depending on NODATA_value
    body = " ".join(lines[header:])
    grid = np.array(body.split(), dtype=np.float32).reshape(tile.nrows, tile.ncols)
    grid[grid <= tile.nodata + 0.5] = np.nan
    try:
        np.save(cache, grid)
    except OSError:
        pass
    return grid


class Terrain:
    """Bilinear elevation lookup over a folder of tiles (searched recursively)."""

    def __init__(self, folder: str | Path, pattern: str = "*.asc"):
        self.tiles = [read_header(p) for p in sorted(Path(folder).rglob(pattern))]
        if not self.tiles:
            raise FileNotFoundError(f"no terrain tiles ({pattern}) under {folder}")
        self._cache: dict[Path, np.ndarray] = {}

    def elevation(self, lat, lon):
        x, y = lambert93(lat, lon)
        return self.elevation_xy(x, y)

    __call__ = elevation

    def elevation_xy(self, x, y):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        out = np.full(x.shape, np.nan)
        todo = np.ones(x.shape, dtype=bool)
        for t in self.tiles:
            xmin, ymin, xmax, ymax = t.bounds
            inside = todo & (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            if not inside.any():
                continue
            grid = self._grid(t)
            out[inside] = _bilinear(grid, t, x[inside], y[inside])
            todo &= ~(inside & np.isfinite(out))
        return out

    def _grid(self, t: Tile) -> np.ndarray:
        g = self._cache.get(t.path)
        if g is None:
            if len(self._cache) > 24:  # keep memory bounded on small machines
                self._cache.pop(next(iter(self._cache)))
            g = self._cache[t.path] = read_grid(t)
        return g


def _bilinear(grid: np.ndarray, t: Tile, x, y):
    # Within half a cell of the tile's edge, the edge value is held (no extrapolation).
    col = np.clip((x - t.x0) / t.cell, 0.0, t.ncols - 1.0)
    row_from_bottom = np.clip((y - t.y0) / t.cell, 0.0, t.nrows - 1.0)
    c0 = np.clip(np.floor(col).astype(int), 0, t.ncols - 2)
    r0 = np.clip(np.floor(row_from_bottom).astype(int), 0, t.nrows - 2)
    fc = col - c0
    fr = row_from_bottom - r0
    top = t.nrows - 1  # row index of the bottom line in the array
    z00 = grid[top - r0, c0]
    z01 = grid[top - r0, c0 + 1]
    z10 = grid[top - r0 - 1, c0]
    z11 = grid[top - r0 - 1, c0 + 1]
    return (z00 * (1 - fc) * (1 - fr) + z01 * fc * (1 - fr) + z10 * (1 - fc) * fr + z11 * fc * fr)
