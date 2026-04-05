"""Local SRTM DEM handling — downloads and caches .hgt tiles."""

import gzip
import math
import os
import shutil

import numpy as np
import requests

from app.config import settings


class LocalDEM:
    """Fast local DEM using downloaded SRTM tiles."""

    def __init__(self, dem_dir: str | None = None):
        self.dem_dir = dem_dir or settings.dem_dir
        self.tiles: dict[str, np.ndarray] = {}
        os.makedirs(self.dem_dir, exist_ok=True)

    def _tile_name(self, lat: float, lon: float) -> str:
        lat_int = int(math.floor(lat))
        lon_int = int(math.floor(lon))
        lat_prefix = "N" if lat_int >= 0 else "S"
        lon_prefix = "E" if lon_int >= 0 else "W"
        return f"{lat_prefix}{abs(lat_int):02d}{lon_prefix}{abs(lon_int):03d}"

    def _download_tile(self, tile_name: str) -> bool:
        hgt_path = os.path.join(self.dem_dir, f"{tile_name}.hgt")
        if os.path.exists(hgt_path):
            return True

        url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{tile_name[:3]}/{tile_name}.hgt.gz"
        try:
            resp = requests.get(url, stream=True, timeout=60)
            if resp.status_code != 200:
                return False

            gz_path = os.path.join(self.dem_dir, f"{tile_name}.hgt.gz")
            with open(gz_path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)

            with gzip.open(gz_path, "rb") as f_in, open(hgt_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            os.remove(gz_path)
            return True
        except Exception:
            return False

    def _load_tile(self, tile_name: str) -> bool:
        if tile_name in self.tiles:
            return True

        hgt_path = os.path.join(self.dem_dir, f"{tile_name}.hgt")
        if not os.path.exists(hgt_path):
            if not self._download_tile(tile_name):
                return False

        try:
            with open(hgt_path, "rb") as f:
                data = f.read()

            size = int(math.sqrt(len(data) / 2))
            arr = np.frombuffer(data, dtype=">i2").reshape((size, size)).astype(np.float32)
            arr[arr == -32768] = np.nan
            self.tiles[tile_name] = arr
            return True
        except Exception:
            return False

    def get_elevation(self, lat: float, lon: float) -> float:
        tile_name = self._tile_name(lat, lon)
        if not self._load_tile(tile_name):
            return float("nan")

        arr = self.tiles[tile_name]
        size = arr.shape[0]

        lat_int = int(math.floor(lat))
        lon_int = int(math.floor(lon))

        col = (lon - lon_int) * (size - 1)
        row = (lat_int + 1 - lat) * (size - 1)

        c0, r0 = int(col), int(row)
        c1, r1 = min(c0 + 1, size - 1), min(r0 + 1, size - 1)

        if r0 < 0 or r0 >= size or c0 < 0 or c0 >= size:
            return float("nan")

        dc, dr = col - c0, row - r0

        v00, v01 = arr[r0, c0], arr[r0, c1]
        v10, v11 = arr[r1, c0], arr[r1, c1]

        if np.isnan(v00) or np.isnan(v01) or np.isnan(v10) or np.isnan(v11):
            return float(arr[int(round(row)), int(round(col))])

        return float(
            v00 * (1 - dc) * (1 - dr)
            + v01 * dc * (1 - dr)
            + v10 * (1 - dc) * dr
            + v11 * dc * dr
        )

    def get_elevations_batch(self, coords: list[tuple[float, float]]) -> np.ndarray:
        """Get elevations for multiple points — coords are (lon, lat)."""
        return np.array([self.get_elevation(lat, lon) for lon, lat in coords])
