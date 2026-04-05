"""v4 — EU-DEM 25m via OpenTopoData API + Savitzky-Golay smoothing, mapzen fallback."""

import math
import time

import numpy as np
import requests
from scipy.signal import savgol_filter

from app.config import settings
from app.services.geometry import haversine_distance, line_length_meters, interpolate_line
from app.services.road_finder import (
    SteepRoad,
    RUNNING_HIGHWAY_TYPES,
    EXCLUDE_HIGHWAY_TYPES,
    EXCLUDE_NAME_KEYWORDS,
    fetch_running_roads,
)


def _fetch_elevations_opentopodata(
    coords: list[tuple[float, float]],
    dataset: str = "eudem25m",
    batch_size: int = 100,
) -> np.ndarray:
    """Fetch elevations from OpenTopoData API."""
    elevations = np.full(len(coords), np.nan)
    base_url = f"https://api.opentopodata.org/v1/{dataset}"

    for i in range(0, len(coords), batch_size):
        batch = coords[i : i + batch_size]
        locations = "|".join(f"{lat},{lon}" for lon, lat in batch)

        try:
            resp = requests.get(
                base_url,
                params={"locations": locations},
                timeout=30,
            )
            if resp.status_code == 429:
                time.sleep(2)
                resp = requests.get(
                    base_url,
                    params={"locations": locations},
                    timeout=30,
                )

            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "OK":
                    for j, result in enumerate(data.get("results", [])):
                        elev = result.get("elevation")
                        if elev is not None:
                            elevations[i + j] = float(elev)
        except Exception:
            pass

        if i + batch_size < len(coords):
            time.sleep(1.1)

    return elevations


def _fetch_elevations_with_fallback(
    coords: list[tuple[float, float]],
) -> np.ndarray:
    """Try EU-DEM 25m first, fill gaps with mapzen."""
    elevations = _fetch_elevations_opentopodata(coords, dataset="eudem25m")

    nan_count = int(np.sum(np.isnan(elevations)))
    if nan_count > len(coords) * 0.2:
        fallback = _fetch_elevations_opentopodata(coords, dataset="mapzen")
        mask = np.isnan(elevations)
        elevations[mask] = fallback[mask]

    return elevations


def _calculate_slope_savgol(
    coords: list[tuple[float, float]],
    elevations: np.ndarray,
) -> dict | None:
    """Slope calculation using Savitzky-Golay filter (preserves peaks better than Gaussian)."""
    if len(coords) < 3 or len(elevations) < 3:
        return None

    distances = [0.0]
    for i in range(1, len(coords)):
        d = haversine_distance(coords[i - 1][1], coords[i - 1][0], coords[i][1], coords[i][0])
        distances.append(distances[-1] + d)
    distances = np.array(distances)
    total_length = distances[-1]

    if total_length < settings.min_segment_length_m:
        return None

    valid = ~np.isnan(elevations)
    if valid.sum() < 3:
        return None

    elevs = np.copy(elevations)
    if not valid.all():
        elevs = np.interp(distances, distances[valid], elevations[valid])

    # Savitzky-Golay: preserves peaks unlike Gaussian
    # Window must be odd, at least 3, at most len(elevs)
    window = min(max(5, len(elevs) // 5 | 1), len(elevs))
    if window % 2 == 0:
        window -= 1
    if window >= 3 and len(elevs) >= window:
        elevs = savgol_filter(elevs, window_length=window, polyorder=2, mode="nearest")

    dz = np.diff(elevs)
    dd = np.diff(distances)
    dd = np.where(dd < 0.1, 0.1, dd)
    slopes = (dz / dd) * 100
    slopes = np.clip(slopes, -settings.max_realistic_slope, settings.max_realistic_slope)

    abs_slopes = np.abs(slopes)
    weights = np.ones(len(slopes))
    if len(slopes) > 4:
        weights[:2] = 0.5
        weights[-2:] = 0.5

    avg_slope = float(np.average(abs_slopes, weights=weights))
    max_slope = float(np.max(abs_slopes))

    total_elevation_change = abs(elevs[-1] - elevs[0])
    elevation_gain = max(0.0, float(elevs[-1] - elevs[0]))
    elevation_loss = max(0.0, float(elevs[0] - elevs[-1]))

    computed_slope_from_delta = (total_elevation_change / total_length) * 100
    avg_slope = min(avg_slope, computed_slope_from_delta * 1.3)

    profile: list[dict] = []
    for i in range(len(slopes)):
        profile.append({
            "d": round(float(distances[i]), 1),
            "e": round(float(elevs[i]), 1),
            "s": round(float(slopes[i]), 1),
        })
    profile.append({
        "d": round(float(distances[-1]), 1),
        "e": round(float(elevs[-1]), 1),
        "s": 0.0,
    })

    return {
        "length_m": total_length,
        "avg_slope": avg_slope,
        "max_slope": max_slope,
        "elevation_gain": elevation_gain,
        "elevation_loss": elevation_loss,
        "start_elevation": float(elevs[0]),
        "end_elevation": float(elevs[-1]),
        "profile": profile,
    }


class V4CopernicusFinder:
    def find(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        min_avg_slope: float,
    ) -> list[SteepRoad]:
        osm_data = fetch_running_roads(lat, lon, radius_m)
        elements = osm_data.get("elements", [])
        if not elements:
            return []

        all_coords: list[tuple[float, float]] = []
        road_data: list[dict] = []

        for element in elements:
            if element.get("type") != "way" or "geometry" not in element:
                continue

            tags = element.get("tags", {})
            highway = tags.get("highway", "")
            name = tags.get("name", "")

            if highway in EXCLUDE_HIGHWAY_TYPES:
                continue
            if tags.get("tunnel"):
                continue
            if any(kw in name.lower() for kw in EXCLUDE_NAME_KEYWORDS):
                continue

            coords = [(p["lon"], p["lat"]) for p in element["geometry"]]
            if len(coords) < 2:
                continue

            length = line_length_meters(coords)
            if length < settings.min_segment_length_m:
                continue

            sampled = interpolate_line(coords, 30)  # 30m step (matches DEM resolution, fewer API calls)

            road_data.append({
                "osm_id": element.get("id"),
                "name": name,
                "highway": highway,
                "incline": tags.get("incline"),
                "surface": tags.get("surface"),
                "access": tags.get("access"),
                "coords": sampled,
                "coord_start_idx": len(all_coords),
            })
            all_coords.extend(sampled)

        if not all_coords:
            return []

        all_elevations = _fetch_elevations_with_fallback(all_coords)

        results: list[SteepRoad] = []
        min_max_slope = 8.0

        for road in road_data:
            start_idx = road["coord_start_idx"]
            n_points = len(road["coords"])
            elevations = all_elevations[start_idx : start_idx + n_points]

            stats = _calculate_slope_savgol(road["coords"], elevations)
            if stats is None:
                continue

            osm_incline = road.get("incline")
            has_osm_tag = False
            if osm_incline:
                try:
                    incline_str = str(osm_incline).lower().strip()
                    if incline_str in ("steep", "yes"):
                        has_osm_tag = True
                        stats["avg_slope"] = 12.0
                    elif incline_str not in ("up", "down"):
                        incline_val = float(
                            incline_str.replace("%", "").replace("\u00b0", "").replace(",", ".")
                        )
                        if "\u00b0" in str(osm_incline):
                            incline_val = math.tan(math.radians(abs(incline_val))) * 100
                        incline_val = abs(incline_val)
                        has_osm_tag = True
                        if incline_val > 0:
                            stats["avg_slope"] = incline_val
                            stats["max_slope"] = max(stats["max_slope"], incline_val)
                except (ValueError, TypeError):
                    pass

            if stats["avg_slope"] < min_avg_slope and stats["max_slope"] < min_max_slope:
                continue

            total_elev_change = max(stats["elevation_gain"], stats["elevation_loss"])
            if total_elev_change < settings.min_elevation_change_m and not has_osm_tag:
                continue

            results.append(SteepRoad(
                name=road["name"] if road["name"] else f"Unnamed ({road['highway']})",
                highway_type=road["highway"],
                length_m=round(stats["length_m"], 1),
                elevation_gain_m=round(stats["elevation_gain"], 1),
                elevation_loss_m=round(stats["elevation_loss"], 1),
                avg_slope_percent=round(stats["avg_slope"], 1),
                max_slope_percent=round(stats["max_slope"], 1),
                start_elevation_m=round(stats["start_elevation"], 1),
                end_elevation_m=round(stats["end_elevation"], 1),
                geometry=road["coords"],
                osm_id=road["osm_id"],
                profile=stats["profile"],
                incline_tag=osm_incline,
                surface=road["surface"],
                access=road["access"],
            ))

        results.sort(key=lambda x: x.avg_slope_percent, reverse=True)
        return results
