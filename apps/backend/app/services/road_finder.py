"""Core pipeline: find steep roads around a point."""

import math
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import requests
from scipy.ndimage import gaussian_filter1d

from app.config import settings
from app.services.elevation import LocalDEM
from app.services.geometry import haversine_distance, line_length_meters, interpolate_line

RUNNING_HIGHWAY_TYPES = [
    "footway", "path", "steps", "pedestrian",
    "residential", "living_street", "track", "service", "unclassified",
]

EXCLUDE_HIGHWAY_TYPES = [
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "trunk", "trunk_link",
    "motorway", "motorway_link",
]

EXCLUDE_NAME_KEYWORDS = ["tunnel", "autoroute"]


# -- OSM fetch ---------------------------------------------------------------

def fetch_running_roads(lat: float, lon: float, radius_m: float) -> dict:
    highway_filter = "|".join(RUNNING_HIGHWAY_TYPES)
    query = f"""
    [out:json][timeout:180];
    (
      way["highway"~"^({highway_filter})$"](around:{radius_m},{lat},{lon});
    );
    out body geom;
    """

    for attempt in range(3):
        try:
            resp = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                timeout=240,
            )
            if resp.status_code in (429, 504):
                time.sleep((attempt + 1) * 15)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            if attempt < 2:
                time.sleep(10)
            else:
                raise

    return {"elements": []}


# -- Slope calculation -------------------------------------------------------

def calculate_road_slope(
    coords: list[tuple[float, float]],
    elevations: np.ndarray,
) -> Optional[dict]:
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

    sigma = max(settings.smoothing_sigma, len(elevs) // 10)
    if len(elevs) > sigma * 2:
        elevs = gaussian_filter1d(elevs, sigma=sigma, mode="nearest")

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


# -- Data structures ---------------------------------------------------------

@dataclass
class SteepRoad:
    name: str
    highway_type: str
    length_m: float
    elevation_gain_m: float
    elevation_loss_m: float
    avg_slope_percent: float
    max_slope_percent: float
    start_elevation_m: float
    end_elevation_m: float
    geometry: list[tuple[float, float]]
    osm_id: int
    profile: list[dict]
    incline_tag: Optional[str] = None
    surface: Optional[str] = None
    access: Optional[str] = None


# -- Main pipeline -----------------------------------------------------------

def find_steep_roads(
    lat: float,
    lon: float,
    radius_m: float = 3000,
    min_avg_slope: float = 5.0,
    min_max_slope: float = 8.0,
    min_length: float | None = None,
    named_only: bool = False,
    dem_source: str = "srtm",
) -> list[SteepRoad]:
    if min_length is None:
        min_length = settings.min_segment_length_m

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
        if length < min_length:
            continue

        sampled = interpolate_line(coords, settings.sample_step_m)

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

    dem = LocalDEM()
    all_elevations = dem.get_elevations_batch(all_coords)

    results: list[SteepRoad] = []

    for road in road_data:
        start_idx = road["coord_start_idx"]
        n_points = len(road["coords"])
        elevations = all_elevations[start_idx : start_idx + n_points]

        stats = calculate_road_slope(road["coords"], elevations)
        if stats is None:
            continue

        osm_incline = road.get("incline")
        has_osm_tag = False
        if osm_incline:
            try:
                incline_str = str(osm_incline).lower().strip()
                if incline_str in ("steep", "yes"):
                    incline_val = 12.0
                    has_osm_tag = True
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

        if not has_osm_tag and stats["length_m"] < settings.min_length_for_steep:
            if total_elev_change < stats["length_m"] * 0.15:
                continue

        expected_slope_from_delta = (total_elev_change / stats["length_m"]) * 100
        if not has_osm_tag and stats["avg_slope"] > 8:
            if expected_slope_from_delta < stats["avg_slope"] * settings.min_slope_elevation_ratio:
                continue

        if named_only and not road["name"] and not has_osm_tag:
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


def roads_to_response(roads: list[SteepRoad]) -> dict:
    features = []
    profiles: dict[int, list[dict]] = {}
    for road in roads:
        props = asdict(road)
        del props["geometry"]
        del props["profile"]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [list(c) for c in road.geometry],
            },
            "properties": props,
        })
        profiles[road.osm_id] = road.profile
    return {
        "geojson": {"type": "FeatureCollection", "features": features},
        "profiles": profiles,
    }
