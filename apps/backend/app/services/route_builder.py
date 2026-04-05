"""Build loop routes through steep roads."""

import math

import numpy as np
import requests

from app.services.road_finder import find_steep_roads, SteepRoad
from app.services.geometry import haversine_distance, interpolate_line
from app.services.elevation import LocalDEM
from app.config import settings


DETOUR_FACTOR = 1.4
MAX_WAYPOINTS = 6


def _midpoint(road: SteepRoad) -> tuple[float, float]:
    """Return (lon, lat) midpoint of a road's geometry."""
    n = len(road.geometry)
    mid = road.geometry[n // 2]
    return mid  # (lon, lat)


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Bearing in degrees from point 1 to point 2."""
    dlon = math.radians(lon2 - lon1)
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360


def _estimate_loop_distance(
    start_lon: float, start_lat: float,
    waypoints: list[tuple[float, float]],
) -> float:
    """Estimate total loop distance through ordered waypoints (lon, lat)."""
    if not waypoints:
        return 0.0
    points = [(start_lon, start_lat)] + waypoints + [(start_lon, start_lat)]
    total = 0.0
    for i in range(len(points) - 1):
        total += haversine_distance(
            points[i][1], points[i][0],
            points[i + 1][1], points[i + 1][0],
        )
    return total * DETOUR_FACTOR


def _road_elevation_gain(road: SteepRoad) -> float:
    """Total elevation gain (sum of positive deltas) from the profile."""
    gain = 0.0
    for i in range(len(road.profile) - 1):
        de = road.profile[i + 1]["e"] - road.profile[i]["e"]
        if de > 0:
            gain += de
    return max(gain, road.elevation_gain_m)


def _select_and_order_roads(
    roads: list[SteepRoad],
    start_lat: float,
    start_lon: float,
    target_distance_m: float,
    target_elevation_gain_m: float,
) -> list[SteepRoad]:
    """Greedily select roads to match elevation gain target within distance budget."""
    if not roads:
        return []

    # Pre-compute gains and distances
    road_gains = {id(r): _road_elevation_gain(r) for r in roads}

    selected: list[SteepRoad] = []
    cumulative_gain = 0.0
    used = set()

    while len(selected) < MAX_WAYPOINTS and cumulative_gain < target_elevation_gain_m:
        best_road = None
        best_score = -1.0

        for road in roads:
            if id(road) in used:
                continue

            gain = road_gains[id(road)]
            if gain < 1:
                continue

            mid = _midpoint(road)
            mid_lon, mid_lat = mid

            # Check distance budget
            current_waypoints = [_midpoint(r) for r in selected]
            candidate_waypoints = current_waypoints + [mid]
            est_dist = _estimate_loop_distance(start_lon, start_lat, candidate_waypoints)
            if est_dist > target_distance_m:
                continue

            dist_to_start = haversine_distance(start_lat, start_lon, mid_lat, mid_lon)
            proximity = 1.0 / (1.0 + dist_to_start / 500)

            # How much of the remaining gain target does this road fill?
            remaining = target_elevation_gain_m - cumulative_gain
            gain_value = min(gain, remaining) / max(remaining, 1)

            # Score: prioritize filling the elevation target, weighted by proximity
            score = gain_value * 0.7 + proximity * 0.3

            if score > best_score:
                best_score = score
                best_road = road

        if best_road is None:
            break

        selected.append(best_road)
        used.add(id(best_road))
        cumulative_gain += road_gains[id(best_road)]

    # Order by bearing angle from start to form a loop
    def road_bearing(road: SteepRoad) -> float:
        mid_lon, mid_lat = _midpoint(road)
        return _bearing(start_lat, start_lon, mid_lat, mid_lon)

    selected.sort(key=road_bearing)
    return selected


def _build_waypoint_coords(
    start_lat: float, start_lon: float,
    roads: list[SteepRoad],
) -> list[tuple[float, float]]:
    """Build ordered coordinate list: start → road start/end pairs → start.

    Routes through both ends of each steep road so the route actually
    traverses the steep segment rather than just passing near its midpoint.
    """
    coords = [(start_lon, start_lat)]
    for road in roads:
        road_start = road.geometry[0]
        road_end = road.geometry[-1]
        # Add the end closest to the previous waypoint first
        prev = coords[-1]
        d_to_start = haversine_distance(prev[1], prev[0], road_start[1], road_start[0])
        d_to_end = haversine_distance(prev[1], prev[0], road_end[1], road_end[0])
        if d_to_start <= d_to_end:
            coords.append(road_start)
            coords.append(road_end)
        else:
            coords.append(road_end)
            coords.append(road_start)
    coords.append((start_lon, start_lat))
    return coords


def _route_via_osrm(
    coords: list[tuple[float, float]],
    mode: str,
) -> dict | None:
    """Call OSRM public API to route through waypoints. Returns route dict or None."""
    profile = "foot" if mode == "foot" else "bike"
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords)
    url = f"https://router.project-osrm.org/route/v1/{profile}/{coord_str}"

    try:
        resp = requests.get(url, params={
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        return data["routes"][0]
    except (requests.RequestException, KeyError, IndexError):
        return None


def _compute_route_profile(
    geometry_coords: list[list[float]],
) -> dict | None:
    """Compute elevation profile for an OSRM route geometry.

    Unlike calculate_road_slope (designed for short steep segments),
    this computes cumulative D+ and D- over the full route with
    light smoothing appropriate for multi-km routes.
    """
    coords = [(c[0], c[1]) for c in geometry_coords]

    sampled = interpolate_line(coords, settings.sample_step_m)
    if len(sampled) < 3:
        return None

    dem = LocalDEM()
    elevations = dem.get_elevations_batch(sampled)

    # Interpolate NaN values
    from scipy.ndimage import gaussian_filter1d

    distances = [0.0]
    for i in range(1, len(sampled)):
        d = haversine_distance(sampled[i - 1][1], sampled[i - 1][0],
                               sampled[i][1], sampled[i][0])
        distances.append(distances[-1] + d)
    distances_arr = np.array(distances)

    valid = ~np.isnan(elevations)
    if valid.sum() < 3:
        return None

    elevs = np.copy(elevations)
    if not valid.all():
        elevs = np.interp(distances_arr, distances_arr[valid], elevations[valid])

    # Light smoothing — sigma proportional to route length but gentle
    # For a 5km route (~333 points at 15m), sigma=5 smooths ~75m
    sigma = max(3, min(len(elevs) // 60, 8))
    if len(elevs) > sigma * 2:
        elevs = gaussian_filter1d(elevs, sigma=sigma, mode="nearest")

    # Compute cumulative D+ and D- (sum of all ups and downs)
    total_gain = 0.0
    total_loss = 0.0
    for i in range(1, len(elevs)):
        de = elevs[i] - elevs[i - 1]
        if de > 0:
            total_gain += de
        else:
            total_loss += abs(de)

    # Build profile with slopes
    dz = np.diff(elevs)
    dd = np.diff(distances_arr)
    dd = np.where(dd < 0.1, 0.1, dd)
    slopes = np.clip((dz / dd) * 100, -18, 18)

    profile = []
    for i in range(len(slopes)):
        profile.append({
            "d": round(float(distances_arr[i]), 1),
            "e": round(float(elevs[i]), 1),
            "s": round(float(slopes[i]), 1),
        })
    profile.append({
        "d": round(float(distances_arr[-1]), 1),
        "e": round(float(elevs[-1]), 1),
        "s": 0.0,
    })

    return {
        "elevation_gain": total_gain,
        "elevation_loss": total_loss,
        "profile": profile,
    }


def build_route(
    lat: float,
    lon: float,
    target_distance_km: float,
    target_elevation_gain_m: float,
    mode: str = "foot",
) -> dict:
    """Main entry point: generate a loop route through steep roads."""
    target_distance_m = target_distance_km * 1000

    # Search radius: ~1/3 of target distance (loop diameter), capped at 8km
    search_radius = min(target_distance_m / 3, 8000)

    is_cycling = mode == "bicycle"

    # Cyclists cover more ground, so search wider
    if is_cycling:
        search_radius = min(search_radius * 1.5, 10000)

    roads = find_steep_roads(
        lat=lat,
        lon=lon,
        radius_m=search_radius,
        min_avg_slope=3.0,
        min_max_slope=5.0,
        min_length=30,
    )

    if not roads:
        raise ValueError("No steep roads found in the area")

    # Filter out roads unsuitable for cycling (steps, narrow footways)
    if is_cycling:
        CYCLING_EXCLUDE = {"steps", "footway", "pedestrian"}
        roads = [r for r in roads if r.highway_type not in CYCLING_EXCLUDE]
        if not roads:
            raise ValueError("No cycleable steep roads found in the area")

    selected = _select_and_order_roads(
        roads, lat, lon, target_distance_m, target_elevation_gain_m,
    )

    if not selected:
        raise ValueError("Could not build a route matching the constraints")

    waypoint_coords = _build_waypoint_coords(lat, lon, selected)
    osrm_route = _route_via_osrm(waypoint_coords, mode)

    if osrm_route is None:
        raise ValueError("Routing service unavailable, please try again")

    route_geom = osrm_route["geometry"]["coordinates"]
    route_distance_m = osrm_route["distance"]
    route_duration_s = osrm_route["duration"]

    profile_stats = _compute_route_profile(route_geom)

    total_gain = 0.0
    total_loss = 0.0
    profile = []
    if profile_stats:
        total_gain = profile_stats["elevation_gain"]
        total_loss = profile_stats["elevation_loss"]
        profile = profile_stats["profile"]

    steep_segments = [
        {
            "name": r.name,
            "length_m": r.length_m,
            "elevation_gain_m": r.elevation_gain_m,
            "avg_slope_percent": r.avg_slope_percent,
        }
        for r in selected
    ]

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": osrm_route["geometry"],
            "properties": {},
        }],
    }

    return {
        "geojson": geojson,
        "total_distance_m": round(route_distance_m, 1),
        "total_elevation_gain_m": round(total_gain, 1),
        "total_elevation_loss_m": round(total_loss, 1),
        "estimated_duration_min": round(route_duration_s / 60, 1),
        "steep_segments": steep_segments,
        "profile": profile,
    }
