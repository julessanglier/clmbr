"""v5 — OSM tag-first: uses incline/ele tags where available, SRTM fallback."""

import math
import time

import numpy as np
import requests

from app.config import settings
from app.services.elevation import LocalDEM
from app.services.geometry import haversine_distance, line_length_meters, interpolate_line
from app.services.road_finder import (
    SteepRoad,
    EXCLUDE_HIGHWAY_TYPES,
    EXCLUDE_NAME_KEYWORDS,
    calculate_road_slope,
)

RUNNING_HIGHWAY_TYPES_EXTENDED = [
    "footway", "path", "steps", "pedestrian",
    "residential", "living_street", "track", "service", "unclassified",
    "cycleway", "bridleway", "tertiary",
]


def _fetch_roads_with_tags(lat: float, lon: float, radius_m: float) -> dict:
    """Fetch roads with incline/ele tags via Overpass, broader highway types."""
    highway_filter = "|".join(RUNNING_HIGHWAY_TYPES_EXTENDED)
    query = f"""
    [out:json][timeout:180];
    (
      way["highway"~"^({highway_filter})$"](around:{radius_m},{lat},{lon});
    );
    out body geom;
    >;
    out skel;
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


def _parse_incline(tag: str) -> float | None:
    """Parse an OSM incline tag value into a slope percent."""
    tag = tag.lower().strip()
    if tag in ("steep", "yes"):
        return 12.0
    if tag in ("up", "down", "no", "flat"):
        return None
    try:
        val = float(tag.replace("%", "").replace("\u00b0", "").replace(",", "."))
        if "\u00b0" in tag:
            val = math.tan(math.radians(abs(val))) * 100
        return abs(val)
    except (ValueError, TypeError):
        return None


class V5OsmTagsFinder:
    def find(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        min_avg_slope: float,
    ) -> list[SteepRoad]:
        osm_data = _fetch_roads_with_tags(lat, lon, radius_m)
        elements = osm_data.get("elements", [])
        if not elements:
            return []

        # Collect node elevations from `ele` tags
        node_elevations: dict[int, float] = {}
        for el in elements:
            if el.get("type") == "node":
                ele = el.get("tags", {}).get("ele")
                if ele:
                    try:
                        node_elevations[el["id"]] = float(ele)
                    except (ValueError, TypeError):
                        pass

        tagged_roads: list[SteepRoad] = []
        dem_roads_data: list[dict] = []
        all_dem_coords: list[tuple[float, float]] = []

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

            incline_tag = tags.get("incline")
            road_name = name if name else f"Unnamed ({highway})"

            # Priority 1: road has incline tag
            if incline_tag:
                slope = _parse_incline(incline_tag)
                if slope is not None and slope >= min_avg_slope:
                    sampled = interpolate_line(coords, settings.sample_step_m)
                    # Build a synthetic profile (linear slope)
                    elev_change = length * slope / 100
                    profile = []
                    for i, c in enumerate(sampled):
                        frac = i / max(len(sampled) - 1, 1)
                        d = frac * length
                        e = 100 + frac * elev_change  # synthetic start at 100m
                        profile.append({"d": round(d, 1), "e": round(e, 1), "s": round(slope, 1)})
                    profile[-1]["s"] = 0.0

                    tagged_roads.append(SteepRoad(
                        name=road_name,
                        highway_type=highway,
                        length_m=round(length, 1),
                        elevation_gain_m=round(elev_change, 1),
                        elevation_loss_m=0.0,
                        avg_slope_percent=round(slope, 1),
                        max_slope_percent=round(slope, 1),
                        start_elevation_m=100.0,
                        end_elevation_m=round(100 + elev_change, 1),
                        geometry=coords,
                        osm_id=element.get("id"),
                        profile=profile,
                        incline_tag=incline_tag,
                        surface=tags.get("surface"),
                        access=tags.get("access"),
                    ))
                    continue

            # Priority 2: start/end nodes have ele tags
            nodes = element.get("nodes", [])
            if len(nodes) >= 2:
                start_ele = node_elevations.get(nodes[0])
                end_ele = node_elevations.get(nodes[-1])
                if start_ele is not None and end_ele is not None:
                    elev_change = abs(end_ele - start_ele)
                    slope = (elev_change / length) * 100 if length > 0 else 0
                    if slope >= min_avg_slope:
                        sampled = interpolate_line(coords, settings.sample_step_m)
                        profile = []
                        for i, c in enumerate(sampled):
                            frac = i / max(len(sampled) - 1, 1)
                            d = frac * length
                            e = start_ele + frac * (end_ele - start_ele)
                            profile.append({"d": round(d, 1), "e": round(e, 1), "s": round(slope, 1)})
                        profile[-1]["s"] = 0.0

                        gain = max(0, end_ele - start_ele)
                        loss = max(0, start_ele - end_ele)
                        tagged_roads.append(SteepRoad(
                            name=road_name,
                            highway_type=highway,
                            length_m=round(length, 1),
                            elevation_gain_m=round(gain, 1),
                            elevation_loss_m=round(loss, 1),
                            avg_slope_percent=round(slope, 1),
                            max_slope_percent=round(slope, 1),
                            start_elevation_m=round(start_ele, 1),
                            end_elevation_m=round(end_ele, 1),
                            geometry=coords,
                            osm_id=element.get("id"),
                            profile=profile,
                            incline_tag=None,
                            surface=tags.get("surface"),
                            access=tags.get("access"),
                        ))
                        continue

            # Priority 3: fall back to DEM
            sampled = interpolate_line(coords, settings.sample_step_m)
            dem_roads_data.append({
                "osm_id": element.get("id"),
                "name": name,
                "highway": highway,
                "incline": incline_tag,
                "surface": tags.get("surface"),
                "access": tags.get("access"),
                "coords": sampled,
                "coord_start_idx": len(all_dem_coords),
            })
            all_dem_coords.extend(sampled)

        # DEM fallback batch
        if all_dem_coords:
            dem = LocalDEM()
            all_elevations = dem.get_elevations_batch(all_dem_coords)

            for road in dem_roads_data:
                start_idx = road["coord_start_idx"]
                n_points = len(road["coords"])
                elevations = all_elevations[start_idx : start_idx + n_points]

                stats = calculate_road_slope(road["coords"], elevations)
                if stats is None:
                    continue

                if stats["avg_slope"] < min_avg_slope and stats["max_slope"] < 8.0:
                    continue

                total_elev_change = max(stats["elevation_gain"], stats["elevation_loss"])
                if total_elev_change < settings.min_elevation_change_m:
                    continue

                tagged_roads.append(SteepRoad(
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
                    incline_tag=road["incline"],
                    surface=road["surface"],
                    access=road["access"],
                ))

        tagged_roads.sort(key=lambda x: x.avg_slope_percent, reverse=True)
        return tagged_roads
