#!/usr/bin/env python3
"""
Find Steep Roads for Running - v3

Improvements:
1. Uses Open-Topo-Data API with EU-DEM (25m resolution) or IGN data for France
2. Focuses on running-suitable roads (footways, paths, residential streets)
3. No aggressive merging - keeps individual segments
4. Better smoothing for realistic slope values
5. Filters out highways/big roads

Usage:
    python3 v3.py --lat 45.7578 --lon 4.832 --radius 2000
    python3 v3.py --city "Lyon, France" --radius 3000
"""

import os
import sys
import json
import math
import argparse
import requests
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import numpy as np
from shapely.geometry import LineString, Point
from pyproj import Transformer
from scipy.ndimage import gaussian_filter1d
import gzip
import shutil

# =========================
# SRTM DEM HANDLING (LOCAL - FAST)
# =========================

class LocalDEM:
    """Fast local DEM using downloaded SRTM tiles."""
    
    def __init__(self, dem_dir: str = "dem_tiles"):
        self.dem_dir = dem_dir
        self.tiles = {}
        os.makedirs(dem_dir, exist_ok=True)
    
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
            print(f"  Downloading {tile_name}...")
            resp = requests.get(url, stream=True, timeout=60)
            if resp.status_code != 200:
                return False
            
            gz_path = os.path.join(self.dem_dir, f"{tile_name}.hgt.gz")
            with open(gz_path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
            
            with gzip.open(gz_path, 'rb') as f_in, open(hgt_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            
            os.remove(gz_path)
            return True
        except Exception as e:
            print(f"  Failed to download {tile_name}: {e}")
            return False
    
    def _load_tile(self, tile_name: str) -> bool:
        if tile_name in self.tiles:
            return True
        
        hgt_path = os.path.join(self.dem_dir, f"{tile_name}.hgt")
        if not os.path.exists(hgt_path):
            if not self._download_tile(tile_name):
                return False
        
        try:
            with open(hgt_path, 'rb') as f:
                data = f.read()
            
            size = int(math.sqrt(len(data) / 2))
            arr = np.frombuffer(data, dtype='>i2').reshape((size, size)).astype(np.float32)
            arr[arr == -32768] = np.nan  # Void values
            
            self.tiles[tile_name] = arr
            return True
        except Exception as e:
            print(f"  Failed to load {tile_name}: {e}")
            return False
    
    def get_elevation(self, lat: float, lon: float) -> float:
        tile_name = self._tile_name(lat, lon)
        
        if not self._load_tile(tile_name):
            return np.nan
        
        arr = self.tiles[tile_name]
        size = arr.shape[0]
        
        lat_int = int(math.floor(lat))
        lon_int = int(math.floor(lon))
        
        # Pixel coordinates with bilinear interpolation
        col = (lon - lon_int) * (size - 1)
        row = (lat_int + 1 - lat) * (size - 1)
        
        c0, r0 = int(col), int(row)
        c1, r1 = min(c0 + 1, size - 1), min(r0 + 1, size - 1)
        
        if r0 < 0 or r0 >= size or c0 < 0 or c0 >= size:
            return np.nan
        
        dc, dr = col - c0, row - r0
        
        v00, v01 = arr[r0, c0], arr[r0, c1]
        v10, v11 = arr[r1, c0], arr[r1, c1]
        
        if np.isnan(v00) or np.isnan(v01) or np.isnan(v10) or np.isnan(v11):
            return float(arr[int(round(row)), int(round(col))])
        
        return float(v00*(1-dc)*(1-dr) + v01*dc*(1-dr) + v10*(1-dc)*dr + v11*dc*dr)
    
    def get_elevations_batch(self, coords: List[Tuple[float, float]]) -> np.ndarray:
        """Get elevations for multiple points - coords are (lon, lat)."""
        return np.array([self.get_elevation(lat, lon) for lon, lat in coords])

# =========================
# CONFIGURATION
# =========================

DEM_DIR = "dem_tiles"

# Slope calculation
SAMPLE_STEP_M = 15              # Distance between elevation samples (increased for speed)
MIN_SEGMENT_LENGTH_M = 50       # Minimum road length to consider
MAX_REALISTIC_SLOPE = 18        # Maximum realistic slope % - lowered more for running roads
SMOOTHING_SIGMA = 3             # Gaussian smoothing strength (increased)
MIN_ELEVATION_CHANGE_M = 6      # Minimum total elevation change to be considered "steep"
MIN_SLOPE_ELEVATION_RATIO = 0.7 # Slope must be consistent with actual elevation delta
MIN_LENGTH_FOR_STEEP = 80       # Short roads need higher confidence (DEM noise is worse)

# Running-suitable highway types (ordered by preference)
RUNNING_HIGHWAY_TYPES = [
    "footway",      # Designated footpaths
    "path",         # Generic paths
    "steps",        # Stairs (very steep!)
    "pedestrian",   # Pedestrian zones
    "residential",  # Residential streets (low traffic)
    "living_street", # Shared space streets
    "track",        # Agricultural/forestry tracks
    "service",      # Service roads (parking lots, etc.) - maybe useful
    "unclassified", # Minor roads
]

# Exclude these highway types (too dangerous for running)
EXCLUDE_HIGHWAY_TYPES = [
    "primary", "primary_link",
    "secondary", "secondary_link", 
    "tertiary", "tertiary_link",
    "trunk", "trunk_link",
    "motorway", "motorway_link",
]

# Exclude roads with these keywords in the name (tunnels, highways)
EXCLUDE_NAME_KEYWORDS = [
    "tunnel", "autoroute"
]

# =========================
# COORDINATE UTILITIES
# =========================

transformer_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance in meters."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


def line_length_meters(coords: List[Tuple[float, float]]) -> float:
    """Calculate total length of a line in meters."""
    total = 0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        total += haversine_distance(lat1, lon1, lat2, lon2)
    return total


def interpolate_line(coords: List[Tuple[float, float]], step_m: float) -> List[Tuple[float, float]]:
    """Interpolate points along a line at regular intervals."""
    if len(coords) < 2:
        return coords
    
    # Build cumulative distances
    distances = [0]
    for i in range(1, len(coords)):
        d = haversine_distance(coords[i-1][1], coords[i-1][0], coords[i][1], coords[i][0])
        distances.append(distances[-1] + d)
    
    total_length = distances[-1]
    if total_length < step_m:
        return coords
    
    # Interpolate at regular intervals
    result = []
    target_distances = np.arange(0, total_length, step_m)
    if target_distances[-1] < total_length:
        target_distances = np.append(target_distances, total_length)
    
    for target_d in target_distances:
        # Find segment containing this distance
        for i in range(len(distances) - 1):
            if distances[i] <= target_d <= distances[i + 1]:
                seg_length = distances[i + 1] - distances[i]
                if seg_length > 0:
                    t = (target_d - distances[i]) / seg_length
                else:
                    t = 0
                lon = coords[i][0] + t * (coords[i + 1][0] - coords[i][0])
                lat = coords[i][1] + t * (coords[i + 1][1] - coords[i][1])
                result.append((lon, lat))
                break
    
    return result

# =========================
# GEOCODING
# =========================

def geocode_location(query: str) -> Tuple[float, float]:
    """Get coordinates for a location name."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": "SteepRoadFinder/2.0"}
    
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    
    data = resp.json()
    if not data:
        raise ValueError(f"Location '{query}' not found")
    
    return float(data[0]["lat"]), float(data[0]["lon"])

# =========================
# ELEVATION DATA - OPEN-TOPO-DATA API
# =========================

def get_elevations_opentopo(coords: List[Tuple[float, float]], 
                            dataset: str = "eudem25m",
                            batch_size: int = 100) -> np.ndarray:
    """
    Get elevations from Open-Topo-Data API.
    
    Datasets:
    - eudem25m: EU-DEM at 25m resolution (Europe)
    - mapzen: Mapzen global DEM (worldwide, ~30m)
    - srtm30m: SRTM 30m (worldwide)
    - aster30m: ASTER 30m (worldwide)
    
    For France, eudem25m is much better than SRTM.
    """
    elevations = np.full(len(coords), np.nan)
    
    # API endpoint - using public instance
    base_url = f"https://api.opentopodata.org/v1/{dataset}"
    
    for i in range(0, len(coords), batch_size):
        batch = coords[i:i+batch_size]
        
        # Format: lat,lon|lat,lon|...
        locations = "|".join([f"{lat},{lon}" for lon, lat in batch])
        
        try:
            resp = requests.get(
                base_url,
                params={"locations": locations},
                timeout=30
            )
            
            if resp.status_code == 429:
                # Rate limited - wait and retry
                print("  Rate limited, waiting 2s...")
                time.sleep(2)
                resp = requests.get(
                    base_url,
                    params={"locations": locations},
                    timeout=30
                )
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "OK":
                    for j, result in enumerate(data.get("results", [])):
                        elev = result.get("elevation")
                        if elev is not None:
                            elevations[i + j] = float(elev)
            else:
                print(f"  Warning: API returned {resp.status_code}")
                
        except Exception as e:
            print(f"  Elevation API error: {e}")
        
        # Rate limiting - API allows 1 request/second for public use
        if i + batch_size < len(coords):
            time.sleep(1.1)
    
    return elevations


def get_elevations_batch_fallback(coords: List[Tuple[float, float]]) -> np.ndarray:
    """Try multiple elevation sources with fallback."""
    
    # Try EU-DEM first (best for Europe)
    print("  Fetching elevations from EU-DEM...")
    elevations = get_elevations_opentopo(coords, dataset="eudem25m")
    
    # Check how many we got
    valid_count = np.sum(~np.isnan(elevations))
    if valid_count < len(coords) * 0.8:
        print(f"  Only got {valid_count}/{len(coords)} from EU-DEM, trying Mapzen...")
        elevations_mapzen = get_elevations_opentopo(coords, dataset="mapzen")
        
        # Fill in missing values
        nan_mask = np.isnan(elevations)
        elevations[nan_mask] = elevations_mapzen[nan_mask]
    
    return elevations

# =========================
# OSM DATA FETCHING
# =========================

def fetch_running_roads(lat: float, lon: float, radius_m: float,
                        include_unnamed: bool = True) -> dict:
    """Fetch running-suitable roads from OSM."""
    
    highway_filter = "|".join(RUNNING_HIGHWAY_TYPES)
    
    # Query includes name tag to help filter
    query = f"""
    [out:json][timeout:180];
    (
      way["highway"~"^({highway_filter})$"](around:{radius_m},{lat},{lon});
    );
    out body geom;
    """
    
    print(f"Fetching running roads within {radius_m}m of ({lat:.4f}, {lon:.4f})...")
    
    for attempt in range(3):
        try:
            resp = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                timeout=240
            )
            
            if resp.status_code == 429 or resp.status_code == 504:
                wait = (attempt + 1) * 15
                print(f"  Overpass API busy, waiting {wait}s...")
                time.sleep(wait)
                continue
                
            resp.raise_for_status()
            data = resp.json()
            print(f"  Found {len(data.get('elements', []))} road segments")
            return data
            
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                print(f"  Retry {attempt + 1}/3: {e}")
                time.sleep(10)
            else:
                raise
    
    return {"elements": []}

# =========================
# SLOPE CALCULATION
# =========================

def calculate_road_slope(coords: List[Tuple[float, float]], 
                         elevations: np.ndarray) -> Optional[dict]:
    """Calculate slope statistics for a road segment."""
    
    if len(coords) < 3 or len(elevations) < 3:
        return None
    
    # Build distance array
    distances = [0]
    for i in range(1, len(coords)):
        d = haversine_distance(coords[i-1][1], coords[i-1][0], 
                               coords[i][1], coords[i][0])
        distances.append(distances[-1] + d)
    distances = np.array(distances)
    total_length = distances[-1]
    
    if total_length < MIN_SEGMENT_LENGTH_M:
        return None
    
    # Interpolate NaN values
    valid = ~np.isnan(elevations)
    if valid.sum() < 3:
        return None
    
    elevs = np.copy(elevations)
    if not valid.all():
        elevs = np.interp(distances, distances[valid], elevations[valid])
    
    # Apply Gaussian smoothing to reduce DEM noise
    # Use stronger smoothing for short roads
    sigma = max(SMOOTHING_SIGMA, len(elevs) // 10)
    if len(elevs) > sigma * 2:
        elevs = gaussian_filter1d(elevs, sigma=sigma, mode='nearest')
    
    # Calculate point-to-point slopes
    dz = np.diff(elevs)
    dd = np.diff(distances)
    dd = np.where(dd < 0.1, 0.1, dd)  # Avoid division by zero
    
    slopes = (dz / dd) * 100
    
    # Clip unrealistic values
    slopes = np.clip(slopes, -MAX_REALISTIC_SLOPE, MAX_REALISTIC_SLOPE)
    
    # Calculate statistics
    abs_slopes = np.abs(slopes)
    
    # For average, use a weighted approach favoring the middle of the road
    # (edges often have more noise)
    weights = np.ones(len(slopes))
    if len(slopes) > 4:
        weights[:2] = 0.5
        weights[-2:] = 0.5
    
    avg_slope = np.average(abs_slopes, weights=weights)
    max_slope = np.max(abs_slopes)
    
    # Elevation changes - use absolute values so direction doesn't matter
    total_elevation_change = abs(elevs[-1] - elevs[0])
    elevation_gain = max(0, elevs[-1] - elevs[0])  # If going up
    elevation_loss = max(0, elevs[0] - elevs[-1])  # If going down
    
    # Sanity check: computed slope should be consistent with actual elevation change
    # If elevation change is tiny, the slope is probably DEM noise
    computed_slope_from_delta = (total_elevation_change / total_length) * 100
    
    # Use the more conservative slope estimate to avoid DEM noise artifacts
    # Take the minimum of: point-to-point average, and total delta slope
    avg_slope = min(avg_slope, computed_slope_from_delta * 1.3)  # Allow some tolerance
    
    return {
        "length_m": total_length,
        "avg_slope": float(avg_slope),
        "max_slope": float(max_slope),
        "elevation_gain": float(elevation_gain),
        "elevation_loss": float(elevation_loss),
        "start_elevation": float(elevs[0]),
        "end_elevation": float(elevs[-1]),
    }

# =========================
# DATA STRUCTURES
# =========================

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
    geometry: List[Tuple[float, float]]
    osm_id: int
    incline_tag: Optional[str] = None
    surface: Optional[str] = None
    access: Optional[str] = None

# =========================
# MAIN PROCESSING
# =========================

def find_steep_roads(
    lat: float,
    lon: float,
    radius_m: float = 3000,
    min_avg_slope: float = 5.0,
    min_max_slope: float = 8.0,
    include_unnamed: bool = True,
    named_only: bool = False,
    min_length: float = MIN_SEGMENT_LENGTH_M,
) -> List[SteepRoad]:
    """Find steep running roads around a point."""
    
    # Fetch OSM data
    osm_data = fetch_running_roads(lat, lon, radius_m, include_unnamed)
    
    elements = osm_data.get("elements", [])
    if not elements:
        print("No roads found!")
        return []
    
    # Collect all coordinates for batch elevation query
    all_coords = []
    road_data = []
    
    print("Preparing road segments...")
    for element in elements:
        if element.get("type") != "way" or "geometry" not in element:
            continue
        
        tags = element.get("tags", {})
        highway = tags.get("highway", "")
        name = tags.get("name", "")
        
        # Skip excluded types
        if highway in EXCLUDE_HIGHWAY_TYPES:
            continue
        
        # Skip tunnels (dark, not great for running)
        if tags.get("tunnel"):
            continue
        name_lower = name.lower()
        if any(kw in name_lower for kw in EXCLUDE_NAME_KEYWORDS):
            continue
        
        # Get coordinates
        coords = [(p["lon"], p["lat"]) for p in element["geometry"]]
        if len(coords) < 2:
            continue
        
        length = line_length_meters(coords)
        if length < min_length:
            continue
        
        # Interpolate for elevation sampling
        sampled = interpolate_line(coords, SAMPLE_STEP_M)
        
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
    
    print(f"  Prepared {len(road_data)} road segments with {len(all_coords)} points")
    
    if not all_coords:
        return []
    
    # Get elevations from local DEM (fast!)
    print("Loading elevation data from local DEM...")
    dem = LocalDEM(DEM_DIR)
    all_elevations = dem.get_elevations_batch(all_coords)
    
    valid_count = np.sum(~np.isnan(all_elevations))
    print(f"  Got {valid_count}/{len(all_coords)} valid elevations")
    
    # Process each road
    print("Calculating slopes...")
    results = []
    
    for road in road_data:
        start_idx = road["coord_start_idx"]
        n_points = len(road["coords"])
        elevations = all_elevations[start_idx:start_idx + n_points]
        
        stats = calculate_road_slope(road["coords"], elevations)
        if stats is None:
            continue
        
        # Check OSM incline tag - use it as primary source if available
        osm_incline = road.get("incline")
        has_osm_tag = False
        if osm_incline:
            try:
                incline_str = str(osm_incline).lower().strip()
                if incline_str in ["steep", "yes"]:
                    incline_val = 12.0  # Conservative estimate
                    has_osm_tag = True
                elif incline_str in ["up", "down"]:
                    pass  # Don't override, just indicates direction
                else:
                    incline_val = float(incline_str.replace("%", "").replace("°", "").replace(",", "."))
                    if "°" in str(osm_incline):
                        incline_val = math.tan(math.radians(abs(incline_val))) * 100
                    incline_val = abs(incline_val)
                    has_osm_tag = True
                    
                    # Use OSM value as the primary source - it's more reliable
                    if has_osm_tag and incline_val > 0:
                        # Blend OSM tag with computed (OSM is more trusted)
                        stats["avg_slope"] = incline_val
                        stats["max_slope"] = max(stats["max_slope"], incline_val)
            except:
                pass
        
        # Apply filters
        # 1. Minimum slope threshold
        if stats["avg_slope"] < min_avg_slope and stats["max_slope"] < min_max_slope:
            continue
        
        # 2. Minimum elevation change (filters out DEM noise on flat roads)
        total_elev_change = max(stats["elevation_gain"], stats["elevation_loss"])
        if total_elev_change < MIN_ELEVATION_CHANGE_M and not has_osm_tag:
            continue
        
        # 3. Short roads without OSM tags need higher elevation change (DEM is noisier)
        if not has_osm_tag and stats["length_m"] < MIN_LENGTH_FOR_STEEP:
            min_change_required = stats["length_m"] * 0.15  # Need ~15% slope worth of change
            if total_elev_change < min_change_required:
                continue
        
        # 4. Consistency check: computed slope should match elevation delta
        # This filters out DEM noise where point-to-point has spikes but total delta is small
        expected_slope_from_delta = (total_elev_change / stats["length_m"]) * 100
        if not has_osm_tag and stats["avg_slope"] > 8:
            # If claimed slope is much higher than what delta suggests, it's likely noise
            if expected_slope_from_delta < stats["avg_slope"] * MIN_SLOPE_ELEVATION_RATIO:
                continue
        
        # 4. Named roads only filter
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
            incline_tag=osm_incline,
            surface=road["surface"],
            access=road["access"],
        ))
    
    # Sort by average slope
    results.sort(key=lambda x: x.avg_slope_percent, reverse=True)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Find steep roads for running",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 find_steep_roads_v2.py --lat 45.7578 --lon 4.832 --radius 2000
    python3 find_steep_roads_v2.py --city "Lyon, France" --radius 3000 --min-slope 6
        """
    )
    
    # Location
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--city", type=str, help="City name to geocode")
    
    # Search parameters
    parser.add_argument("--radius", type=float, default=3000,
                        help="Search radius in meters (default: 3000)")
    parser.add_argument("--min-slope", type=float, default=5.0,
                        help="Minimum average slope %% (default: 5)")
    parser.add_argument("--min-length", type=float, default=50,
                        help="Minimum segment length in meters (default: 50)")
    parser.add_argument("--named-only", action="store_true",
                        help="Only include named roads (reduces noise)")
    
    # Output
    parser.add_argument("-o", "--output", type=str, help="Output JSON file")
    parser.add_argument("--top", type=int, help="Only output top N results")
    
    args = parser.parse_args()
    
    # Get coordinates
    if args.city:
        lat, lon = geocode_location(args.city)
        print(f"Geocoded '{args.city}' to ({lat:.4f}, {lon:.4f})")
        location_name = args.city.replace(" ", "_").replace(",", "")
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        location_name = f"{lat:.2f}_{lon:.2f}"
    else:
        parser.error("Either --city or both --lat and --lon required")
        return
    
    # Find steep roads
    roads = find_steep_roads(
        lat=lat,
        lon=lon,
        radius_m=args.radius,
        min_avg_slope=args.min_slope,
        min_length=args.min_length,
        named_only=args.named_only,
    )
    
    # Limit results
    if args.top:
        roads = roads[:args.top]
    
    # Save output
    output_file = args.output or f"steep_running_roads_{location_name}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in roads], f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Found {len(roads)} steep running roads")
    print(f"📁 Saved to {output_file}")
    
    # Summary
    if roads:
        print(f"\nTop 15 steepest roads for running:")
        print("-" * 90)
        print(f"{'#':>2} {'Name':<35} {'Type':<12} {'Avg':>6} {'Max':>6} {'Len':>6} {'Gain':>5}")
        print("-" * 90)
        for i, road in enumerate(roads[:15], 1):
            name = road.name[:33] + ".." if len(road.name) > 35 else road.name
            tag_marker = "⭐" if road.incline_tag else ""
            print(f"{i:2}. {name:<35} {road.highway_type:<12} {road.avg_slope_percent:5.1f}% "
                  f"{road.max_slope_percent:5.1f}% {road.length_m:5.0f}m {road.elevation_gain_m:4.0f}m {tag_marker}")


if __name__ == "__main__":
    main()
