#!/usr/bin/env python3
"""
Find Steep Roads - Improved approach for finding steep roads around a location.

Key improvements over extract.py:
1. Uses point + radius instead of city bounding box (more precise)
2. Better elevation data handling with multiple sources and robust smoothing
3. Filters out unrealistic slopes caused by DEM noise
4. Prioritizes OSM incline tags when available (most accurate)
5. Segments roads to find the steepest portions
6. Better highway type filtering for realistic road candidates
7. Calculates gradient over longer segments to reduce noise

Usage:
    python v2.py --lat 45.75 --lon 4.83 --radius 3000
    python v2.py --city "Lyon, France" --radius 5000
"""

import os
import sys
import json
import math
import argparse
import requests
import gzip
import shutil
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from pyproj import Transformer
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import rasterio

# =========================
# CONFIGURATION
# =========================

DEM_DIR = "dem_tiles"
OUTPUT_DIR = "."

# Slope calculation parameters
SAMPLE_STEP_M = 10              # Distance between elevation samples (meters)
GRADIENT_WINDOW_M = 50          # Window for gradient calculation (reduces noise)
MIN_SEGMENT_LENGTH_M = 30       # Minimum road segment length to consider
MAX_REALISTIC_SLOPE = 35        # Maximum realistic road slope (%) - filter outliers

# Filtering thresholds
MIN_AVG_SLOPE = 4.0             # Minimum average slope to include
MIN_MAX_SLOPE = 6.0             # Or minimum max slope to include

# Highway types to consider (ordered by preference for steep road detection)
HIGHWAY_TYPES_CYCLING = [
    "residential", "tertiary", "secondary", "primary", "unclassified",
    "cycleway", "path", "track"
]
HIGHWAY_TYPES_ALL = HIGHWAY_TYPES_CYCLING + [
    "footway", "steps", "pedestrian", "service"
]

# =========================
# DATA CLASSES
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
    osm_id: Optional[int] = None
    incline_tag: Optional[str] = None
    surface: Optional[str] = None
    confidence: str = "computed"  # "osm_tag", "computed_verified", "computed"

# =========================
# COORDINATE UTILITIES
# =========================

transformer_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
transformer_from_mercator = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in meters."""
    R = 6371000  # Earth radius in meters
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
    line = LineString(coords)
    # Convert to meters for accurate interpolation
    coords_m = [transformer_to_mercator.transform(lon, lat) for lon, lat in coords]
    line_m = LineString(coords_m)
    length_m = line_m.length
    
    if length_m < step_m:
        return coords
    
    distances = np.arange(0, length_m, step_m)
    if distances[-1] < length_m:
        distances = np.append(distances, length_m)
    
    points = []
    for d in distances:
        pt = line_m.interpolate(d)
        lon, lat = transformer_from_mercator.transform(pt.x, pt.y)
        points.append((lon, lat))
    
    return points

# =========================
# GEOCODING
# =========================

def geocode_location(query: str) -> Tuple[float, float]:
    """Get coordinates for a location name using Nominatim."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": "SteepRoadFinder/1.0"}
    
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    
    data = resp.json()
    if not data:
        raise ValueError(f"Location '{query}' not found")
    
    return float(data[0]["lat"]), float(data[0]["lon"])

# =========================
# SRTM DEM HANDLING
# =========================

def get_srtm_tile_name(lat: float, lon: float) -> str:
    """Get SRTM tile name for a coordinate."""
    lat_int = int(math.floor(lat))
    lon_int = int(math.floor(lon))
    lat_prefix = "N" if lat_int >= 0 else "S"
    lon_prefix = "E" if lon_int >= 0 else "W"
    return f"{lat_prefix}{abs(lat_int):02d}{lon_prefix}{abs(lon_int):03d}"


def get_required_tiles(lat: float, lon: float, radius_m: float) -> List[str]:
    """Get all SRTM tiles needed to cover a circular area."""
    # Rough conversion: 1 degree ≈ 111km
    radius_deg = radius_m / 111000 * 1.5  # Add margin
    
    tiles = set()
    for dlat in np.arange(-radius_deg, radius_deg + 0.5, 0.5):
        for dlon in np.arange(-radius_deg, radius_deg + 0.5, 0.5):
            tiles.add(get_srtm_tile_name(lat + dlat, lon + dlon))
    
    return list(tiles)


def download_srtm_tile(tile_name: str, dem_dir: str = DEM_DIR) -> Optional[str]:
    """Download and extract SRTM tile if not present."""
    os.makedirs(dem_dir, exist_ok=True)
    
    hgt_path = os.path.join(dem_dir, f"{tile_name}.hgt")
    if os.path.exists(hgt_path):
        return hgt_path
    
    # Try AWS elevation tiles
    lat_prefix = tile_name[:1]
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{tile_name[:3]}/{tile_name}.hgt.gz"
    
    try:
        print(f"Downloading {tile_name}...")
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code != 200:
            print(f"  Warning: Could not download {tile_name}")
            return None
        
        gz_path = os.path.join(dem_dir, f"{tile_name}.hgt.gz")
        with open(gz_path, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
        
        with gzip.open(gz_path, 'rb') as f_in, open(hgt_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        
        os.remove(gz_path)
        print(f"  Downloaded {tile_name}")
        return hgt_path
    except Exception as e:
        print(f"  Error downloading {tile_name}: {e}")
        return None


class DEMManager:
    """Manages DEM tiles and elevation queries with caching."""
    
    def __init__(self, dem_dir: str = DEM_DIR):
        self.dem_dir = dem_dir
        self.tiles: Dict[str, np.ndarray] = {}
        self.tile_bounds: Dict[str, Tuple[float, float, float, float]] = {}
    
    def load_tile(self, tile_name: str) -> bool:
        """Load a DEM tile into memory."""
        if tile_name in self.tiles:
            return True
        
        hgt_path = os.path.join(self.dem_dir, f"{tile_name}.hgt")
        if not os.path.exists(hgt_path):
            hgt_path = download_srtm_tile(tile_name, self.dem_dir)
            if not hgt_path:
                return False
        
        try:
            with open(hgt_path, 'rb') as f:
                data = f.read()
            
            # SRTM1 tiles are 3601x3601, SRTM3 are 1201x1201
            size = int(math.sqrt(len(data) / 2))
            arr = np.frombuffer(data, dtype='>i2').reshape((size, size))
            
            # Parse tile bounds from name
            lat_sign = 1 if tile_name[0] == 'N' else -1
            lon_sign = 1 if tile_name[3] == 'E' else -1
            lat_origin = int(tile_name[1:3]) * lat_sign
            lon_origin = int(tile_name[4:7]) * lon_sign
            
            self.tiles[tile_name] = arr.astype(np.float32)
            self.tile_bounds[tile_name] = (lat_origin, lon_origin, lat_origin + 1, lon_origin + 1)
            
            # Replace void values with NaN
            self.tiles[tile_name][self.tiles[tile_name] == -32768] = np.nan
            
            return True
        except Exception as e:
            print(f"Error loading tile {tile_name}: {e}")
            return False
    
    def get_elevation(self, lat: float, lon: float, interpolate: bool = True) -> float:
        """Get elevation at a point with optional bilinear interpolation."""
        tile_name = get_srtm_tile_name(lat, lon)
        
        if tile_name not in self.tiles:
            if not self.load_tile(tile_name):
                return np.nan
        
        arr = self.tiles[tile_name]
        bounds = self.tile_bounds[tile_name]
        lat_origin, lon_origin = bounds[0], bounds[1]
        
        size = arr.shape[0]
        pixel_size = 1.0 / (size - 1)
        
        # Calculate pixel coordinates
        col = (lon - lon_origin) / pixel_size
        row = (lat_origin + 1 - lat) / pixel_size
        
        if interpolate:
            # Bilinear interpolation
            c0, r0 = int(col), int(row)
            c1, r1 = min(c0 + 1, size - 1), min(r0 + 1, size - 1)
            
            if r0 < 0 or r0 >= size or c0 < 0 or c0 >= size:
                return np.nan
            
            dc = col - c0
            dr = row - r0
            
            v00 = arr[r0, c0]
            v01 = arr[r0, c1]
            v10 = arr[r1, c0]
            v11 = arr[r1, c1]
            
            # Skip if any value is void
            if np.isnan(v00) or np.isnan(v01) or np.isnan(v10) or np.isnan(v11):
                return float(arr[int(round(row)), int(round(col))])
            
            value = (v00 * (1-dc) * (1-dr) + 
                    v01 * dc * (1-dr) + 
                    v10 * (1-dc) * dr + 
                    v11 * dc * dr)
            return float(value)
        else:
            r, c = int(round(row)), int(round(col))
            if 0 <= r < size and 0 <= c < size:
                return float(arr[r, c])
            return np.nan
    
    def get_elevations_batch(self, coords: List[Tuple[float, float]]) -> np.ndarray:
        """Get elevations for multiple points efficiently."""
        return np.array([self.get_elevation(lat, lon) for lon, lat in coords])

# =========================
# OPEN ELEVATION API (FALLBACK)
# =========================

def get_elevations_open_elevation(coords: List[Tuple[float, float]], 
                                   batch_size: int = 100) -> np.ndarray:
    """Get elevations from Open Elevation API."""
    elevations = np.full(len(coords), np.nan)
    
    for i in range(0, len(coords), batch_size):
        batch = coords[i:i+batch_size]
        locations = [{"latitude": lat, "longitude": lon} for lon, lat in batch]
        
        try:
            resp = requests.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": locations},
                timeout=30
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                for j, result in enumerate(results):
                    elev = result.get("elevation")
                    if elev is not None:
                        elevations[i + j] = float(elev)
        except Exception as e:
            print(f"Open Elevation API error: {e}")
    
    return elevations

# =========================
# OSM DATA FETCHING
# =========================

def fetch_roads_around_point(lat: float, lon: float, radius_m: float,
                             highway_types: List[str] = None,
                             max_retries: int = 3) -> dict:
    """Fetch road data from OSM Overpass API with retry logic."""
    if highway_types is None:
        highway_types = HIGHWAY_TYPES_ALL
    
    highway_filter = "|".join(highway_types)
    
    query = f"""
    [out:json][timeout:120];
    (
      way["highway"~"^({highway_filter})$"](around:{radius_m},{lat},{lon});
    );
    out body geom;
    """
    
    print(f"Fetching roads within {radius_m}m of ({lat:.4f}, {lon:.4f})...")
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                timeout=180
            )
            resp.raise_for_status()
            
            data = resp.json()
            print(f"  Found {len(data.get('elements', []))} road segments")
            
            return data
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                import time
                time.sleep(wait_time)
            else:
                raise
    
    return {"elements": []}

# =========================
# SLOPE CALCULATION
# =========================

def calculate_slopes(coords: List[Tuple[float, float]], 
                     elevations: np.ndarray,
                     gradient_window_m: float = GRADIENT_WINDOW_M) -> dict:
    """
    Calculate slope statistics for a road segment.
    
    Uses gradient calculation over a window to reduce noise from DEM errors.
    """
    if len(coords) < 2 or len(elevations) < 2:
        return None
    
    # Calculate cumulative distance
    distances = [0]
    for i in range(1, len(coords)):
        lon1, lat1 = coords[i-1]
        lon2, lat2 = coords[i]
        d = haversine_distance(lat1, lon1, lat2, lon2)
        distances.append(distances[-1] + d)
    distances = np.array(distances)
    total_length = distances[-1]
    
    if total_length < MIN_SEGMENT_LENGTH_M:
        return None
    
    # Interpolate NaN values
    valid = ~np.isnan(elevations)
    if valid.sum() < 2:
        return None
    
    elevs = np.copy(elevations)
    if not valid.all():
        elevs = np.interp(distances, distances[valid], elevations[valid])
    
    # Apply Savitzky-Golay filter for smoothing (preserves peaks better than Gaussian)
    window = max(5, min(len(elevs) // 3, 15))
    if window % 2 == 0:
        window += 1
    if len(elevs) > window:
        try:
            elevs = savgol_filter(elevs, window, polyorder=2)
        except:
            # Fallback to Gaussian
            elevs = gaussian_filter1d(elevs, sigma=2)
    
    # Calculate slopes using gradient over a window
    # This reduces noise from DEM inaccuracies
    n_points = len(elevs)
    slopes = []
    
    # Calculate gradient using central differences over a window
    for i in range(n_points):
        # Find points within the gradient window
        window_start = max(0, np.searchsorted(distances, distances[i] - gradient_window_m/2))
        window_end = min(n_points, np.searchsorted(distances, distances[i] + gradient_window_m/2))
        
        if window_end > window_start:
            d_dist = distances[window_end-1] - distances[window_start]
            d_elev = elevs[window_end-1] - elevs[window_start]
            if d_dist > 0:
                slope = (d_elev / d_dist) * 100
                # Clip unrealistic slopes
                slope = np.clip(slope, -MAX_REALISTIC_SLOPE, MAX_REALISTIC_SLOPE)
                slopes.append(slope)
    
    if not slopes:
        return None
    
    slopes = np.array(slopes)
    
    # Calculate statistics
    abs_slopes = np.abs(slopes)
    elevation_changes = np.diff(elevs)
    
    return {
        "length_m": total_length,
        "avg_slope": float(np.mean(abs_slopes)),
        "max_slope": float(np.max(abs_slopes)),
        "elevation_gain": float(np.sum(elevation_changes[elevation_changes > 0])),
        "elevation_loss": float(np.abs(np.sum(elevation_changes[elevation_changes < 0]))),
        "start_elevation": float(elevs[0]),
        "end_elevation": float(elevs[-1]),
        "smoothed_elevations": elevs.tolist()
    }

# =========================
# MAIN PROCESSING
# =========================

def find_steep_roads(
    lat: float, 
    lon: float, 
    radius_m: float = 5000,
    min_avg_slope: float = MIN_AVG_SLOPE,
    min_max_slope: float = MIN_MAX_SLOPE,
    highway_types: List[str] = None,
    include_unnamed: bool = False,
    use_open_elevation: bool = False
) -> List[SteepRoad]:
    """
    Find steep roads around a point.
    
    Args:
        lat, lon: Center point coordinates
        radius_m: Search radius in meters
        min_avg_slope: Minimum average slope percentage to include
        min_max_slope: Minimum max slope percentage to include
        highway_types: List of OSM highway types to include
        include_unnamed: Whether to include roads without names
        use_open_elevation: Use Open Elevation API instead of DEM
    
    Returns:
        List of SteepRoad objects sorted by average slope
    """
    
    if highway_types is None:
        highway_types = HIGHWAY_TYPES_ALL
    
    # Initialize elevation data
    if not use_open_elevation:
        dem = DEMManager()
        required_tiles = get_required_tiles(lat, lon, radius_m)
        print(f"Loading {len(required_tiles)} DEM tiles...")
        for tile in required_tiles:
            dem.load_tile(tile)
    
    # Fetch OSM data
    osm_data = fetch_roads_around_point(lat, lon, radius_m, highway_types)
    
    # Group roads by name for merging
    roads_by_name = defaultdict(list)
    road_info = {}
    
    for element in osm_data.get("elements", []):
        if element.get("type") != "way" or "geometry" not in element:
            continue
        
        tags = element.get("tags", {})
        name = tags.get("name", "").strip()
        
        if not name and not include_unnamed:
            continue
        
        coords = [(p["lon"], p["lat"]) for p in element["geometry"]]
        if len(coords) < 2:
            continue
        
        key = name if name else f"unnamed_{element['id']}"
        roads_by_name[key].append(LineString(coords))
        
        # Store road metadata
        if key not in road_info:
            road_info[key] = {
                "highway": tags.get("highway", ""),
                "incline": tags.get("incline"),
                "surface": tags.get("surface"),
                "osm_id": element.get("id")
            }
    
    print(f"Processing {len(roads_by_name)} unique roads...")
    
    results = []
    
    for name, lines in roads_by_name.items():
        # Merge line segments
        if len(lines) > 1:
            merged = linemerge(lines)
            if merged.geom_type == "MultiLineString":
                # Take the longest segment
                segments = list(merged.geoms)
                segments.sort(key=lambda x: x.length, reverse=True)
                merged = segments[0]
        else:
            merged = lines[0]
        
        # Get coordinates
        coords = list(merged.coords)
        length = line_length_meters(coords)
        
        if length < MIN_SEGMENT_LENGTH_M:
            continue
        
        # Interpolate points for elevation sampling
        sampled_coords = interpolate_line(coords, SAMPLE_STEP_M)
        
        # Get elevations
        if use_open_elevation:
            elevations = get_elevations_open_elevation(sampled_coords)
        else:
            elevations = dem.get_elevations_batch(sampled_coords)
        
        # Calculate slopes
        stats = calculate_slopes(sampled_coords, elevations)
        
        if stats is None:
            continue
        
        info = road_info.get(name, {})
        
        # Check OSM incline tag
        osm_incline = info.get("incline")
        confidence = "computed"
        
        if osm_incline:
            # Parse OSM incline value
            try:
                incline_val = float(osm_incline.replace("%", "").replace("°", ""))
                if "°" in str(osm_incline):
                    # Convert degrees to percent
                    incline_val = math.tan(math.radians(incline_val)) * 100
                # Trust OSM tag if it's higher than computed
                if abs(incline_val) > stats["avg_slope"]:
                    stats["avg_slope"] = abs(incline_val)
                    confidence = "osm_tag"
            except:
                pass
        
        # Apply filters
        if stats["avg_slope"] < min_avg_slope and stats["max_slope"] < min_max_slope:
            continue
        
        # Check if name suggests a hill
        name_lower = name.lower()
        hill_keywords = ["montée", "côte", "colline", "hill", "mount", "chemin"]
        if any(kw in name_lower for kw in hill_keywords):
            confidence = "computed_verified" if confidence == "computed" else confidence
        
        results.append(SteepRoad(
            name=name if name else "Unnamed Road",
            highway_type=info.get("highway", "unknown"),
            length_m=round(stats["length_m"], 1),
            elevation_gain_m=round(stats["elevation_gain"], 1),
            elevation_loss_m=round(stats["elevation_loss"], 1),
            avg_slope_percent=round(stats["avg_slope"], 2),
            max_slope_percent=round(stats["max_slope"], 2),
            start_elevation_m=round(stats["start_elevation"], 1),
            end_elevation_m=round(stats["end_elevation"], 1),
            geometry=[(lon, lat) for lon, lat in sampled_coords],
            osm_id=info.get("osm_id"),
            incline_tag=osm_incline,
            surface=info.get("surface"),
            confidence=confidence
        ))
    
    # Sort by average slope descending
    results.sort(key=lambda x: x.avg_slope_percent, reverse=True)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Find steep roads around a location",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python find_steep_roads.py --lat 45.75 --lon 4.83 --radius 3000
    python find_steep_roads.py --city "Lyon, France" --radius 5000
    python find_steep_roads.py --city "San Francisco" --min-slope 8 --top 20
        """
    )
    
    # Location options
    parser.add_argument("--lat", type=float, help="Latitude of center point")
    parser.add_argument("--lon", type=float, help="Longitude of center point")
    parser.add_argument("--city", type=str, help="City name to geocode")
    
    # Search parameters
    parser.add_argument("--radius", type=float, default=5000, 
                        help="Search radius in meters (default: 5000)")
    parser.add_argument("--min-slope", type=float, default=MIN_AVG_SLOPE,
                        help=f"Minimum average slope %% (default: {MIN_AVG_SLOPE})")
    parser.add_argument("--min-max-slope", type=float, default=MIN_MAX_SLOPE,
                        help=f"Minimum max slope %% (default: {MIN_MAX_SLOPE})")
    
    # Output options
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path")
    parser.add_argument("--top", type=int, help="Only output top N results")
    parser.add_argument("--include-unnamed", action="store_true",
                        help="Include roads without names")
    
    # Highway type filtering
    parser.add_argument("--cycling-only", action="store_true",
                        help="Only include roads suitable for cycling")
    
    # Elevation source
    parser.add_argument("--use-api", action="store_true",
                        help="Use Open Elevation API instead of DEM files")
    
    args = parser.parse_args()
    
    # Get center coordinates
    if args.city:
        lat, lon = geocode_location(args.city)
        print(f"Geocoded '{args.city}' to ({lat:.4f}, {lon:.4f})")
        location_name = args.city.replace(" ", "_").replace(",", "")
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        location_name = f"{lat:.2f}_{lon:.2f}"
    else:
        parser.error("Either --city or both --lat and --lon are required")
        return
    
    # Select highway types
    highway_types = HIGHWAY_TYPES_CYCLING if args.cycling_only else HIGHWAY_TYPES_ALL
    
    # Find steep roads
    roads = find_steep_roads(
        lat=lat,
        lon=lon,
        radius_m=args.radius,
        min_avg_slope=args.min_slope,
        min_max_slope=args.min_max_slope,
        highway_types=highway_types,
        include_unnamed=args.include_unnamed,
        use_open_elevation=args.use_api
    )
    
    # Limit results if requested
    if args.top:
        roads = roads[:args.top]
    
    # Output
    output_file = args.output or f"steep_roads_{location_name}.json"
    
    # Convert to dict for JSON serialization
    output_data = [asdict(road) for road in roads]
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Found {len(roads)} steep roads")
    print(f"📁 Saved to {output_file}")
    
    # Print summary
    if roads:
        print(f"\nTop 10 steepest roads:")
        print("-" * 80)
        for i, road in enumerate(roads[:10], 1):
            conf = "⭐" if road.confidence == "osm_tag" else ""
            print(f"{i:2}. {road.name[:40]:<40} | {road.avg_slope_percent:5.1f}% avg | "
                  f"{road.max_slope_percent:5.1f}% max | {road.length_m:6.0f}m {conf}")


if __name__ == "__main__":
    main()
