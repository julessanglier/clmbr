import os
import requests
import gzip
import shutil
import json
import numpy as np
import rasterio
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from pyproj import Transformer
from tqdm import tqdm
import urllib.parse

# =========================
# CONFIGURATION
# =========================
CITY_NAME = "Lyon"                  # Input city
DEM_DIR = "dem_tiles"               # Folder to store DEM tiles
SAMPLE_STEP_M = 5                   # Distance between points for slope calculation
MIN_ROAD_LENGTH_M = 20              # Minimum length to consider
MIN_AVG_SLOPE = 2.0                 # Default slope threshold
LOWER_SLOPE_FOR_NAME = 1.0          # Lower threshold for "Montée"/"Chemin"
SHORT_LENGTH_OVERRIDE = 100         # Short roads override threshold
OUTPUT_FILE = f"steep_roads_{CITY_NAME}.json"

# =========================
# HELPERS
# =========================

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def meters_length(line):
    """Compute length of a LineString in meters using Web Mercator projection."""
    coords = [transformer.transform(*c) for c in line.coords]
    return LineString(coords).length

def parse_incline(incline_value):
    """Convert OSM incline tag to float percent. Returns 0 if not numeric."""
    try:
        return float(str(incline_value).replace("%",""))
    except (ValueError, TypeError):
        return 0.0

def interpolate_nan(array):
    """Interpolate NaN values in a 1D numpy array."""
    n = len(array)
    x = np.arange(n)
    good = ~np.isnan(array)
    if good.sum() < 2:
        return array
    array_interp = np.copy(array)
    array_interp[np.isnan(array)] = np.interp(x[np.isnan(array)], x[good], array[good])
    return array_interp

# =========================
# FETCH CITY BBOX FROM OSM
# =========================
def get_city_bbox(city_name):
    url = f"https://nominatim.openstreetmap.org/search"
    params = {
        "q": city_name,
        "format": "json",
        "limit": 1
    }
    headers = {"User-Agent": "PythonScript"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)

    if resp.status_code != 200:
        raise ValueError(f"Geocoding API error: {resp.status_code}")

    data = resp.json()
    if not data:
        raise ValueError(f"City '{city_name}' not found")

    # bounding box: [south, north, west, east]
    bbox_osm = data[0]["boundingbox"]
    south, north = float(bbox_osm[0]), float(bbox_osm[1])
    west, east = float(bbox_osm[2]), float(bbox_osm[3])
    return (south, west, north, east)

def get_city_center(city_name):
    """Get city center coordinates (lat, lon) from Nominatim."""
    url = f"https://nominatim.openstreetmap.org/search"
    params = {
        "q": city_name,
        "format": "json",
        "limit": 1
    }
    headers = {"User-Agent": "PythonScript"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)

    if resp.status_code != 200:
        raise ValueError(f"Geocoding API error: {resp.status_code}")

    data = resp.json()
    if not data:
        raise ValueError(f"City '{city_name}' not found")

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return (lat, lon)

# =========================
# DETERMINE REQUIRED SRTM TILES
# =========================
import math
def srtm_tiles_for_bbox(bbox):
    minlat, minlon, maxlat, maxlon = bbox
    tiles = []
    for lat in range(math.floor(minlat), math.ceil(maxlat)):
        for lon in range(math.floor(minlon), math.ceil(maxlon)):
            lat_prefix = "N" if lat >= 0 else "S"
            lon_prefix = "E" if lon >= 0 else "W"
            name = f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}"
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat_prefix}{abs(lat):02d}/{name}.hgt.gz"
            tiles.append((name, url))
    return tiles

# =========================
# DOWNLOAD AND CONVERT DEM TILES
# =========================
def download_and_convert_dem(tiles, dem_dir=DEM_DIR):
    os.makedirs(dem_dir, exist_ok=True)
    for name, url in tiles:
        gz_path = os.path.join(dem_dir, f"{name}.hgt.gz")
        hgt_path = os.path.join(dem_dir, f"{name}.hgt")
        tif_path = os.path.join(dem_dir, f"{name}.tif")

        if not os.path.exists(tif_path):
            if not os.path.exists(hgt_path):
                print(f"Downloading {name}...")
                r = requests.get(url, stream=True)
                with open(gz_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
                print(f"Decompressing {name}...")
                with gzip.open(gz_path, 'rb') as f_in, open(hgt_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # Convert HGT -> GeoTIFF
            print(f"Converting {hgt_path} -> {tif_path}...")
            hgt_to_tif(hgt_path, tif_path)

def hgt_to_tif(hgt_path, tif_path):
    """Convert SRTM1 HGT to GeoTIFF usable by rasterio."""
    import numpy as np
    from rasterio.transform import from_origin

    with open(hgt_path, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype='>i2').reshape((3601,3601)).astype(np.int16)

    filename = os.path.basename(hgt_path)
    lat_str, lon_str = filename[:3], filename[3:7]
    lat_origin = int(lat_str[1:]) * (1 if lat_str[0]=='N' else -1)
    lon_origin = int(lon_str[1:]) * (1 if lon_str[0]=='E' else -1)

    pixel_size = 1/3600
    transform = from_origin(lon_origin, lat_origin + 1, pixel_size, pixel_size)

    os.makedirs(os.path.dirname(tif_path), exist_ok=True)
    with rasterio.open(
        tif_path,
        'w', driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=np.int16,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(arr, 1)
    print(f"Converted {hgt_path} -> {tif_path}")

# =========================
# LOAD DEM FILES
# =========================
def load_dems(dem_dir=DEM_DIR):
    dems = []
    for f in os.listdir(dem_dir):
        if f.endswith(".tif"):
            dem = rasterio.open(os.path.join(dem_dir,f))
            arr = dem.read(1)
            dems.append((dem, arr))
    return dems

def elevation(lon, lat, dems):
    for dem, arr in dems:
        try:
            row, col = dem.index(lon, lat)
            val = arr[row, col]
            if val != dem.nodata:
                return float(val)
        except:
            continue
    return np.nan

# =========================
# FETCH OSM ROADS
# =========================
def fetch_osm_roads(city_name, radius_meters=None):
    """
    Fetch OSM roads for a city.
    
    Args:
        city_name: Name of the city
        radius_meters: Optional radius in meters from city center to filter results.
                      If None, uses area-based query (original behavior).
    """
    if radius_meters is not None:
        # Get city center coordinates
        lat, lon = get_city_center(city_name)
        query = f"""
        [out:json][timeout:50];
        (
          way["highway"~"residential|tertiary|secondary|unclassified|service|steps|footway|pedestrian"](around:{radius_meters},{lat},{lon});
        );
        out geom;
        """
        print(f"Fetching OSM roads for {city_name} within {radius_meters}m of city center ({lat}, {lon})...")
    else:
        query = f"""
        [out:json][timeout:50];
        area["name"="{city_name}"]->.searchArea;
        (
          way["highway"~"residential|tertiary|secondary|unclassified|service|steps|footway|pedestrian"](area.searchArea);
        );
        out geom;
        """
        print(f"Fetching OSM roads for {city_name}...")
    resp = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=120)
    return resp.json()

# =========================
# MAIN PIPELINE
# =========================
def main(city_name):
    bbox = get_city_bbox(city_name)
    print(f"City {city_name} bounding box: {bbox}")

    tiles = srtm_tiles_for_bbox(bbox)
    download_and_convert_dem(tiles)

    dems = load_dems()
    osm_data = fetch_osm_roads(city_name, 2 * 1000)

    # Group and merge roads by name
    roads_by_name = {}
    road_tags = {}
    for el in osm_data["elements"]:
        if el["type"] != "way" or "geometry" not in el:
            continue
        name = el.get("tags", {}).get("name", "")
        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(coords) < 2:
            continue
        roads_by_name.setdefault(name, []).append(LineString(coords))
        road_tags[name] = el.get("tags", {})

    merged_roads = []
    for name, lines in roads_by_name.items():
        merged = linemerge(MultiLineString(lines)) if len(lines) > 1 else lines[0]
        if merged.geom_type == "MultiLineString":
            for part in merged.geoms:
                merged_roads.append({"name": name, "geometry": part, "tags": road_tags.get(name,{})})
        else:
            merged_roads.append({"name": name, "geometry": merged, "tags": road_tags.get(name,{})})

    # Compute slopes and apply Option 3 hybrid logic
    results = []
    for road in tqdm(merged_roads, desc="Processing roads"):
        line = road["geometry"]
        length_m = meters_length(line)
        if length_m < MIN_ROAD_LENGTH_M:
            continue

        distances = np.arange(0, length_m + SAMPLE_STEP_M, SAMPLE_STEP_M)
        points = [line.interpolate(d / length_m, normalized=True) for d in distances]
        lons = [p.x for p in points]
        lats = [p.y for p in points]
        elevs = np.array([elevation(lon, lat, dems) for lon, lat in zip(lons, lats)])
        elevs = interpolate_nan(elevs)
        valid_mask = ~np.isnan(elevs)
        if valid_mask.sum() < 2:
            continue

        dz = np.diff(elevs[valid_mask])
        slopes = (dz / SAMPLE_STEP_M) * 100
        avg_slope = np.mean(np.abs(slopes))
        max_slope = np.max(np.abs(slopes))
        total_gain = np.sum(dz[dz > 0])

        tags = road.get("tags", {})
        incline_tagged = "incline" in tags
        name_includes_montee = any(k in road["name"].lower() for k in ["montée","chemin"])

        slope_threshold = MIN_AVG_SLOPE
        if name_includes_montee:
            slope_threshold = LOWER_SLOPE_FOR_NAME

        keep = False
        if incline_tagged:
            keep = True
            avg_slope = max(avg_slope, parse_incline(tags.get("incline")))
        elif avg_slope >= slope_threshold:
            keep = True
        elif length_m < SHORT_LENGTH_OVERRIDE:
            keep = True

        if not keep:
            continue

        results.append({
            "name": road["name"],
            "length_m": round(length_m,1),
            "elevation_gain_m": round(total_gain,1),
            "avg_slope_percent": round(avg_slope,2),
            "max_slope_percent": round(max_slope,2),
            "geometry": list(zip(lons, lats)),
            "tags": tags
        })

    # Remove empty name roads
    results = [r for r in results if r["name"].strip() != ""]

    # Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Done for {city_name}")
    print(f"Detected {len(results)} steep roads")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main(CITY_NAME)