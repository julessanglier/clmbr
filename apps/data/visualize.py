import json
import folium
from shapely.geometry import LineString

# =========================
# CONFIGURATION
# =========================

INPUT_JSON = "steep_roads_Orléans.json"
MAP_OUTPUT = "steep_roads_Orléans.html"

# =========================
# LOAD RESULTS
# =========================

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    roads = json.load(f)

print(f"Loaded {len(roads)} steep roads")

# =========================
# CREATE MAP CENTERED ON LYON
# =========================

# Rough center of your bounding box
MAP_CENTER = [45.75, 4.83]
m = folium.Map(location=MAP_CENTER, zoom_start=13, tiles="CartoDB positron")

# =========================
# ADD ROADS TO MAP
# =========================

for road in roads:
    line_coords = [(lat, lon) for lon, lat in road["geometry"]]  # Folium expects (lat, lon)
    avg_slope = road.get("avg_slope_percent", 0)

    # Color coding based on slope
    if avg_slope >= 10:
        color = "red"
    elif avg_slope >= 7:
        color = "orange"
    else:
        color = "yellow"

    tooltip = f"{road.get('name','Unnamed')} | Avg slope: {avg_slope:.1f}% | Gain: {road.get('elevation_gain_m',0)}m | Length: {road.get('length_m',0)}m"
    
    folium.PolyLine(
        locations=line_coords,
        color=color,
        weight=4,
        opacity=0.8,
        tooltip=tooltip
    ).add_to(m)

# =========================
# SAVE MAP
# =========================

m.save(MAP_OUTPUT)
print(f"Map saved to {MAP_OUTPUT}")