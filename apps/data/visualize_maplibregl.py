#!/usr/bin/env python3
"""
Visualize steep roads on an interactive dark map using MapLibre GL JS.

Works with the output from find_steep_roads_v2.py
"""

import json
import argparse


def slope_to_color(slope: float) -> str:
    """Convert slope percentage to a color (green -> yellow -> red)."""
    if slope < 5:
        return "#22c55e"  # Green
    elif slope < 8:
        return "#eab308"  # Yellow
    elif slope < 12:
        return "#f97316"  # Orange
    elif slope < 15:
        return "#ef4444"  # Red
    else:
        return "#dc2626"  # Dark red


def create_map(roads: list, center: tuple = None, output_file: str = "steep_roads_map.html"):
    """Create an interactive MapLibre GL JS map with steep roads."""
    
    if not roads:
        print("No roads to display!")
        return
    
    # Calculate map center from roads if not provided
    if center is None:
        all_lats = []
        all_lons = []
        for road in roads:
            for lon, lat in road["geometry"]:
                all_lats.append(lat)
                all_lons.append(lon)
        center = (sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons))
    
    # Build GeoJSON features
    features = []
    for road in roads:
        coords = road["geometry"]
        avg_slope = road.get("avg_slope_percent", 0)
        
        feature = {
            "type": "Feature",
            "properties": {
                "name": road.get("name", "Unnamed"),
                "avg_slope": avg_slope,
                "max_slope": road.get("max_slope_percent", 0),
                "length": road.get("length_m", 0),
                "gain": road.get("elevation_gain_m", 0),
                "loss": road.get("elevation_loss_m", 0),
                "start_elev": road.get("start_elevation_m", 0),
                "end_elev": road.get("end_elevation_m", 0),
                "highway_type": road.get("highway_type", "unknown"),
                "surface": road.get("surface") or "unknown",
                "incline_tag": road.get("incline_tag") or "",
                "color": slope_to_color(avg_slope),
                "width": max(3, min(8, avg_slope / 2)),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            }
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Steep Roads Map</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://unpkg.com/maplibre-gl@4.1.2/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@4.1.2/dist/maplibre-gl.css" rel="stylesheet" />
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
        
        .legend {{
            position: absolute;
            bottom: 30px;
            left: 20px;
            background: rgba(20, 20, 20, 0.9);
            padding: 15px 20px;
            border-radius: 8px;
            color: #fff;
            font-size: 13px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            backdrop-filter: blur(10px);
        }}
        .legend h4 {{ 
            margin: 0 0 12px 0; 
            font-size: 14px; 
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .legend-item {{ 
            display: flex; 
            align-items: center; 
            margin: 6px 0;
        }}
        .legend-color {{ 
            width: 24px; 
            height: 4px; 
            border-radius: 2px;
            margin-right: 10px; 
        }}
        
        .maplibregl-popup-content {{
            background: rgba(20, 20, 20, 0.95) !important;
            color: #fff !important;
            padding: 16px 20px !important;
            border-radius: 10px !important;
            box-shadow: 0 8px 30px rgba(0,0,0,0.6) !important;
            backdrop-filter: blur(10px);
            min-width: 220px;
        }}
        .maplibregl-popup-anchor-bottom .maplibregl-popup-tip {{
            border-top-color: rgba(20, 20, 20, 0.95) !important;
        }}
        .maplibregl-popup-anchor-top .maplibregl-popup-tip {{
            border-bottom-color: rgba(20, 20, 20, 0.95) !important;
        }}
        .maplibregl-popup-anchor-left .maplibregl-popup-tip {{
            border-right-color: rgba(20, 20, 20, 0.95) !important;
        }}
        .maplibregl-popup-anchor-right .maplibregl-popup-tip {{
            border-left-color: rgba(20, 20, 20, 0.95) !important;
        }}
        .maplibregl-popup-close-button {{
            color: #888 !important;
            font-size: 20px !important;
            padding: 4px 10px !important;
        }}
        
        .popup-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #fff;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }}
        .popup-row {{
            display: flex;
            justify-content: space-between;
            margin: 6px 0;
            font-size: 13px;
        }}
        .popup-label {{
            color: #888;
        }}
        .popup-value {{
            color: #fff;
            font-weight: 500;
        }}
        .popup-slope {{
            font-size: 24px;
            font-weight: 700;
            margin: 8px 0;
        }}
        .popup-tag {{
            display: inline-block;
            background: #22c55e;
            color: #000;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 8px;
        }}
        
        .info-panel {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(20, 20, 20, 0.9);
            padding: 15px 20px;
            border-radius: 8px;
            color: #fff;
            font-size: 14px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        .info-panel h3 {{
            margin: 0 0 5px 0;
            font-size: 18px;
        }}
        .info-panel p {{
            margin: 0;
            color: #888;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="info-panel">
        <h3>🏃 Steep Roads</h3>
        <p>{len(roads)} roads found</p>
    </div>
    
    <div class="legend">
        <h4>Slope</h4>
        <div class="legend-item">
            <div class="legend-color" style="background: #22c55e;"></div>
            <span>&lt; 5% Gentle</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #eab308;"></div>
            <span>5-8% Moderate</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #f97316;"></div>
            <span>8-12% Steep</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ef4444;"></div>
            <span>12-15% Very Steep</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #dc2626;"></div>
            <span>&gt; 15% Extreme</span>
        </div>
    </div>

    <script>
        const geojsonData = {json.dumps(geojson)};
        
        const map = new maplibregl.Map({{
            container: 'map',
            style: {{
                version: 8,
                sources: {{
                    'carto-dark': {{
                        type: 'raster',
                        tiles: [
                            'https://a.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}@2x.png',
                            'https://b.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}@2x.png',
                            'https://c.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}@2x.png'
                        ],
                        tileSize: 256,
                        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                    }}
                }},
                layers: [{{
                    id: 'carto-dark-layer',
                    type: 'raster',
                    source: 'carto-dark',
                    minzoom: 0,
                    maxzoom: 20
                }}]
            }},
            center: [{center[1]}, {center[0]}],
            zoom: 14
        }});

        map.addControl(new maplibregl.NavigationControl(), 'top-right');
        map.addControl(new maplibregl.FullscreenControl(), 'top-right');

        map.on('load', () => {{
            map.addSource('roads', {{
                type: 'geojson',
                data: geojsonData
            }});

            // Glow effect layer
            map.addLayer({{
                id: 'roads-glow',
                type: 'line',
                source: 'roads',
                layout: {{
                    'line-join': 'round',
                    'line-cap': 'round'
                }},
                paint: {{
                    'line-color': ['get', 'color'],
                    'line-width': ['+', ['get', 'width'], 6],
                    'line-opacity': 0.3,
                    'line-blur': 4
                }}
            }});

            // Main road layer
            map.addLayer({{
                id: 'roads-layer',
                type: 'line',
                source: 'roads',
                layout: {{
                    'line-join': 'round',
                    'line-cap': 'round'
                }},
                paint: {{
                    'line-color': ['get', 'color'],
                    'line-width': ['get', 'width'],
                    'line-opacity': 0.9
                }}
            }});

            // Hover effect
            map.on('mouseenter', 'roads-layer', () => {{
                map.getCanvas().style.cursor = 'pointer';
            }});

            map.on('mouseleave', 'roads-layer', () => {{
                map.getCanvas().style.cursor = '';
            }});

            // Click popup
            map.on('click', 'roads-layer', (e) => {{
                const props = e.features[0].properties;
                const coords = e.lngLat;
                
                const elevChange = Math.max(props.gain, props.loss);
                const direction = props.gain > props.loss ? '↗' : '↘';
                const tagBadge = props.incline_tag ? '<span class="popup-tag">OSM Tag</span>' : '';
                
                const html = `
                    <div class="popup-title">${{props.name}}${{tagBadge}}</div>
                    <div class="popup-slope" style="color: ${{props.color}}">${{props.avg_slope.toFixed(1)}}%</div>
                    <div class="popup-row">
                        <span class="popup-label">Max slope</span>
                        <span class="popup-value">${{props.max_slope.toFixed(1)}}%</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Length</span>
                        <span class="popup-value">${{Math.round(props.length)}}m</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Elevation ${{direction}}</span>
                        <span class="popup-value">${{Math.round(props.start_elev)}}m → ${{Math.round(props.end_elev)}}m</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Type</span>
                        <span class="popup-value">${{props.highway_type}}</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Surface</span>
                        <span class="popup-value">${{props.surface}}</span>
                    </div>
                `;
                
                new maplibregl.Popup({{ closeButton: true, maxWidth: '300px' }})
                    .setLngLat(coords)
                    .setHTML(html)
                    .addTo(map);
            }});
        }});
    </script>
</body>
</html>'''
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Map saved to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Visualize steep roads on a dark map")
    parser.add_argument("input", help="Input JSON file from find_steep_roads_v2.py")
    parser.add_argument("-o", "--output", help="Output HTML file", default=None)
    parser.add_argument("--lat", type=float, help="Map center latitude")
    parser.add_argument("--lon", type=float, help="Map center longitude")
    parser.add_argument("--min-slope", type=float, default=0, 
                        help="Only show roads with avg slope >= this value")
    
    args = parser.parse_args()
    
    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        roads = json.load(f)
    
    print(f"Loaded {len(roads)} roads from {args.input}")
    
    # Filter by minimum slope
    if args.min_slope > 0:
        roads = [r for r in roads if r.get("avg_slope_percent", 0) >= args.min_slope]
        print(f"  Filtered to {len(roads)} roads with avg slope >= {args.min_slope}%")
    
    # Determine output file
    output_file = args.output
    if output_file is None:
        output_file = args.input.replace(".json", ".html")
    
    # Get center
    center = None
    if args.lat is not None and args.lon is not None:
        center = (args.lat, args.lon)
    
    # Create map
    create_map(roads, center, output_file)


if __name__ == "__main__":
    main()
