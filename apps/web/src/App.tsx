import { useRef, useEffect, useState, useCallback } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import CitySearch from './CitySearch'
import ElevationProfile from './ElevationProfile'
import DevMenu, { type DevParams } from './DevMenu'
import RouteCreator, { type RouteResult } from './RouteCreator'

const DARK_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'

const TERRAIN_SOURCE_ID = 'terrain-dem'

function ensureTerrainSource(m: maplibregl.Map) {
  if (!m.isStyleLoaded() || m.getSource(TERRAIN_SOURCE_ID)) return
  m.addSource(TERRAIN_SOURCE_ID, {
    type: 'raster-dem',
    tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
    encoding: 'terrarium',
    tileSize: 256,
    maxzoom: 15,
  })
}

const BUILDINGS_SOURCE_ID = 'openmaptiles'
const BUILDINGS_LAYER_ID = 'buildings-3d'

function toggle3DTerrain(m: maplibregl.Map, enabled: boolean) {
  if (!m.isStyleLoaded()) return
  ensureTerrainSource(m)
  if (enabled) {
    m.setTerrain({ source: TERRAIN_SOURCE_ID, exaggeration: 1.5 })

    if (!m.getSource(BUILDINGS_SOURCE_ID)) {
      m.addSource(BUILDINGS_SOURCE_ID, {
        type: 'vector',
        url: 'https://tiles.openfreemap.org/planet',
      })
    }
    if (!m.getLayer(BUILDINGS_LAYER_ID)) {
      m.addLayer({
        id: BUILDINGS_LAYER_ID,
        source: BUILDINGS_SOURCE_ID,
        'source-layer': 'building',
        type: 'fill-extrusion',
        paint: {
          'fill-extrusion-color': '#1a1a2e',
          'fill-extrusion-height': ['get', 'render_height'],
          'fill-extrusion-base': ['get', 'render_min_height'],
          'fill-extrusion-opacity': 0.7,
        },
      })
    }
  } else {
    m.setTerrain(undefined as any)
    if (m.getLayer(BUILDINGS_LAYER_ID)) m.removeLayer(BUILDINGS_LAYER_ID)
    if (m.getSource(BUILDINGS_SOURCE_ID)) m.removeSource(BUILDINGS_SOURCE_ID)
  }
}

const ENABLE_CAMERA_TRAVEL = import.meta.env.VITE_ENABLE_CAMERA_TRAVEL === 'true'
const CAMERA_TRAVEL_SPEED = 30 // meters per second
const CAMERA_LOOK_AHEAD_M = 40 // how far ahead the camera looks

function getCameraTravelCoords(
  geojson: GeoJSON.FeatureCollection,
  osmId: string,
  profile: { d: number; e: number }[],
): [number, number, number][] | null {
  const feature = geojson.features.find((f) => {
    const id = f.properties?.osm_id
    return String(id) === String(osmId)
  })
  if (!feature) return null

  const geom = feature.geometry
  if (geom.type !== 'LineString') return null

  const coords = geom.coordinates as [number, number][]
  if (coords.length < 2 || profile.length < 2) return null

  const totalDist = profile[profile.length - 1].d
  const result: [number, number, number][] = []

  for (let i = 0; i < coords.length; i++) {
    const t = i / (coords.length - 1)
    const d = t * totalDist
    let elev = profile[0].e
    for (let j = 0; j < profile.length - 1; j++) {
      if (d >= profile[j].d && d <= profile[j + 1].d) {
        const segT = (d - profile[j].d) / (profile[j + 1].d - profile[j].d || 1)
        elev = profile[j].e + segT * (profile[j + 1].e - profile[j].e)
        break
      }
    }
    result.push([coords[i][0], coords[i][1], elev])
  }

  // Sort by elevation so we always travel uphill (low → high)
  if (result[0][2] > result[result.length - 1][2]) {
    result.reverse()
  }

  return result
}

function startCameraTravel(
  m: maplibregl.Map,
  path: [number, number, number][],
  onDone: () => void,
): () => void {
  let cancelled = false

  // Build cumulative distance array
  const distances = [0]
  for (let i = 1; i < path.length; i++) {
    const [lon1, lat1] = path[i - 1]
    const [lon2, lat2] = path[i]
    const dlat = (lat2 - lat1) * 111320
    const dlon = (lon2 - lon1) * 111320 * Math.cos(((lat1 + lat2) / 2) * Math.PI / 180)
    distances.push(distances[i - 1] + Math.sqrt(dlat * dlat + dlon * dlon))
  }
  const totalDist = distances[distances.length - 1]
  const totalTime = (totalDist / CAMERA_TRAVEL_SPEED) * 1000 // ms

  function interpAtDist(d: number): [number, number, number] {
    const clamped = Math.max(0, Math.min(d, totalDist))
    for (let i = 0; i < distances.length - 1; i++) {
      if (clamped >= distances[i] && clamped <= distances[i + 1]) {
        const t = (clamped - distances[i]) / (distances[i + 1] - distances[i] || 1)
        return [
          path[i][0] + t * (path[i + 1][0] - path[i][0]),
          path[i][1] + t * (path[i + 1][1] - path[i][1]),
          path[i][2] + t * (path[i + 1][2] - path[i][2]),
        ]
      }
    }
    return path[path.length - 1]
  }

  function bearingBetween(
    lon1: number, lat1: number, lon2: number, lat2: number,
  ): number {
    const dLon = (lon2 - lon1) * Math.PI / 180
    const lat1r = lat1 * Math.PI / 180
    const lat2r = lat2 * Math.PI / 180
    const y = Math.sin(dLon) * Math.cos(lat2r)
    const x = Math.cos(lat1r) * Math.sin(lat2r) -
              Math.sin(lat1r) * Math.cos(lat2r) * Math.cos(dLon)
    return ((Math.atan2(y, x) * 180 / Math.PI) + 360) % 360
  }

  const startTime = performance.now()

  function frame(now: number) {
    if (cancelled) return
    const elapsed = now - startTime
    const progress = Math.min(elapsed / totalTime, 1)

    const currentDist = progress * totalDist
    const lookDist = Math.min(currentDist + CAMERA_LOOK_AHEAD_M, totalDist)

    const pos = interpAtDist(currentDist)
    const lookAt = interpAtDist(lookDist)
    const bearing = bearingBetween(pos[0], pos[1], lookAt[0], lookAt[1])

    m.jumpTo({
      center: [pos[0], pos[1]],
      zoom: 20,
      pitch: 85,
      bearing,
    })

    if (progress < 1) {
      requestAnimationFrame(frame)
    } else {
      onDone()
    }
  }

  requestAnimationFrame(frame)

  return () => { cancelled = true }
}

// -- Animus-style scan animation over search area while loading --

const ANIMUS_PULSE_S = 4 // time for one full center-to-edge pulse

function makeDisc(
  lat: number, lon: number, radiusM: number, steps: number = 80,
): [number, number][] {
  if (radiusM < 1) return [[lon, lat], [lon, lat], [lon, lat], [lon, lat]]
  const coords: [number, number][] = []
  const dLon = 1 / (111320 * Math.cos((lat * Math.PI) / 180))
  const dLat = 1 / 111320
  for (let i = 0; i <= steps; i++) {
    const a = (i / steps) * 2 * Math.PI
    coords.push([lon + radiusM * Math.cos(a) * dLon, lat + radiusM * Math.sin(a) * dLat])
  }
  return coords
}

function startScanAnimation(
  m: maplibregl.Map, lat: number, lon: number, radiusM: number,
): () => void {
  let cancelled = false
  const empty: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: [] }

  if (!m.getSource('animus-src')) m.addSource('animus-src', { type: 'geojson', data: empty })
  // Outer glow — wide blurred halo
  if (!m.getLayer('animus-glow')) {
    m.addLayer({
      id: 'animus-glow', type: 'line', source: 'animus-src',
      paint: { 'line-color': '#bae6fd', 'line-width': 20, 'line-opacity': 0, 'line-blur': 14 },
    })
  }
  // Inner shimmer
  if (!m.getLayer('animus-inner')) {
    m.addLayer({
      id: 'animus-inner', type: 'line', source: 'animus-src',
      paint: { 'line-color': '#7dd3fc', 'line-width': 6, 'line-opacity': 0, 'line-blur': 4 },
    })
  }
  // Sharp bright edge
  if (!m.getLayer('animus-edge')) {
    m.addLayer({
      id: 'animus-edge', type: 'line', source: 'animus-src',
      paint: { 'line-color': '#f0f9ff', 'line-width': 2.5, 'line-opacity': 0 },
    })
  }

  const startTime = performance.now()

  function frame(now: number) {
    if (cancelled) return
    const elapsed = (now - startTime) / 1000
    // t goes 0→1 per pulse, then resets
    const t = (elapsed % ANIMUS_PULSE_S) / ANIMUS_PULSE_S

    // Ease-out: fast start, decelerates toward edge
    const eased = 1 - Math.pow(1 - t, 3)
    const waveR = radiusM * eased

    // Opacity: fade in quickly, hold, fade out in last 10%
    const opacity = t < 0.05
      ? t / 0.05
      : t < 0.9
        ? 1
        : (1 - t) / 0.1

    const ring = makeDisc(lat, lon, waveR)
    ;(m.getSource('animus-src') as maplibregl.GeoJSONSource)?.setData({
      type: 'FeatureCollection',
      features: [{ type: 'Feature', geometry: { type: 'LineString', coordinates: ring }, properties: {} }],
    })

    if (m.getLayer('animus-glow')) m.setPaintProperty('animus-glow', 'line-opacity', opacity * 0.4)
    if (m.getLayer('animus-inner')) m.setPaintProperty('animus-inner', 'line-opacity', opacity * 0.5)
    if (m.getLayer('animus-edge')) m.setPaintProperty('animus-edge', 'line-opacity', opacity * 0.9)

    requestAnimationFrame(frame)
  }

  requestAnimationFrame(frame)

  return () => {
    cancelled = true
    for (const id of ['animus-edge', 'animus-inner', 'animus-glow']) {
      if (m.getLayer(id)) m.removeLayer(id)
    }
    if (m.getSource('animus-src')) m.removeSource('animus-src')
  }
}

const DEFAULT_CENTER: [number, number] = [2.3522, 48.8566]
const DEFAULT_ZOOM = 12

/** Convert map zoom level to a search radius in meters.
 *  zoom 16 → ~300m, zoom 14 → ~1200m, zoom 12 → ~4000m, zoom 10 → ~10000m */
function radiusFromZoom(zoom: number): number {
  const metersPerPx = 156543.03 * Math.cos(0) / Math.pow(2, zoom)
  // Use ~40% of the viewport width (assume ~800px) as radius, clamped
  const r = metersPerPx * 320
  return Math.max(300, Math.min(r, 10000))
}

type Profiles = Record<string, { d: number; e: number; s: number }[]>

interface RoadProperties {
  name: string
  highway_type: string
  length_m: number
  elevation_gain_m: number
  elevation_loss_m: number
  avg_slope_percent: number
  max_slope_percent: number
  start_elevation_m: number
  end_elevation_m: number
  osm_id: number
  incline_tag?: string
  surface?: string
  access?: string
}

type RoadPropsMap = Record<string, RoadProperties>

function addRoadsLayer(
  map: maplibregl.Map,
  geojson: GeoJSON.FeatureCollection,
  onRoadClick: (osmId: string, name: string) => void,
) {
  if (map.getSource('roads')) {
    (map.getSource('roads') as maplibregl.GeoJSONSource).setData(geojson)
    return
  }

  map.addSource('roads', { type: 'geojson', data: geojson })

  map.addLayer({
    id: 'roads-line',
    type: 'line',
    source: 'roads',
    paint: {
      'line-color': [
        'interpolate',
        ['linear'],
        ['get', 'avg_slope_percent'],
        5, '#22c55e',
        7, '#eab308',
        9, '#f97316',
        12, '#ef4444',
      ],
      'line-width': [
        'interpolate',
        ['linear'],
        ['get', 'avg_slope_percent'],
        5, 2,
        12, 5,
      ],
      'line-opacity': 0.85,
    },
    layout: {
      'line-cap': 'round',
      'line-join': 'round',
    },
  })

  // Glow layer behind the selected road
  map.addLayer({
    id: 'roads-selected-glow',
    type: 'line',
    source: 'roads',
    filter: ['==', ['get', 'osm_id'], ''],
    paint: {
      'line-color': '#ffffff',
      'line-width': 12,
      'line-opacity': 0.25,
      'line-blur': 6,
    },
    layout: {
      'line-cap': 'round',
      'line-join': 'round',
    },
  })

  // Bright highlight on top of the selected road
  map.addLayer({
    id: 'roads-selected-line',
    type: 'line',
    source: 'roads',
    filter: ['==', ['get', 'osm_id'], ''],
    paint: {
      'line-color': '#ffffff',
      'line-width': 4,
      'line-opacity': 1,
    },
    layout: {
      'line-cap': 'round',
      'line-join': 'round',
    },
  })

  const popup = new maplibregl.Popup({
    closeButton: false,
    closeOnClick: false,
    className: 'road-popup',
    offset: 12,
  })

  map.on('mouseenter', 'roads-line', (e) => {
    map.getCanvas().style.cursor = 'pointer'
    const props = e.features?.[0]?.properties
    if (!props) return

    popup
      .setLngLat(e.lngLat)
      .setHTML(
        `<div style="font-family:system-ui,sans-serif;font-size:12px;line-height:1.5;color:#e5e5e5">
          <div style="font-weight:600;color:#fff">${props.name}</div>
          <div style="color:#999;font-size:11px;margin-top:2px">
            ${props.avg_slope_percent}% avg · ${props.max_slope_percent}% max · ${Math.round(props.length_m)}m
          </div>
        </div>`,
      )
      .addTo(map)
  })

  map.on('mousemove', 'roads-line', (e) => {
    popup.setLngLat(e.lngLat)
  })

  map.on('mouseleave', 'roads-line', () => {
    map.getCanvas().style.cursor = ''
    popup.remove()
  })

  map.on('click', 'roads-line', (e) => {
    const props = e.features?.[0]?.properties
    if (!props) return
    popup.remove()
    onRoadClick(String(props.osm_id), props.name)
  })
}

function highlightRoad(m: maplibregl.Map, osmId: string | null) {
  if (!m.isStyleLoaded() || !m.getLayer('roads-selected-line')) return

  if (osmId) {
    // osm_id may be stored as number in GeoJSON properties, match both
    const filter: maplibregl.FilterSpecification = [
      'any',
      ['==', ['get', 'osm_id'], osmId],
      ['==', ['to-string', ['get', 'osm_id']], osmId],
    ]
    m.setFilter('roads-selected-line', filter)
    m.setFilter('roads-selected-glow', filter)
    // Dim unselected roads
    m.setPaintProperty('roads-line', 'line-opacity', 0.3)
  } else {
    m.setFilter('roads-selected-line', ['==', ['get', 'osm_id'], ''])
    m.setFilter('roads-selected-glow', ['==', ['get', 'osm_id'], ''])
    m.setPaintProperty('roads-line', 'line-opacity', 0.85)
  }
}

function flyToRoad(m: maplibregl.Map, osmId: string) {
  const source = m.getSource('roads') as maplibregl.GeoJSONSource | undefined
  if (!source) return

  // Query rendered + source features to find the geometry
  const features = m.querySourceFeatures('roads', {
    filter: [
      'any',
      ['==', ['get', 'osm_id'], osmId],
      ['==', ['to-string', ['get', 'osm_id']], osmId],
    ],
  })

  if (features.length === 0) return

  // Compute bounding box across all matching features
  const bounds = new maplibregl.LngLatBounds()
  for (const f of features) {
    const geom = f.geometry
    if (geom.type === 'LineString') {
      for (const coord of geom.coordinates) {
        bounds.extend(coord as [number, number])
      }
    } else if (geom.type === 'MultiLineString') {
      for (const line of geom.coordinates) {
        for (const coord of line) {
          bounds.extend(coord as [number, number])
        }
      }
    }
  }

  if (bounds.isEmpty()) return

  m.fitBounds(bounds, {
    padding: { top: 80, bottom: 80, left: 80, right: 400 },
    maxZoom: 17,
    duration: 1000,
  })
}

function addRouteLayer(m: maplibregl.Map, geojson: GeoJSON.FeatureCollection) {
  if (m.getSource('generated-route')) {
    (m.getSource('generated-route') as maplibregl.GeoJSONSource).setData(geojson)
    return
  }

  m.addSource('generated-route', { type: 'geojson', data: geojson })

  m.addLayer({
    id: 'generated-route-glow',
    type: 'line',
    source: 'generated-route',
    paint: {
      'line-color': '#8b5cf6',
      'line-width': 12,
      'line-opacity': 0.2,
      'line-blur': 4,
    },
    layout: { 'line-cap': 'round', 'line-join': 'round' },
  })

  m.addLayer({
    id: 'generated-route-line',
    type: 'line',
    source: 'generated-route',
    paint: {
      'line-color': '#8b5cf6',
      'line-width': 4,
      'line-opacity': 0.9,
    },
    layout: { 'line-cap': 'round', 'line-join': 'round' },
  })
}

function removeRouteLayer(m: maplibregl.Map) {
  if (m.getLayer('generated-route-line')) m.removeLayer('generated-route-line')
  if (m.getLayer('generated-route-glow')) m.removeLayer('generated-route-glow')
  if (m.getSource('generated-route')) m.removeSource('generated-route')
}

function fitRouteOnMap(m: maplibregl.Map, geojson: GeoJSON.FeatureCollection) {
  const bounds = new maplibregl.LngLatBounds()
  for (const feature of geojson.features) {
    const geom = feature.geometry
    if (geom.type === 'LineString') {
      for (const coord of geom.coordinates) {
        bounds.extend(coord as [number, number])
      }
    }
  }
  if (!bounds.isEmpty()) {
    m.fitBounds(bounds, {
      padding: { top: 60, bottom: 60, left: 60, right: 400 },
      maxZoom: 16,
      duration: 1000,
    })
  }
}

function App() {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<maplibregl.Map | null>(null)
  const profilesRef = useRef<Profiles>({})
  const roadPropsRef = useRef<RoadPropsMap>({})
  const geojsonRef = useRef<GeoJSON.FeatureCollection | null>(null)
  const [locationError, setLocationError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [roadCount, setRoadCount] = useState<number | null>(null)
  const [selectedRoad, setSelectedRoad] = useState<{ osmId: string; name: string } | null>(null)
  const [reliefOn, setReliefOn] = useState(false)
  const [routeMode, setRouteMode] = useState(false)
  const [routeStart, setRouteStart] = useState<{ lat: number; lon: number } | null>(null)
  const [routeResult, setRouteResult] = useState<RouteResult | null>(null)
  const [routeLoading, setRouteLoading] = useState(false)
  const startMarkerRef = useRef<maplibregl.Marker | null>(null)
  const userMarkerRef = useRef<maplibregl.Marker | null>(null)
  const routeModeRef = useRef(false)
  routeModeRef.current = routeMode
  const lastFetchRef = useRef<{ lat: number; lon: number; zoom: number } | null>(null)
  const loadingRef = useRef(false)
  loadingRef.current = loading
  const [devParams, setDevParams] = useState<DevParams>({
    radius: 2000,
    minSlope: 5,
    model: 'v3',
  })
  const devParamsRef = useRef(devParams)
  devParamsRef.current = devParams

  const handleRoadClick = useCallback((osmId: string, name: string) => {
    setSelectedRoad({ osmId, name })
  }, [])

  const abortRef = useRef<AbortController | null>(null)
  const stopScanRef = useRef<(() => void) | null>(null)
  const fetchRoadsRef = useRef<(m: maplibregl.Map, lat: number, lon: number) => void>(() => {})

  const fetchRoads = useCallback(async (m: maplibregl.Map, lat: number, lon: number) => {
    // Abort any in-flight request and stop its animation
    if (abortRef.current) abortRef.current.abort()
    if (stopScanRef.current) stopScanRef.current()

    const controller = new AbortController()
    abortRef.current = controller

    const dp = devParamsRef.current
    const zoom = m.getZoom()
    const radius = radiusFromZoom(zoom)
    lastFetchRef.current = { lat, lon, zoom }
    setLoading(true)
    setSelectedRoad(null)

    const stopScan = startScanAnimation(m, lat, lon, radius)
    stopScanRef.current = stopScan

    try {
      const params = new URLSearchParams({
        lat: String(lat),
        lon: String(lon),
        radius: String(radius),
        min_slope: String(dp.minSlope),
        model: dp.model,
      })
      const res = await fetch(`/api/roads?${params}`, { signal: controller.signal })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const newGeojson = data.geojson as GeoJSON.FeatureCollection
      const newProfiles = data.profiles as Profiles

      // Merge new results into existing — deduplicate by osm_id
      const existingIds = new Set(
        (geojsonRef.current?.features ?? []).map((f) => String(f.properties?.osm_id)),
      )
      const merged: GeoJSON.FeatureCollection = {
        type: 'FeatureCollection',
        features: [
          ...(geojsonRef.current?.features ?? []),
          ...newGeojson.features.filter((f) => !existingIds.has(String(f.properties?.osm_id))),
        ],
      }

      profilesRef.current = { ...profilesRef.current, ...newProfiles }
      geojsonRef.current = merged
      for (const feature of newGeojson.features) {
        const p = feature.properties as unknown as RoadProperties
        roadPropsRef.current[String(p.osm_id)] = p
      }
      setRoadCount(merged.features.length)
      addRoadsLayer(m, merged, handleRoadClick)
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      console.error('Failed to fetch roads:', err)
      setLocationError('Failed to load roads.')
    } finally {
      stopScan()
      stopScanRef.current = null
      setLoading(false)
    }
  }, [handleRoadClick])
  fetchRoadsRef.current = fetchRoads

  const handleCitySelect = useCallback((lat: number, lon: number) => {
    const m = map.current
    if (!m) return
    setSelectedRoad(null)
    m.flyTo({ center: [lon, lat], zoom: 14 })
    // auto-fetch will trigger via moveend
  }, [])

  const handleRefetch = useCallback(() => {
    const m = map.current
    if (!m) return
    const center = m.getCenter()
    fetchRoads(m, center.lat, center.lng)
  }, [fetchRoads])

  useEffect(() => {
    if (!mapContainer.current) return

    const m = new maplibregl.Map({
      container: mapContainer.current,
      style: DARK_STYLE,
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      maxPitch: 70,
    })

    map.current = m

    m.on('style.load', () => {
      ensureTerrainSource(m)
    })

    // Auto-fetch roads when map settles after user interaction
    let debounceTimer: ReturnType<typeof setTimeout> | null = null

    m.on('moveend', (e) => {
      // Skip programmatic moves (flyTo, easeTo from city search / locate)
      if ((e as any).originalEvent == null && !m.isMoving()) {
        // Could be end of flyTo — let it settle, then check
      }
      if (debounceTimer) clearTimeout(debounceTimer)
      debounceTimer = setTimeout(() => {
        if (routeModeRef.current) return
        if (m.isMoving() || m.isZooming()) return

        const center = m.getCenter()
        const zoom = m.getZoom()
        if (zoom < 11) return

        const last = lastFetchRef.current
        if (last) {
          const dist = Math.sqrt(
            Math.pow(center.lat - last.lat, 2) + Math.pow(center.lng - last.lon, 2)
          ) * 111320
          const zoomDelta = Math.abs(zoom - last.zoom)
          if (dist < radiusFromZoom(zoom) * 0.3 && zoomDelta < 1) return
        }

        fetchRoadsRef.current(m, center.lat, center.lng)
      }, 1200)
    })

    m.on('click', (e) => {
      if (!routeModeRef.current) return
      const { lat, lng: lon } = e.lngLat

      // Remove existing start marker
      if (startMarkerRef.current) {
        startMarkerRef.current.remove()
      }

      const marker = new maplibregl.Marker({ color: '#22c55e' })
        .setLngLat([lon, lat])
        .addTo(m)
      startMarkerRef.current = marker
      setRouteStart({ lat, lon })
    })

    return () => m.remove()
  }, [])

  useEffect(() => {
    const m = map.current
    if (!m) return
    toggle3DTerrain(m, reliefOn)
    m.easeTo({ pitch: reliefOn ? 60 : 0, duration: 800 })
  }, [reliefOn])

  useEffect(() => {
    const m = map.current
    if (!m) return
    highlightRoad(m, selectedRoad?.osmId ?? null)
    if (!selectedRoad) return

    if (ENABLE_CAMERA_TRAVEL && reliefOn && geojsonRef.current) {
      const profile = profilesRef.current[selectedRoad.osmId]
      if (profile) {
        const path = getCameraTravelCoords(geojsonRef.current, selectedRoad.osmId, profile)
        if (path && path.length >= 2) {
          const cancel = startCameraTravel(m, path, () => {})
          return () => cancel()
        }
      }
    }

    flyToRoad(m, selectedRoad.osmId)
  }, [selectedRoad, reliefOn])

  // When entering route mode, deselect road. When leaving, clean up.
  useEffect(() => {
    const m = map.current
    if (routeMode) {
      setSelectedRoad(null)
      if (m) m.getCanvas().style.cursor = 'crosshair'
    } else {
      if (m) {
        m.getCanvas().style.cursor = ''
        removeRouteLayer(m)
      }
      if (startMarkerRef.current) {
        startMarkerRef.current.remove()
        startMarkerRef.current = null
      }
      setRouteStart(null)
      setRouteResult(null)
    }
  }, [routeMode])

  const handleRouteGenerate = useCallback(async (distanceKm: number, elevationGainM: number, mode: string) => {
    if (!routeStart) return
    setRouteLoading(true)
    try {
      const res = await fetch('/api/route', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat: routeStart.lat,
          lon: routeStart.lon,
          target_distance_km: distanceKm,
          target_elevation_gain_m: elevationGainM,
          mode,
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => null)
        throw new Error(err?.detail || `HTTP ${res.status}`)
      }
      const data: RouteResult = await res.json()
      setRouteResult(data)

      const m = map.current
      if (m) {
        addRouteLayer(m, data.geojson as GeoJSON.FeatureCollection)
        fitRouteOnMap(m, data.geojson as GeoJSON.FeatureCollection)
      }
    } catch (err) {
      console.error('Route generation failed:', err)
      setLocationError(err instanceof Error ? err.message : 'Route generation failed')
    } finally {
      setRouteLoading(false)
    }
  }, [routeStart])

  const handleRouteClear = useCallback(() => {
    const m = map.current
    if (m) removeRouteLayer(m)
    setRouteResult(null)
  }, [])

  const handleLocate = useCallback(() => {
    const m = map.current
    if (!m) return
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { longitude, latitude } = pos.coords
        m.flyTo({ center: [longitude, latitude], zoom: 14 })
        if (userMarkerRef.current) userMarkerRef.current.remove()
        userMarkerRef.current = new maplibregl.Marker({ color: '#3b82f6' })
          .setLngLat([longitude, latitude])
          .addTo(m)
        setLocationError(null)
        // auto-fetch will trigger via moveend
      },
      (err) => {
        switch (err.code) {
          case err.PERMISSION_DENIED:
            setLocationError('Location access denied.')
            break
          case err.POSITION_UNAVAILABLE:
            setLocationError('Location unavailable.')
            break
          case err.TIMEOUT:
            setLocationError('Location request timed out.')
            break
        }
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 300000 },
    )
  }, [fetchRoads])

  const selectedProfile = selectedRoad ? profilesRef.current[selectedRoad.osmId] : null
  const selectedProps = selectedRoad ? roadPropsRef.current[selectedRoad.osmId] : null

  return (
    <div className="relative h-screen w-screen">
      <div ref={mapContainer} className="h-full w-full" />
      <CitySearch onSelect={handleCitySelect} onLocate={handleLocate} />
      {import.meta.env.VITE_DEV_MENU === 'true' && (
        <DevMenu
          params={devParams}
          onChange={setDevParams}
          onRefetch={handleRefetch}
          loading={loading}
        />
      )}

      {loading && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 rounded-md bg-black/80 px-4 py-2 text-sm text-white">
          Loading steep roads...
        </div>
      )}

      {!loading && roadCount !== null && !selectedRoad && (
        <div className="absolute bottom-6 left-4 rounded-md bg-black/70 px-3 py-2 text-xs text-white/80">
          {roadCount} steep road{roadCount !== 1 ? 's' : ''} found
          <div className="mt-1 flex items-center gap-2">
            {[
              { color: '#22c55e', label: '5%' },
              { color: '#eab308', label: '7%' },
              { color: '#f97316', label: '9%' },
              { color: '#ef4444', label: '12%+' },
            ].map((s) => (
              <span key={s.label} className="flex items-center gap-1">
                <span
                  className="inline-block h-2 w-4 rounded-sm"
                  style={{ background: s.color }}
                />
                {s.label}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="absolute top-4 right-4 z-10 flex gap-2">
        <button
          onClick={() => setRouteMode((v) => !v)}
          className={`flex h-9 items-center justify-center rounded-md border px-2.5 text-xs font-medium backdrop-blur-sm transition-colors ${
            routeMode
              ? 'border-violet-400/40 bg-violet-500/20 text-violet-300'
              : 'border-white/10 bg-black/70 text-white/50 hover:text-white'
          }`}
          title={routeMode ? 'Exit route creator' : 'Create a route'}
        >
          Route
        </button>
<button
          onClick={() => setReliefOn((v) => !v)}
          className={`flex h-9 items-center justify-center rounded-md border px-2.5 text-xs font-medium backdrop-blur-sm transition-colors ${
            reliefOn
              ? 'border-white/20 bg-white/15 text-white'
              : 'border-white/10 bg-black/70 text-white/50 hover:text-white'
          }`}
          title={reliefOn ? 'Hide relief' : 'Show relief'}
        >
          3D
        </button>
      </div>

      {/* Elevation profile panel (hidden in route mode) */}
      {!routeMode && selectedProfile && selectedRoad && selectedProps && (
        <ElevationProfile
          profile={selectedProfile}
          name={selectedRoad.name}
          properties={selectedProps}
          onClose={() => setSelectedRoad(null)}
        />
      )}

      {/* Route creator panel */}
      {routeMode && (
        <RouteCreator
          start={routeStart}
          result={routeResult}
          loading={routeLoading}
          onGenerate={handleRouteGenerate}
          onClear={handleRouteClear}
          onClose={() => setRouteMode(false)}
        />
      )}

      {locationError && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 rounded-md bg-black/80 px-4 py-2 text-sm text-white">
          {locationError}
        </div>
      )}
    </div>
  )
}

export default App
