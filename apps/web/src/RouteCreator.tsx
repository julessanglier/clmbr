import { useState } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

interface SteepSegment {
  name: string
  length_m: number
  elevation_gain_m: number
  avg_slope_percent: number
}

export interface RouteResult {
  geojson: GeoJSON.FeatureCollection
  total_distance_m: number
  total_elevation_gain_m: number
  total_elevation_loss_m: number
  estimated_duration_min: number
  steep_segments: SteepSegment[]
  profile: { d: number; e: number; s: number }[]
}

interface RouteCreatorProps {
  start: { lat: number; lon: number } | null
  result: RouteResult | null
  loading: boolean
  onGenerate: (distanceKm: number, elevationGainM: number, mode: string) => void
  onClear: () => void
  onClose: () => void
}

function formatDuration(min: number): string {
  const h = Math.floor(min / 60)
  const m = Math.round(min % 60)
  return h > 0 ? `${h}h${m > 0 ? ` ${m}min` : ''}` : `${m} min`
}

function formatDistance(m: number): string {
  return m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`
}

export default function RouteCreator({
  start,
  result,
  loading,
  onGenerate,
  onClear,
  onClose,
}: RouteCreatorProps) {
  const [distanceKm, setDistanceKm] = useState(5)
  const [elevationGain, setElevationGain] = useState(200)
  const [mode, setMode] = useState('foot')

  return (
    <div className="absolute top-0 right-0 bottom-0 z-20 flex w-[360px] flex-col border-l border-white/10 bg-black/85 backdrop-blur-md">
      {/* Header */}
      <div className="flex items-center justify-between px-4 pt-4 pb-2">
        <div className="text-sm font-medium text-white">Route Creator</div>
        <button
          onClick={onClose}
          className="flex h-6 w-6 items-center justify-center rounded text-white/40 hover:bg-white/10 hover:text-white"
        >
          ✕
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 pb-4">
        {/* Step 1: Start point */}
        <div className="mb-4">
          <div className="mb-1.5 text-[10px] font-medium tracking-wide text-white/40 uppercase">
            Start point
          </div>
          {start ? (
            <div className="rounded-md bg-white/5 px-3 py-2 text-xs text-white/70">
              {start.lat.toFixed(5)}, {start.lon.toFixed(5)}
            </div>
          ) : (
            <div className="rounded-md border border-dashed border-white/20 px-3 py-3 text-center text-xs text-white/40">
              Click on the map to set your start point
            </div>
          )}
        </div>

        {/* Step 2: Parameters */}
        <div className="mb-3">
          <label className="mb-1 block text-xs text-white/60">Distance (km)</label>
          <Input
            type="number"
            value={distanceKm}
            onChange={(e) => setDistanceKm(Number(e.target.value))}
            min={1}
            max={50}
            step={0.5}
            className="h-7 bg-white/5 text-xs text-white border-white/10"
          />
        </div>

        <div className="mb-3">
          <label className="mb-1 block text-xs text-white/60">Elevation gain (m)</label>
          <Input
            type="number"
            value={elevationGain}
            onChange={(e) => setElevationGain(Number(e.target.value))}
            min={50}
            max={3000}
            step={50}
            className="h-7 bg-white/5 text-xs text-white border-white/10"
          />
        </div>

        <div className="mb-4">
          <label className="mb-1 block text-xs text-white/60">Mode</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            className="w-full rounded border border-white/10 bg-white/5 px-2 py-1.5 text-xs text-white outline-none"
          >
            <option value="foot">Running</option>
            <option value="bicycle">Cycling</option>
          </select>
        </div>

        <Button
          onClick={() => onGenerate(distanceKm, elevationGain, mode)}
          disabled={!start || loading}
          className="mb-3 w-full border-white/10 bg-white/5 text-xs text-white hover:bg-white/10"
          variant="outline"
          size="sm"
        >
          {loading ? 'Generating...' : 'Generate Route'}
        </Button>

        {/* Results */}
        {result && (
          <>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <MetricCard label="Distance" value={formatDistance(result.total_distance_m)} />
              <MetricCard label="Duration" value={formatDuration(result.estimated_duration_min)} />
              <MetricCard label="Elevation gain" value={`${Math.round(result.total_elevation_gain_m)} m`} sub="D+" />
              <MetricCard label="Elevation loss" value={`${Math.round(result.total_elevation_loss_m)} m`} sub="D−" />
            </div>

            {/* Steep segments included */}
            {result.steep_segments.length > 0 && (
              <div className="mt-3">
                <div className="mb-1.5 text-[10px] font-medium tracking-wide text-white/40 uppercase">
                  Steep segments ({result.steep_segments.length})
                </div>
                <div className="space-y-1">
                  {result.steep_segments.map((seg, i) => (
                    <div key={i} className="rounded bg-white/5 px-2.5 py-1.5 text-[11px] text-white/60">
                      <span className="text-white/90">{seg.name}</span>
                      <span className="ml-1.5">
                        {seg.avg_slope_percent.toFixed(1)}% · {Math.round(seg.length_m)}m · +{Math.round(seg.elevation_gain_m)}m
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-3 flex gap-2">
              <Button
                disabled
                className="flex-1 border-white/10 bg-white/5 text-xs text-white/40"
                variant="outline"
                size="sm"
              >
                Export GPX
              </Button>
              <Button
                onClick={onClear}
                className="flex-1 border-white/10 bg-white/5 text-xs text-white hover:bg-white/10"
                variant="outline"
                size="sm"
              >
                Clear
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-md bg-white/5 px-3 py-2">
      <div className="text-[10px] text-white/40">{label}</div>
      <div className="mt-0.5 text-sm font-semibold text-white">
        {value}
        {sub && <span className="ml-1 text-[10px] font-normal text-white/30">{sub}</span>}
      </div>
    </div>
  )
}
