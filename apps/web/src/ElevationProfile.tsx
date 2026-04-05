import type { ReactNode } from 'react'

interface ProfilePoint {
  d: number
  e: number
  s: number
}

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

interface ElevationProfileProps {
  profile: ProfilePoint[]
  name: string
  properties: RoadProperties
  onClose: () => void
}

function slopeColor(slope: number): string {
  const abs = Math.abs(slope)
  if (abs >= 12) return '#dc2626'
  if (abs >= 10) return '#ea580c'
  if (abs >= 8) return '#f97316'
  if (abs >= 6) return '#fb923c'
  if (abs >= 4) return '#fbbf24'
  if (abs >= 2) return '#fde68a'
  return '#d4d4d4'
}

function formatDistance(m: number): string {
  return m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`
}

function formatSurface(surface: string): string {
  return surface
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function slopeBands(pts: { d: number; s: number }[]) {
  const bands = [
    { label: '< 5%', min: 0, max: 5, dist: 0, color: '#d4d4d4' },
    { label: '5–8%', min: 5, max: 8, dist: 0, color: '#fbbf24' },
    { label: '8–12%', min: 8, max: 12, dist: 0, color: '#f97316' },
    { label: '12%+', min: 12, max: Infinity, dist: 0, color: '#dc2626' },
  ]
  for (let i = 0; i < pts.length - 1; i++) {
    const abs = Math.abs(pts[i].s)
    const segDist = pts[i + 1].d - pts[i].d
    for (const band of bands) {
      if (abs >= band.min && abs < band.max) {
        band.dist += segDist
        break
      }
    }
  }
  return bands
}

export default function ElevationProfile({ profile, name, properties, onClose }: ElevationProfileProps) {
  if (profile.length < 2) return null

  const needsFlip = profile[0].e > profile[profile.length - 1].e
  const reversed = needsFlip ? [...profile].reverse() : null
  const totalD = profile[profile.length - 1].d
  const pts = reversed
    ? reversed.map((p, i) => ({
        d: totalD - p.d,
        e: p.e,
        s: i < reversed.length - 1 ? -reversed[i + 1].s : 0,
      }))
    : profile

  const W = 320
  const H = 180
  const PAD_L = 40
  const PAD_R = 10
  const PAD_T = 10
  const PAD_B = 28
  const chartW = W - PAD_L - PAD_R
  const chartH = H - PAD_T - PAD_B

  const maxD = pts[pts.length - 1].d
  const elevations = pts.map((p) => p.e)
  const minE = Math.min(...elevations)
  const maxE = Math.max(...elevations)
  const elevRange = maxE - minE || 1
  const elevPad = elevRange * 0.1

  const scaleX = (d: number) => PAD_L + (d / maxD) * chartW
  const scaleY = (e: number) =>
    PAD_T + chartH - ((e - (minE - elevPad)) / (elevRange + 2 * elevPad)) * chartH

  const linePath = pts
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${scaleX(p.d).toFixed(1)},${scaleY(p.e).toFixed(1)}`)
    .join(' ')

  const bars: ReactNode[] = []
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i]
    const p1 = pts[i + 1]
    const x0 = scaleX(p0.d)
    const x1 = scaleX(p1.d)
    const y0 = scaleY(p0.e)
    const y1 = scaleY(p1.e)
    const yBottom = PAD_T + chartH
    const color = slopeColor(p0.s)

    const path = `M${x0.toFixed(1)},${yBottom} L${x0.toFixed(1)},${y0.toFixed(1)} L${x1.toFixed(1)},${y1.toFixed(1)} L${x1.toFixed(1)},${yBottom} Z`
    bars.push(<path key={i} d={path} fill={color} opacity={0.7} />)

    const segW = x1 - x0
    if (segW > 18 && Math.abs(p0.s) >= 1) {
      const cx = (x0 + x1) / 2
      const cy = Math.min(y0, y1) - 5
      bars.push(
        <text
          key={`t${i}`}
          x={cx}
          y={Math.max(cy, PAD_T + 8)}
          textAnchor="middle"
          fill="#fff"
          fontSize="8"
          fontFamily="ui-monospace, monospace"
          fontWeight="600"
        >
          {Math.abs(p0.s).toFixed(0)}%
        </text>,
      )
    }
  }

  const yTicks: ReactNode[] = []
  const tickCount = 4
  for (let i = 0; i <= tickCount; i++) {
    const e = minE - elevPad + ((elevRange + 2 * elevPad) * i) / tickCount
    const y = scaleY(e)
    yTicks.push(
      <g key={`y${i}`}>
        <line x1={PAD_L} x2={PAD_L + chartW} y1={y} y2={y} stroke="#333" strokeWidth="0.5" />
        <text x={PAD_L - 5} y={y + 3} textAnchor="end" fill="#666" fontSize="8" fontFamily="ui-monospace, monospace">
          {Math.round(e)}m
        </text>
      </g>,
    )
  }

  const xTicks: ReactNode[] = []
  const distStep = maxD < 200 ? 50 : maxD < 500 ? 100 : maxD < 1000 ? 200 : 500
  for (let d = 0; d <= maxD; d += distStep) {
    const x = scaleX(d)
    xTicks.push(
      <text key={`x${d}`} x={x} y={PAD_T + chartH + 14} textAnchor="middle" fill="#666" fontSize="8" fontFamily="ui-monospace, monospace">
        {d >= 1000 ? `${(d / 1000).toFixed(1)}km` : `${Math.round(d)}m`}
      </text>,
    )
  }

  const bands = slopeBands(pts)
  const totalBandDist = bands.reduce((s, b) => s + b.dist, 0) || 1

  return (
    <div className="absolute top-0 right-0 bottom-0 z-20 flex w-[360px] flex-col border-l border-white/10 bg-black/85 backdrop-blur-md transition-transform duration-300">
      {/* Header */}
      <div className="flex items-start justify-between px-4 pt-4 pb-2">
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium text-white">{name}</div>
          <div className="mt-1 flex flex-wrap gap-x-2 text-xs text-white/50">
            <span>{formatDistance(properties.length_m)}</span>
            <span>{Math.round(minE)}m → {Math.round(maxE)}m</span>
            {properties.surface && (
              <span>{formatSurface(properties.surface)}</span>
            )}
          </div>
        </div>
        <button
          onClick={onClose}
          className="ml-2 flex h-6 w-6 shrink-0 items-center justify-center rounded text-white/40 hover:bg-white/10 hover:text-white"
        >
          ✕
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 pb-4">
        {/* Chart */}
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
          {yTicks}
          {bars}
          <path d={linePath} fill="none" stroke="#fff" strokeWidth="1.5" strokeLinejoin="round" />
          {xTicks}
          <text x={scaleX(0)} y={scaleY(elevations[0]) - 6} textAnchor="start" fill="#fff" fontSize="9" fontFamily="ui-monospace, monospace" fontWeight="600">
            {Math.round(elevations[0])}m
          </text>
          <text x={scaleX(maxD)} y={scaleY(elevations[elevations.length - 1]) - 6} textAnchor="end" fill="#fff" fontSize="9" fontFamily="ui-monospace, monospace" fontWeight="600">
            {Math.round(elevations[elevations.length - 1])}m
          </text>
        </svg>

        {/* Key metrics */}
        <div className="mt-3 grid grid-cols-2 gap-2">
          <MetricCard label="Elevation gain" value={`${Math.round(properties.elevation_gain_m)} m`} sub="D+" />
          <MetricCard label="Elevation loss" value={`${Math.round(properties.elevation_loss_m)} m`} sub="D−" />
          <MetricCard label="Avg slope" value={`${properties.avg_slope_percent.toFixed(1)}%`} />
          <MetricCard label="Max slope" value={`${properties.max_slope_percent.toFixed(1)}%`} />
        </div>

        {/* Slope distribution */}
        <div className="mt-3">
          <div className="mb-1.5 text-[10px] font-medium tracking-wide text-white/40 uppercase">Slope breakdown</div>
          <div className="flex h-3 overflow-hidden rounded-full">
            {bands.map((b) => {
              const pct = (b.dist / totalBandDist) * 100
              if (pct < 0.5) return null
              return (
                <div
                  key={b.label}
                  style={{ width: `${pct}%`, backgroundColor: b.color }}
                  className="h-full opacity-80 first:rounded-l-full last:rounded-r-full"
                />
              )
            })}
          </div>
          <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-white/50">
            {bands.map((b) => {
              const pct = (b.dist / totalBandDist) * 100
              if (pct < 0.5) return null
              return (
                <span key={b.label} className="flex items-center gap-1">
                  <span className="inline-block h-1.5 w-3 rounded-sm" style={{ backgroundColor: b.color }} />
                  {b.label} ({Math.round(pct)}%)
                </span>
              )
            })}
          </div>
        </div>

        {/* Road info */}
        {(properties.surface || properties.highway_type) && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {properties.highway_type && (
              <Tag>{formatSurface(properties.highway_type)}</Tag>
            )}
            {properties.surface && (
              <Tag>{formatSurface(properties.surface)}</Tag>
            )}
            {properties.access && properties.access !== 'yes' && (
              <Tag>Access: {properties.access}</Tag>
            )}
          </div>
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

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full bg-white/8 px-2 py-0.5 text-[10px] text-white/50">
      {children}
    </span>
  )
}
