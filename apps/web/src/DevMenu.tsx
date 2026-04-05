import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

export interface DevParams {
  radius: number
  minSlope: number
  model: string
}

interface DevMenuProps {
  params: DevParams
  onChange: (params: DevParams) => void
  onRefetch: () => void
  loading: boolean
}

const MODELS = [
  { value: 'v3', label: 'v3 — SRTM' },
  { value: 'v4', label: 'v4 — EU-DEM 25m' },
  { value: 'v5', label: 'v5 — OSM Tags' },
]

export default function DevMenu({ params, onChange, onRefetch, loading }: DevMenuProps) {
  return (
    <div className="absolute top-16 left-4 z-10">
        <div className="w-72 rounded-md border border-white/10 bg-black/85 p-3 backdrop-blur-md">
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-white/40">
            Dev
          </div>

          <label className="mb-1 block text-xs text-white/60">Model</label>
          <select
            value={params.model}
            onChange={(e) => onChange({ ...params, model: e.target.value })}
            className="mb-3 w-full rounded border border-white/10 bg-white/5 px-2 py-1.5 text-xs text-white outline-none"
          >
            {MODELS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>

          <label className="mb-1 block text-xs text-white/60">Radius (m)</label>
          <Input
            type="number"
            value={params.radius}
            onChange={(e) => onChange({ ...params, radius: Number(e.target.value) })}
            min={200}
            max={10000}
            step={100}
            className="mb-3 h-7 bg-white/5 text-xs text-white border-white/10"
          />

          <label className="mb-1 block text-xs text-white/60">Min avg slope (%)</label>
          <Input
            type="number"
            value={params.minSlope}
            onChange={(e) => onChange({ ...params, minSlope: Number(e.target.value) })}
            min={1}
            max={20}
            step={0.5}
            className="mb-3 h-7 bg-white/5 text-xs text-white border-white/10"
          />

          <Button
            onClick={onRefetch}
            disabled={loading}
            variant="outline"
            size="sm"
            className="w-full border-white/10 bg-white/5 text-xs text-white hover:bg-white/10"
          >
            {loading ? 'Loading...' : 'Refetch roads'}
          </Button>
        </div>
    </div>
  )
}
