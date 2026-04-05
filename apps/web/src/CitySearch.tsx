import { useState, useRef, useEffect, useCallback } from 'react'
import { Input } from '@/components/ui/input'

interface Suggestion {
  name: string
  lat: number
  lon: number
}

interface CitySearchProps {
  onSelect: (lat: number, lon: number) => void
  onLocate: () => void
}

export default function CitySearch({ onSelect, onLocate }: CitySearchProps) {
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [open, setOpen] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const fetchSuggestions = useCallback(async (q: string) => {
    if (q.length < 2) {
      setSuggestions([])
      return
    }
    try {
      const res = await fetch(`/api/geocode?q=${encodeURIComponent(q)}`)
      if (!res.ok) return
      const data: Suggestion[] = await res.json()
      setSuggestions(data)
      setOpen(data.length > 0)
    } catch {
      // ignore
    }
  }, [])

  function handleChange(value: string) {
    setQuery(value)
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => fetchSuggestions(value), 300)
  }

  function handleSelect(s: Suggestion) {
    setQuery(s.name)
    setOpen(false)
    setSuggestions([])
    onSelect(s.lat, s.lon)
  }

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  return (
    <div ref={containerRef} className="absolute top-4 left-4 z-20 w-72">
      <div className="flex gap-1.5">
        <Input
          placeholder="Search city..."
          value={query}
          onChange={(e) => handleChange(e.target.value)}
          onFocus={() => suggestions.length > 0 && setOpen(true)}
          className="truncate bg-black/70 text-white placeholder:text-white/40 border-white/10 backdrop-blur-sm"
        />
        <button
          onClick={onLocate}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-white/10 bg-black/70 text-white/50 backdrop-blur-sm hover:text-white"
          title="Go to my location"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8" />
            <path d="M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
          </svg>
        </button>
      </div>
      {open && (
        <ul className="mt-1 max-h-60 overflow-y-auto rounded-md bg-black/80 backdrop-blur-sm border border-white/10">
          {suggestions.map((s, i) => (
            <li
              key={i}
              className="cursor-pointer truncate px-3 py-2 text-sm text-white/80 hover:bg-white/10 hover:text-white"
              onClick={() => handleSelect(s)}
            >
              {s.name}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
