import { useState, useRef, useEffect, useCallback } from 'react'
import { Input } from '@/components/ui/input'

interface Suggestion {
  name: string
  lat: number
  lon: number
}

interface CitySearchProps {
  onSelect: (lat: number, lon: number) => void
}

export default function CitySearch({ onSelect }: CitySearchProps) {
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
      <Input
        placeholder="Search city..."
        value={query}
        onChange={(e) => handleChange(e.target.value)}
        onFocus={() => suggestions.length > 0 && setOpen(true)}
        className="truncate bg-black/70 text-white placeholder:text-white/40 border-white/10 backdrop-blur-sm"
      />
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
