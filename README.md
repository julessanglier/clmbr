# clmbr

Find the steepest roads around you — built for runners and cyclists.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | React 19, TypeScript, Vite, MapLibre GL, Tailwind v4, shadcn/ui |
| Backend | Python, FastAPI, NumPy, SciPy, Shapely |
| Tooling | Devbox (Nix), Just, Cocogitto, Gitleaks |

## Getting started

```bash
devbox shell
just install
```

Run both servers:

```bash
just dev-web       # http://localhost:5173
just dev-backend   # http://localhost:8001
```

## API

### `GET /api/roads`

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `lat` | float | required | Latitude |
| `lon` | float | required | Longitude |
| `radius` | float | 2000 | Search radius (meters) |
| `min_slope` | float | 5.0 | Minimum average slope % |
| `model` | string | v3 | Road finder model |

Returns a GeoJSON `FeatureCollection` with road geometries, slope stats, and elevation profiles.

### `GET /health`

Health check.

## Commands

```bash
just install        # install all dependencies
just dev-web        # start frontend dev server
just dev-backend    # start backend dev server
just build          # production build (web)
just lint           # ESLint
just cog-check      # validate conventional commits
just cog-changelog  # generate changelog
just cog-bump       # bump version
just gitleaks       # scan for secrets
```

## License

MIT
