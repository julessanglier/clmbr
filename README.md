# Climbr - Find Steep Roads for Running and Cycling

Climbr is an experimental project to discover and retrieve the steepest nearby roads — focused on use by steep runners and road cyclists.

## What it does
- Extracts and ranks road segments by steepness using OSM road geometry and elevation sources.
- Prioritises running-suitable roads (footways, paths, residential streets) and filters out major highways.

## Files of interest
- Code and sample data live under [apps/data](apps/data).
- Primary extractor script: [apps/data/v3.py](apps/data/v3.py)

## Devbox (brief)
- This repo includes a `devbox.json` (see `apps/data/devbox.json`). To use the devbox environment: install `devbox`, then either run from the repository root:

```bash
devbox shell apps/data
```

or change into `apps/data` and run:

```bash
cd apps/data
devbox shell
```

Once inside the devbox, you can run the main script with Python 3:

```bash
python3 v3.py --city "Lyon, France" --radius 3000
```

## About `apps/data/v3.py`
- `v3.py` searches for running-suitable roads via the Overpass API, samples geometry, and gathers elevations from local SRTM tiles (fast) with an Open-Topo-Data fallback.
- It interpolates and smooths elevation samples, computes point-to-point and aggregate slope statistics, and applies filters (minimum slope, elevation change, length, and OSM tags) to produce a ranked list of steep segments.
- Output is written as JSON (see script `--output` option).

## Status
- Experimental: heuristics, thresholds and the API are works in progress. No API usage details are provided here.

## TODO
- make an API (wip)
- make a mobile app

## License
This project is released under the MIT License — see the LICENSE file.

