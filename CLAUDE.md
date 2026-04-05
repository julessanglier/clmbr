# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Context

This is a monorepo for **clmbr** — a tool to find steep roads for running and cycling. The user is non-technical — prefer making changes directly rather than explaining how to do them manually.

## Environment

Uses [devbox](https://www.jetify.com/devbox) to pin Node.js 22 + Python 3 via Nix. Always run commands inside `devbox shell`.

## Commands

Use `just` for all common tasks:

```bash
just install          # install all dependencies (web + backend)
just dev-web          # start Vite dev server (http://localhost:5173)
just dev-backend      # start FastAPI server (http://localhost:8000)
just build            # production build (web)
just lint             # ESLint (web)
just preview          # preview production build (web)
```

## Project structure

```
apps/
├── web/              -- React + Vite + TypeScript frontend
│   └── src/
│       ├── components/
│       │   └── ui/       -- shadcn components (auto-generated, do not edit)
│       ├── lib/
│       │   └── utils.ts  -- cn() helper
│       ├── App.tsx
│       ├── main.tsx
│       └── index.css     -- Tailwind v4 + shadcn CSS variables
├── backend/          -- Python FastAPI backend
│   └── main.py
└── data/             -- Python data pipeline (steep road finder)
    ├── v3.py         -- main extraction logic
    └── api.py        -- FastAPI wrapper for v3
```

## UI — shadcn enforced

**All UI must use shadcn components.** Do not write raw HTML elements for interactive UI (buttons, inputs, dialogs, etc.) or reach for other component libraries.

- Browse available components at https://ui.shadcn.com/docs/components
- Before using a component, install it by running `npx shadcn@latest add <name>` from `apps/web/`
- Components are installed into `apps/web/src/components/ui/` — import from `@/components/ui/<name>`
- Use the `cn()` helper from `@/lib/utils` for conditional class names
- Style with Tailwind utility classes; do not write custom CSS unless there is no Tailwind equivalent

## Guidelines

- Prefer editing existing files over creating new ones.
- Do not add comments unless the logic is genuinely non-obvious.
- React Compiler is not enabled — `useMemo`/`useCallback` are fine to use.
