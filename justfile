web := "apps/web"
backend := "apps/backend"

install:
    cd {{web}} && npm install
    cd {{backend}} && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

dev-web:
    cd {{web}} && npm run dev

dev-backend:
    cd {{backend}} && . .venv/bin/activate && uvicorn app.main:app --reload --port 8001

build:
    cd {{web}} && npm run build

lint:
    cd {{web}} && npm run lint

preview:
    cd {{web}} && npm run preview

# Secret scanning
gitleaks:
    gitleaks detect --source . --verbose

# Conventional commits & releases
cog-check:
    cog check

cog-changelog:
    cog changelog

cog-bump *args:
    cog bump {{args}}
