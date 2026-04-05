from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logging import log
from app.api.routes import health, roads, geocoding, route_creator

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(roads.router, prefix="/api")
app.include_router(geocoding.router, prefix="/api")
app.include_router(route_creator.router, prefix="/api")

log.info("app.started", title=settings.app_name)
