from fastapi import HTTPException

from app.services.models.base import RoadFinderModel
from app.services.models.v3_srtm import V3SrtmFinder
from app.services.models.v4_copernicus import V4CopernicusFinder
from app.services.models.v5_osm_tags import V5OsmTagsFinder

_REGISTRY: dict[str, RoadFinderModel] = {
    "v3": V3SrtmFinder(),
    "v4": V4CopernicusFinder(),
    "v5": V5OsmTagsFinder(),
}


def get_model(name: str) -> RoadFinderModel:
    model = _REGISTRY.get(name)
    if model is None:
        raise HTTPException(status_code=400, detail=f"Unknown model: {name}")
    return model
