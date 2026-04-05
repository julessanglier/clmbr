import time

from fastapi import APIRouter, Query

from app.logging import log
from app.services.models import get_model
from app.services.road_finder import roads_to_response

router = APIRouter()


@router.get("/roads")
def get_roads(
    lat: float = Query(...),
    lon: float = Query(...),
    radius: float = Query(default=2000),
    min_slope: float = Query(default=5.0),
    model: str = Query(default="v3"),
):
    log.info("roads.request", lat=lat, lon=lon, radius=radius, min_slope=min_slope, model=model)
    start = time.monotonic()

    finder = get_model(model)
    results = finder.find(
        lat=lat,
        lon=lon,
        radius_m=radius,
        min_avg_slope=min_slope,
    )
    response = roads_to_response(results)

    elapsed = round(time.monotonic() - start, 2)
    log.info("roads.response", model=model, road_count=len(results), elapsed_s=elapsed)
    return response
