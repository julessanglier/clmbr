import time

from fastapi import APIRouter, Query

from app.logging import log
from app.schemas.roads import GeocodeSuggestion
from app.services.geocoding import search

router = APIRouter()


@router.get("/geocode", response_model=list[GeocodeSuggestion])
def geocode(q: str = Query(..., min_length=2)):
    log.info("geocode.request", query=q)
    start = time.monotonic()

    results = search(q)

    elapsed = round(time.monotonic() - start, 2)
    log.info("geocode.response", query=q, result_count=len(results), elapsed_s=elapsed)
    return results
