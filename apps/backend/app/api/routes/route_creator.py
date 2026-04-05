from fastapi import APIRouter, HTTPException

from app.schemas.routes import RouteRequest, RouteResponse
from app.services.route_builder import build_route

router = APIRouter()


@router.post("/route")
def create_route(req: RouteRequest) -> RouteResponse:
    try:
        result = build_route(
            lat=req.lat,
            lon=req.lon,
            target_distance_km=req.target_distance_km,
            target_elevation_gain_m=req.target_elevation_gain_m,
            mode=req.mode,
        )
        return RouteResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
