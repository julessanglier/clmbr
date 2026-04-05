from pydantic import BaseModel


class RouteRequest(BaseModel):
    lat: float
    lon: float
    target_distance_km: float
    target_elevation_gain_m: float
    mode: str = "foot"  # "foot" or "bicycle"


class SteepSegmentSummary(BaseModel):
    name: str
    length_m: float
    elevation_gain_m: float
    avg_slope_percent: float


class RouteResponse(BaseModel):
    geojson: dict
    total_distance_m: float
    total_elevation_gain_m: float
    total_elevation_loss_m: float
    estimated_duration_min: float
    steep_segments: list[SteepSegmentSummary]
    profile: list[dict]
