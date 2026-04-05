from pydantic import BaseModel


class ProfilePoint(BaseModel):
    d: float
    e: float
    s: float


class RoadProperties(BaseModel):
    name: str
    highway_type: str
    length_m: float
    elevation_gain_m: float
    elevation_loss_m: float
    avg_slope_percent: float
    max_slope_percent: float
    start_elevation_m: float
    end_elevation_m: float
    osm_id: int
    incline_tag: str | None = None
    surface: str | None = None
    access: str | None = None


class GeocodeSuggestion(BaseModel):
    name: str
    lat: float
    lon: float
