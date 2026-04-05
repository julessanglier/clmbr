"""v3 — SRTM local DEM (existing pipeline, wrapped)."""

from app.services.road_finder import SteepRoad, find_steep_roads


class V3SrtmFinder:
    def find(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        min_avg_slope: float,
    ) -> list[SteepRoad]:
        return find_steep_roads(
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            min_avg_slope=min_avg_slope,
        )
