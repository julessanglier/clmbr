from typing import Protocol

from app.services.road_finder import SteepRoad


class RoadFinderModel(Protocol):
    def find(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        min_avg_slope: float,
    ) -> list[SteepRoad]: ...
