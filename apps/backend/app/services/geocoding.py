"""Geocoding via Nominatim (OpenStreetMap)."""

import requests

from app.schemas.roads import GeocodeSuggestion


def search(query: str, limit: int = 5) -> list[GeocodeSuggestion]:
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "limit": limit, "addressdetails": 1},
        headers={"User-Agent": "clmbr/1.0"},
        timeout=10,
    )
    resp.raise_for_status()
    return [
        GeocodeSuggestion(
            name=item.get("display_name", ""),
            lat=float(item["lat"]),
            lon=float(item["lon"]),
        )
        for item in resp.json()
    ]
