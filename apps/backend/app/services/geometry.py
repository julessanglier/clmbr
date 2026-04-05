"""Geometry helpers for coordinate math."""

import math

import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def line_length_meters(coords: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        total += haversine_distance(lat1, lon1, lat2, lon2)
    return total


def interpolate_line(coords: list[tuple[float, float]], step_m: float) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return coords

    distances = [0.0]
    for i in range(1, len(coords)):
        d = haversine_distance(coords[i - 1][1], coords[i - 1][0], coords[i][1], coords[i][0])
        distances.append(distances[-1] + d)

    total_length = distances[-1]
    if total_length < step_m:
        return coords

    result = []
    target_distances = np.arange(0, total_length, step_m)
    if target_distances[-1] < total_length:
        target_distances = np.append(target_distances, total_length)

    for target_d in target_distances:
        for i in range(len(distances) - 1):
            if distances[i] <= target_d <= distances[i + 1]:
                seg_length = distances[i + 1] - distances[i]
                t = (target_d - distances[i]) / seg_length if seg_length > 0 else 0
                lon = coords[i][0] + t * (coords[i + 1][0] - coords[i][0])
                lat = coords[i][1] + t * (coords[i + 1][1] - coords[i][1])
                result.append((lon, lat))
                break

    return result
