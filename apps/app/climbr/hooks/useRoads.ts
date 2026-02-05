export interface Road {
  name: string;
  avg_slope_percent: number;
  geometry: [number, number][];
}

import roadsData from "@/assets/steep_running_roads_45.76_4.83.json";

export function useRoads() {
  // Map and type the data once
  return (roadsData as any[]).map((r) => ({
    name: r.name,
    avg_slope_percent: r.avg_slope_percent,
    geometry: r.geometry,
  }));
}
