import React, { useState, useEffect, JSX, ReactNode } from "react";
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TextInput,
  TouchableOpacity,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import roadsData from "../../assets/steep_running_roads_45.76_4.83.json";
import { getSlopeColor } from "@/lib/utils";
import { Road } from "@/hooks/useRoads";
import { useRouter } from "expo-router";

const DIFFICULTY_SCALE = [
  { level: 5, min: 16, max: 100, label: "Extreme" },
  { level: 4, min: 12, max: 16, label: "Hard" },
  { level: 3, min: 8, max: 12, label: "Challenging" },
  { level: 2, min: 4, max: 8, label: "Moderate" },
  { level: 1, min: 0, max: 4, label: "Easy" },
];

function getDifficulty(slope: number) {
  return (
    DIFFICULTY_SCALE.find((d) => slope >= d.min && slope < d.max) ||
    DIFFICULTY_SCALE[0]
  );
}

function getMaxElevation(road: GroupedRoad) {
  return Math.max(
    ...road.segments.map((s) => s.start_elevation_m ?? s.end_elevation_m ?? 0),
  );
}

interface RoadSegment {
  name: string;
  avg_slope_percent: number;
  length_m: number;
  geometry: number[][];
  [key: string]: any;
}

interface GroupedRoad {
  name: string;
  segments: RoadSegment[];
}

function groupSegmentsByName(segments: RoadSegment[]): GroupedRoad[] {
  const map: { [name: string]: RoadSegment[] } = {};
  segments.forEach((seg) => {
    if (!map[seg.name]) map[seg.name] = [];
    map[seg.name].push(seg);
  });
  return Object.entries(map).map(([name, segments]) => ({ name, segments }));
}

const RoadsScreen = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [groupedRoads, setGroupedRoads] = useState<GroupedRoad[]>([]);
  const [expandedLevel, setExpandedLevel] = useState<number | null>(null);
  const router = useRouter();

  useEffect(() => {
    let segments: RoadSegment[] = [];
    if (Array.isArray(roadsData)) {
      segments = roadsData as RoadSegment[];
    } else if (roadsData && Array.isArray((roadsData as any).segments)) {
      segments = (roadsData as any).segments;
    }
    let grouped = groupSegmentsByName(segments);
    // Sort by greatest avg slope of root segment

    const getMaxSlope = (road: GroupedRoad) =>
      Math.max(...road.segments.map((seg) => seg.avg_slope_percent));

    grouped = grouped.sort((a, b) => getMaxSlope(b) - getMaxSlope(a));
    setGroupedRoads(grouped);
  }, []);

  // Add difficulty info to each road
  const roadsWithDifficulty = groupedRoads.map((road) => {
    const avgSlope =
      road.segments.reduce((sum, seg) => sum + seg.avg_slope_percent, 0) /
      road.segments.length;
    const totalLength = road.segments.reduce(
      (sum, seg) => sum + seg.length_m,
      0,
    );
    const startPoint = {
      latitude: road.segments[0]?.geometry[0][1],
      longitude: road.segments[0]?.geometry[0][0],
    };

    const maxElev = getMaxElevation(road);
    return {
      ...road,
      avgSlope,
      totalLength,
      maxElev,
      difficulty: getDifficulty(avgSlope),
      startPoint,
    };
  });

  // Filter by search
  const filteredRoads = searchQuery
    ? roadsWithDifficulty.filter((road) =>
        road.name.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    : roadsWithDifficulty;

  // Group by difficulty level, sorted by max elevation (highest first)
  const roadsByDifficulty: Record<number, typeof roadsWithDifficulty> = {};
  DIFFICULTY_SCALE.forEach((d) => {
    roadsByDifficulty[d.level] = [];
  });
  filteredRoads.forEach((road) => {
    roadsByDifficulty[road.difficulty.level].push(road);
  });
  // Sort each group by elevation descending
  Object.keys(roadsByDifficulty).forEach((lvl) => {
    roadsByDifficulty[Number(lvl)].sort((a, b) => b.maxElev - a.maxElev);
  });

  const handleSearch = (text: string) => {
    setSearchQuery(text);
  };

  const toggleStack = (level: number) => {
    setExpandedLevel(expandedLevel === level ? null : level);
  };

  return (
    <View style={styles.container}>
      <View
        style={[
          styles.scrollView,
          { flex: 1, flexDirection: "column", rowGap: 12 },
        ]}
      >
        <TextInput
          style={styles.searchInput}
          placeholder="Search roads..."
          placeholderTextColor="#888"
          onChangeText={handleSearch}
        />
        <ScrollView showsVerticalScrollIndicator={false}>
          {DIFFICULTY_SCALE.map((diff) => {
            const roads = roadsByDifficulty[diff.level];
            if (!roads.length) return null;
            const isOpen = expandedLevel === diff.level;
            const stackColor = getSlopeColor((diff.min + diff.max) / 2);
            const maxStackCards = 2;

            return (
              <View key={diff.level}>
                {/* Difficulty Label */}
                <TouchableOpacity onPress={() => toggleStack(diff.level)}>
                  <View style={styles.difficultyLabel}>
                    <Ionicons
                      name={"flame"}
                      size={18}
                      color={stackColor}
                      style={{ marginRight: 6 }}
                    />
                    <Text
                      style={[styles.difficultyText, { color: stackColor }]}
                    >
                      {diff.label}
                    </Text>
                    <Text style={styles.difficultyCount}>({roads.length})</Text>
                  </View>
                </TouchableOpacity>

                {/* Stacked or Expanded */}
                {!isOpen ? (
                  <TouchableOpacity
                    activeOpacity={0.9}
                    onPress={() => toggleStack(diff.level)}
                  >
                    <View
                      style={{
                        height:
                          90 + Math.min(roads.length - 1, maxStackCards) * 6,
                      }}
                    >
                      {roads
                        .slice(0, maxStackCards + 1)
                        .reverse()
                        .map((road, idx, arr) => {
                          const offset = (arr.length - 1 - idx) * 6;

                          return (
                            <View
                              key={road.name}
                              style={[
                                styles.card,
                                styles.stackedCard,
                                {
                                  position: "absolute",
                                  top: offset,
                                  left: offset,
                                  right: -offset,
                                  zIndex: idx,
                                  opacity: 1 - (arr.length - 1 - idx) * 0.12,
                                },
                              ]}
                            >
                              <RoadCard road={road} top={true} />
                            </View>
                          );
                        })}
                    </View>
                  </TouchableOpacity>
                ) : (
                  <View>
                    {roads.map((road) => {
                      return (
                        <TouchableOpacity
                          key={road.name}
                          activeOpacity={0.95}
                          onPress={() =>
                            router.push(
                              `/map?mLatitude=${road.startPoint.latitude}&mLongitude=${road.startPoint.longitude}`,
                            )
                          }
                        >
                          <RoadCard road={road} />
                        </TouchableOpacity>
                      );
                    })}
                  </View>
                )}
              </View>
            );
          })}
          <View style={{ height: 32 }} />
        </ScrollView>
      </View>
    </View>
  );
};

interface RoadWithDifficulty extends GroupedRoad {
  avgSlope: number;
  totalLength: number;
  maxElev: number;
  difficulty: { level: number; min: number; max: number; label: string };
  startPoint: { latitude: number; longitude: number };
}

const RoadCard = ({
  road,
  top = false,
}: {
  road: RoadWithDifficulty;
  top?: boolean;
}) => {
  const parent = (children: ReactNode) =>
    top ? children : <View style={styles.card}>{children}</View>;

  return parent(
    <View style={styles.cardHeader}>
      <View style={styles.info}>
        <Text style={styles.name}>{road.name}</Text>
        <Text style={styles.details}>
          <Ionicons name="trending-up" size={16} color="#888" />
          {"  "}Slope:{" "}
          <Text style={styles.bold}>{road.avgSlope.toFixed(1)}%</Text>
          {"   "}
          <Ionicons name="resize" size={16} color="#888" />
          {"  "}Length: <Text style={styles.bold}>{road.totalLength}m</Text>
        </Text>
      </View>
    </View>,
  );
};

const styles = StyleSheet.create({
  container: {
    marginTop: 10,
    flex: 1,
    backgroundColor: "#f4f6fa",
    padding: 12,
  },
  scrollView: {
    marginTop: 30,
  },
  card: {
    backgroundColor: "#fff",
    borderRadius: 16,
    marginBottom: 18,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    elevation: 2,
    paddingVertical: 10,
    paddingHorizontal: 14,
  },
  cardHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 2,
  },
  iconCircle: {
    width: 38,
    height: 38,
    borderRadius: 19,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12,
  },
  info: {
    flex: 1,
    marginRight: 10,
  },
  name: {
    fontSize: 17,
    fontWeight: "bold",
    color: "#222",
    marginBottom: 2,
  },
  details: {
    fontSize: 14,
    color: "#666",
    marginTop: 2,
    flexDirection: "row",
    alignItems: "center",
  },
  bold: {
    fontWeight: "bold",
    color: "#222",
  },
  searchInput: {
    height: 50,
    backgroundColor: "#fff",
    color: "black",
    borderRadius: 16,
    fontSize: 14,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    paddingLeft: 12,
  },
  stackedCard: {
    marginBottom: 0,
  },
  difficultyLabel: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
    marginLeft: 4,
  },
  difficultyText: {
    fontWeight: "bold",
    fontSize: 15,
  },
  difficultyCount: {
    color: "#888",
    fontSize: 14,
    marginLeft: 6,
  },
});

export default RoadsScreen;
