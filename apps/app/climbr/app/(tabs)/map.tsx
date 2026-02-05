import { StyleSheet, View, Text, Platform } from "react-native";
import { useState, useEffect } from "react";
import MapView, { Polyline, Marker } from "react-native-maps";
import * as Location from "expo-location";
import { useRoads } from "@/hooks/useRoads";
import { getSlopeColor } from "@/lib/utils";
import { useLocalSearchParams } from "expo-router";

export default function Map() {
  const {mLatitude, mLongitude} = useLocalSearchParams();

  const [mapReady, setMapReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [location, setLocation] = useState<Location.LocationObject | null>(
    null,
  );
  const [region, setRegion] = useState({
    latitude: 45.76,
    longitude: 4.83,
    latitudeDelta: 0.0922,
    longitudeDelta: 0.0421,
  });
  const [permissionStatus, setPermissionStatus] = useState<
    "undetermined" | "granted" | "denied"
  >("undetermined");

  const roads = useRoads();
  useEffect(() => {
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();
      setPermissionStatus(status);
      if (status !== "granted") {
        setError("Permission to access location was denied");
        return;
      }
      let loc = await Location.getCurrentPositionAsync({});
      setLocation(loc);
      setRegion({
        latitude: loc.coords.latitude,
        longitude: loc.coords.longitude,
        latitudeDelta: 0.01,
        longitudeDelta: 0.01,
      });
    })();
  }, []);

  useEffect(() => {
    if (mLatitude && mLongitude) {
      console.log("Setting region to:", mLatitude, mLongitude);
      setRegion({
        latitude: Number(mLatitude),
        longitude: Number(mLongitude),
        latitudeDelta: 0.01,
        longitudeDelta: 0.01,
      });
    }
  }, [mLatitude, mLongitude]);

  return (
    <View style={styles.container}>
      <>
        {error && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}
        <MapView
          style={styles.map}
          region={region}
          showsUserLocation={permissionStatus === "granted"}
          onMapReady={() => {
            setMapReady(true);
          }}
        >
          {roads.map((road, idx) => (
            <Polyline
              key={idx}
              coordinates={road.geometry.map(([lng, lat]) => ({
                latitude: lat,
                longitude: lng,
              }))}
              strokeWidth={5}
              strokeColor={getSlopeColor(road.avg_slope_percent)}
            />
          ))}
          {mLatitude && mLongitude && (
            <Marker
              key={"searchLocation"}
              coordinate={{
                latitude: Number(mLatitude),
                longitude: Number(mLongitude),
              }}
            />
          )}
        </MapView>
        {!mapReady && !error && (
          <View style={styles.loadingContainer}>
            <Text style={styles.loadingText}>Loading map...</Text>
          </View>
        )}
      </>
    </View>
  );
}

const styles = StyleSheet.create({
  headerImage: {
    color: "#808080",
    bottom: -90,
    left: -35,
    position: "absolute",
  },
  titleContainer: {
    flexDirection: "row",
    gap: 8,
  },
  container: {
    backgroundColor: "#f0f0f0", // Add background color to see container
    borderRadius: 8,
  },
  map: {
    width: "100%",
    height: "100%",
    borderRadius: 8,
  },
  errorContainer: {
    position: "absolute",
    top: 10,
    left: 10,
    right: 10,
    backgroundColor: "red",
    padding: 10,
    borderRadius: 5,
    zIndex: 1000,
  },
  errorText: {
    color: "white",
    textAlign: "center",
  },
  loadingContainer: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(240, 240, 240, 0.8)",
  },
  loadingText: {
    fontSize: 16,
    color: "#666",
  },
});
