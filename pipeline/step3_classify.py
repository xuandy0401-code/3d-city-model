"""
Step 3 – Classify each street-view image to a building using geometric ray-casting.

Algorithm
---------
For every image (lat, lon, heading):
  1. Convert camera position to UTM metres.
  2. Cast a ray in the heading direction (length = MAX_RAY_DISTANCE_M).
  3. Find the nearest OSM building polygon that the ray intersects.
  4. Also record the best-matching facade edge (for texture mapping later).

Outputs:
  output/classification/classification.csv
  output/classification/classification_map.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from shapely.geometry import LineString, Point, MultiPolygon
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    META_DIR, OSM_DIR, CLASS_DIR,
    UTM_CRS, MAX_RAY_DISTANCE_M, MIN_FACADE_DISTANCE_M,
)

WGS_TO_UTM = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _cam_to_utm(lon: float, lat: float):
    return WGS_TO_UTM.transform(lon, lat)


def _ray(cam_x, cam_y, heading_deg, length=MAX_RAY_DISTANCE_M):
    h = np.radians(heading_deg)
    dx, dy = np.sin(h), np.cos(h)   # East, North
    return LineString([(cam_x, cam_y),
                       (cam_x + length * dx, cam_y + length * dy)])


def _nearest_facade_edge(geom, cam_x, cam_y, ray):
    """
    Return (edge_idx, facade_normal_deg, facade_dist_m) for the polygon edge
    that is nearest to the ray intersection point.
    """
    coords = list(geom.exterior.coords)
    cam_pt = Point(cam_x, cam_y)

    best_idx, best_dist, best_mid = 0, float("inf"), None
    n_edges = len(coords) - 1
    for i in range(n_edges):
        a, b = coords[i], coords[i + 1]
        edge = LineString([a, b])
        if edge.intersects(ray):
            mid = edge.centroid
            d = cam_pt.distance(mid)
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_mid = mid

    if best_mid is None:
        # Fallback: closest edge to ray (no direct intersection)
        for i in range(n_edges):
            a, b = coords[i], coords[i + 1]
            mid = LineString([a, b]).centroid
            d = cam_pt.distance(mid)
            if d < best_dist:
                best_dist = d
                best_idx = i

    # Facade outward normal angle (degrees from North, clockwise)
    a, b = coords[best_idx], coords[best_idx + 1]
    ex, ey = b[0] - a[0], b[1] - a[1]   # edge vector
    # Outward normal: rotate 90° clockwise → (ey, -ex)
    nx, ny = ey, -ex
    normal_deg = (np.degrees(np.arctan2(nx, ny)) + 360) % 360

    return best_idx, normal_deg, best_dist


# ── main ──────────────────────────────────────────────────────────────────────

def classify_images(meta_df: pd.DataFrame,
                    buildings_gdf: gpd.GeoDataFrame) -> pd.DataFrame:

    assert buildings_gdf.crs.to_epsg() == 32630, "buildings must be in UTM 32630"

    results = []
    n = len(meta_df)

    for i, row in meta_df.iterrows():
        if i % 50 == 0:
            print(f"  Processing image {i+1}/{n} …")

        cam_x, cam_y = _cam_to_utm(row["lon"], row["lat"])
        cam_pt       = Point(cam_x, cam_y)
        ray          = _ray(cam_x, cam_y, row["heading"])

        best_bid  = None
        best_dist = float("inf")
        best_eidx = None
        best_norm = None

        for _, brow in buildings_gdf.iterrows():
            geom = brow.geometry
            if geom is None or geom.is_empty:
                continue

            # Unwrap MultiPolygon
            polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]

            for poly in polys:
                # Skip if camera is inside the building
                if poly.contains(cam_pt):
                    continue

                if not ray.intersects(poly):
                    continue

                intersection = ray.intersection(poly)
                dist = cam_pt.distance(intersection)

                if dist < MIN_FACADE_DISTANCE_M:
                    continue   # too close — probably camera clipping

                if dist < best_dist:
                    best_dist = dist
                    best_bid  = int(brow["building_id"])
                    eidx, norm, _ = _nearest_facade_edge(poly, cam_x, cam_y, ray)
                    best_eidx = eidx
                    best_norm = norm

        results.append({
            "filename":    row["filename"],
            "location_id": int(row["location_id"]),
            "lat":         row["lat"],
            "lon":         row["lon"],
            "heading":     int(row["heading"]),
            "pitch":       int(row["pitch"]),
            "building_id": best_bid,             # None = no building in this direction
            "distance_m":  round(best_dist, 2) if best_bid is not None else None,
            "facade_edge": best_eidx,
            "facade_normal_deg": round(best_norm, 1) if best_norm is not None else None,
        })

    df = pd.DataFrame(results)
    os.makedirs(CLASS_DIR, exist_ok=True)
    out_csv = os.path.join(CLASS_DIR, "classification.csv")
    df.to_csv(out_csv, index=False)

    classified = df["building_id"].notna().sum()
    print(f"[Step 3] Classified {classified}/{len(df)} images to a building "
          f"({classified/len(df)*100:.1f}%)")
    bids = df["building_id"].dropna().nunique()
    print(f"         {bids} distinct buildings have at least one matching image")
    print(f"         Saved → {out_csv}")

    _plot_classification(df, buildings_gdf)
    return df


def _plot_classification(class_df: pd.DataFrame, buildings_gdf: gpd.GeoDataFrame):
    fig, ax = plt.subplots(figsize=(14, 12))
    gdf_wgs = buildings_gdf.to_crs("EPSG:4326")
    gdf_wgs.plot(ax=ax, facecolor="#d0d0d0", edgecolor="#888888", linewidth=0.7)

    # Count images per building
    counts = class_df.groupby("building_id").size().to_dict()
    gdf_wgs["img_count"] = gdf_wgs["building_id"].map(counts).fillna(0)
    max_c = max(counts.values()) if counts else 1

    gdf_wgs[gdf_wgs["img_count"] > 0].plot(
        ax=ax, column="img_count", cmap="Blues",
        vmin=0, vmax=max_c,
        edgecolor="#333",
        linewidth=0.7,
        legend=True,
        legend_kwds={"label": "# images assigned", "shrink": 0.55},
    )

    # Plot camera positions with heading arrows
    classified = class_df[class_df["building_id"].notna()]
    unclassified = class_df[class_df["building_id"].isna()]

    ax.scatter(classified["lon"], classified["lat"],
               c="blue", s=15, zorder=5, label=f"Classified ({len(classified)})", alpha=0.7)
    ax.scatter(unclassified["lon"], unclassified["lat"],
               c="red", s=15, zorder=5, label=f"No building ({len(unclassified)})", alpha=0.7)

    # Heading arrows (one per unique location, just show h=0 arrow)
    for loc_id, grp in classified.groupby("location_id"):
        row0 = grp.iloc[0]
        h = np.radians(row0["heading"])
        ax.annotate("",
            xy=(row0["lon"] + 0.0002 * np.sin(h),
                row0["lat"] + 0.0002 * np.cos(h) * 0.6),
            xytext=(row0["lon"], row0["lat"]),
            arrowprops=dict(arrowstyle="->", color="navy", lw=0.5),
        )

    ax.set_title("Building Classification via Ray-Casting\n"
                 "IC South Kensington Campus", fontsize=13)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.tight_layout()

    out = os.path.join(CLASS_DIR, "classification_map.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"         Map saved → {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def run():
    meta_df = pd.read_csv(os.path.join(META_DIR, "image_metadata.csv"))
    buildings_gdf = gpd.read_file(os.path.join(OSM_DIR, "buildings.gpkg"))
    return classify_images(meta_df, buildings_gdf)


if __name__ == "__main__":
    run()
