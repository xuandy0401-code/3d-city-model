"""
Step 2 – Fetch building footprints from OpenStreetMap for the IC campus area.

Uses the Overpass API directly via HTTP requests (more robust than osmnx
when behind a VPN or corporate proxy that blocks HTTPS).

Outputs:
  output/osm_data/buildings.gpkg    – GeoPackage with building polygons (UTM)
  output/osm_data/buildings_map.png – overview map
"""

import os
import sys
import json
import warnings
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from shapely.geometry import Polygon, MultiPolygon
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(__file__))
from config import CAMPUS_BBOX, UTM_CRS, OSM_DIR, DEFAULT_BUILDING_HEIGHT_M, METERS_PER_FLOOR

warnings.filterwarnings("ignore")

OVERPASS_ENDPOINTS = [
    "http://overpass-api.de/api/interpreter",          # HTTP first (avoids SSL issues)
    "https://overpass-api.de/api/interpreter",
    "http://overpass.kumi.systems/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


# ── Overpass query ────────────────────────────────────────────────────────────

def _overpass_query(south, west, north, east) -> dict:
    """
    Build and execute an Overpass QL query that fetches all building ways
    and relations with full geometry.
    """
    bbox_str = f"{south},{west},{north},{east}"
    query = f"""
[out:json][timeout:60];
(
  way[building]({bbox_str});
  relation[building]({bbox_str});
);
out geom;
"""
    last_err = None
    for url in OVERPASS_ENDPOINTS:
        try:
            print(f"  Trying: {url}")
            resp = requests.post(
                url,
                data={"data": query},
                timeout=60,
                verify=False,             # tolerate VPN certificate interception
            )
            resp.raise_for_status()
            data = resp.json()
            print(f"  OK – {len(data.get('elements', []))} elements received")
            return data
        except Exception as e:
            last_err = e
            print(f"  Failed: {type(e).__name__}: {str(e)[:100]}")
    raise RuntimeError(f"All Overpass endpoints failed. Last: {last_err}")


# ── Geometry builders ─────────────────────────────────────────────────────────

def _way_to_polygon(el: dict) -> Polygon | None:
    """Build a Shapely Polygon from an Overpass 'way' element with geometry."""
    geom = el.get("geometry", [])
    if len(geom) < 3:
        return None
    coords = [(g["lon"], g["lat"]) for g in geom]
    try:
        return Polygon(coords)
    except Exception:
        return None


def _relation_to_multipolygon(el: dict) -> MultiPolygon | None:
    """Build a MultiPolygon from an Overpass 'relation' element."""
    outer_rings = []
    for member in el.get("members", []):
        if member.get("role") == "outer" and member.get("type") == "way":
            geom = member.get("geometry", [])
            if len(geom) >= 3:
                coords = [(g["lon"], g["lat"]) for g in geom]
                try:
                    outer_rings.append(Polygon(coords))
                except Exception:
                    pass
    if not outer_rings:
        return None
    if len(outer_rings) == 1:
        return outer_rings[0]
    return MultiPolygon(outer_rings)


# ── Tag helpers ───────────────────────────────────────────────────────────────

def _get_height(tags: dict) -> float:
    h = tags.get("height", "")
    if h:
        try:
            return float(str(h).replace("m", "").strip())
        except ValueError:
            pass
    levels = tags.get("building:levels", "")
    if levels:
        try:
            return float(levels) * METERS_PER_FLOOR
        except ValueError:
            pass
    return DEFAULT_BUILDING_HEIGHT_M


# ── Main fetch ────────────────────────────────────────────────────────────────

def fetch_buildings() -> gpd.GeoDataFrame:
    south, west, north, east = CAMPUS_BBOX
    print(f"[Step 2] Fetching OSM buildings for IC campus "
          f"({south},{west}) → ({north},{east}) …")

    data = _overpass_query(south, west, north, east)

    records = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if not tags.get("building"):
            continue

        etype = el.get("type")
        if etype == "way":
            poly = _way_to_polygon(el)
        elif etype == "relation":
            poly = _relation_to_multipolygon(el)
        else:
            continue

        if poly is None or poly.is_empty:
            continue

        records.append({
            "geometry":         poly,
            "osm_id":           el.get("id"),
            "name":             tags.get("name"),
            "building":         tags.get("building"),
            "building:levels":  tags.get("building:levels"),
            "height":           tags.get("height"),
            "height_m":         _get_height(tags),
        })

    if not records:
        raise RuntimeError("No building elements returned from Overpass")

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # Reproject to UTM for metric operations
    gdf_utm = gdf.to_crs(UTM_CRS)

    # Remove tiny slivers (< 20 m²) and invalid geometries
    gdf_utm = gdf_utm[gdf_utm.geometry.is_valid &
                      (gdf_utm.geometry.area > 20)].copy()
    gdf_utm = gdf_utm.reset_index(drop=True)
    gdf_utm["building_id"] = gdf_utm.index

    os.makedirs(OSM_DIR, exist_ok=True)
    out_gpkg = os.path.join(OSM_DIR, "buildings.gpkg")
    gdf_utm.to_file(out_gpkg, driver="GPKG")

    named = gdf_utm["name"].notna().sum()
    print(f"[Step 2] Found {len(gdf_utm)} valid buildings "
          f"({named} with names)")
    print(f"         Height range: {gdf_utm['height_m'].min():.1f}–"
          f"{gdf_utm['height_m'].max():.1f} m")
    print(f"         Saved → {out_gpkg}")

    _plot_buildings(gdf_utm)
    return gdf_utm


# ── Visualisation ─────────────────────────────────────────────────────────────

def _plot_buildings(gdf: gpd.GeoDataFrame):
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_wgs = gdf.to_crs("EPSG:4326")

    vmin = gdf["height_m"].quantile(0.05)
    vmax = gdf["height_m"].quantile(0.95)
    gdf_wgs.plot(
        ax=ax,
        column="height_m",
        cmap="YlOrRd",
        vmin=vmin, vmax=vmax,
        edgecolor="gray",
        linewidth=0.5,
        legend=True,
        legend_kwds={"label": "Estimated height (m)", "shrink": 0.6},
    )

    if "name" in gdf_wgs.columns:
        for _, row in gdf_wgs[gdf_wgs["name"].notna()].iterrows():
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            ax.annotate(str(row["name"]), (cx, cy),
                        fontsize=5.5, ha="center", va="center",
                        color="black", weight="bold")

    bounds = gdf_wgs.total_bounds   # minx, miny, maxx, maxy
    m = 0.001
    ax.set_xlim(bounds[0] - m, bounds[2] + m)
    ax.set_ylim(bounds[1] - m, bounds[3] + m)
    ax.set_title("IC South Kensington – OSM Building Footprints\n"
                 "(coloured by estimated height)", fontsize=13)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.tight_layout()

    out = os.path.join(OSM_DIR, "buildings_map.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"         Map → {out}")


if __name__ == "__main__":
    fetch_buildings()
