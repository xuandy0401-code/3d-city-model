"""
Generate building_data.json for the interactive website.

Includes:
  - Building metadata (name, height)
  - Classified street-view image list
  - Building footprint polygon in local metres  ← used by Three.js ExtrudeGeometry
"""

import os, json
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, LinearRing

BASE = os.path.dirname(os.path.abspath(__file__))


def _ensure_ccw(coords: list) -> list:
    """Force CCW winding (Three.js Shape convention for outer boundary)."""
    ring = LinearRing(coords)
    return coords if ring.is_ccw else coords[::-1]


def _ensure_cw(coords: list) -> list:
    """Force CW winding (Three.js Path convention for holes)."""
    ring = LinearRing(coords)
    return coords if not ring.is_ccw else coords[::-1]


def _ring_to_local(ring, ox: float, oy: float) -> list:
    """Convert a Shapely ring to [[local_east, local_north], …]."""
    return [[round(x - ox, 2), round(y - oy, 2)]
            for x, y in list(ring.coords)[:-1]]   # drop closing duplicate


def poly_to_local(poly, ox: float, oy: float):
    """Return (exterior_coords, holes_list_or_None) in local metres."""
    ext = _ensure_ccw(_ring_to_local(poly.exterior, ox, oy))
    holes = [_ensure_cw(_ring_to_local(ring, ox, oy))
             for ring in poly.interiors
             if len(ring.coords) >= 4]
    return ext, (holes if holes else None)


def main():
    # ── Load UTM origin from pipeline output ──────────────────────────────────
    meta_path = os.path.join(BASE, "output", "models", "building_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    ox = meta["utm_origin"]["east"]
    oy = meta["utm_origin"]["north"]
    print(f"UTM origin: E={ox:.1f}, N={oy:.1f}")

    # ── Load data ─────────────────────────────────────────────────────────────
    class_df = pd.read_csv(
        os.path.join(BASE, "output", "classification", "classification.csv"))
    buildings_gdf = gpd.read_file(
        os.path.join(BASE, "output", "osm_data", "buildings.gpkg"))

    # ── Build per-building image lists ────────────────────────────────────────
    classified = class_df[class_df["building_id"].notna()].copy()
    by_building: dict[int, list] = {}
    for _, row in classified.iterrows():
        bid = int(row["building_id"])
        img: dict = {
            "filename": row["filename"],
            "heading":  int(row["heading"]),
            "pitch":    int(row["pitch"]),
        }
        if pd.notna(row.get("distance_m")):
            img["distance_m"] = round(float(row["distance_m"]), 1)
        by_building.setdefault(bid, []).append(img)

    # ── Assemble output (all 440 buildings) ───────────────────────────────────
    out: dict = {}
    skipped = 0
    for _, brow in buildings_gdf.iterrows():
        bid    = int(brow["building_id"])
        name   = brow.get("name")
        height = float(brow["height_m"])

        geom = brow.geometry
        if geom is None or geom.is_empty:
            skipped += 1
            continue

        # Use largest polygon for MultiPolygon buildings
        poly = (max(geom.geoms, key=lambda g: g.area)
                if isinstance(geom, MultiPolygon) else geom)

        if len(poly.exterior.coords) < 4:   # degenerate
            skipped += 1
            continue

        ext, holes = poly_to_local(poly, ox, oy)

        entry: dict = {
            "id":       bid,
            "name":     str(name) if pd.notna(name) else f"Building {bid}",
            "has_name": bool(pd.notna(name)),
            "height_m": round(height, 1),
            "images":   by_building.get(bid, []),
            "footprint": ext,
        }
        if holes:
            entry["holes"] = holes

        out[str(bid)] = entry

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.join(BASE, "website"), exist_ok=True)
    out_path = os.path.join(BASE, "website", "building_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    n_img  = sum(1 for v in out.values() if v["images"])
    total  = sum(len(v["images"]) for v in out.values())
    fp_sizes = [len(v["footprint"]) for v in out.values()]
    print(f"✓ {out_path}")
    print(f"  Buildings: {len(out)} exported, {skipped} skipped")
    print(f"  With images: {n_img} buildings, {total} total assignments")
    print(f"  Footprint vertices: min={min(fp_sizes)}, "
          f"max={max(fp_sizes)}, avg={sum(fp_sizes)/len(fp_sizes):.1f}")


if __name__ == "__main__":
    main()
