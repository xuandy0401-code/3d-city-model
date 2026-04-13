"""
Step 4 – Generate a LoD1 (block) 3D model from OSM building footprints.

For each building polygon the footprint is extruded vertically to its height.
The model is saved in Wavefront OBJ + MTL format.

Coordinate system in the OBJ file:
  Origin = campus centroid in UTM (metres)
  +X = East,  +Y = Up (height),  +Z = South  (standard OBJ convention)
  (Note: OBJ has Y-up, so we swap N↔Y when writing)

Outputs
-------
  output/models/campus_lod1.obj
  output/models/campus_lod1.mtl
  output/models/building_meta.json   – per-building facade geometry for Step 5
"""

import os, sys, json
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    OSM_DIR, MODEL_DIR, UTM_CRS,
    CAMPUS_CENTER_WGS,
)

WGS_TO_UTM = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)


# ── coordinate helpers ───────────────────────────────────────────────────────

def _campus_origin_utm():
    lon, lat = CAMPUS_CENTER_WGS[1], CAMPUS_CENTER_WGS[0]
    return WGS_TO_UTM.transform(lon, lat)


def _utm_to_local(x, y, ox, oy):
    """Shift from UTM to local metres with origin at campus centre."""
    return x - ox, y - oy


# OBJ uses Y-up.  We store East→+X, Height→+Y, North→−Z
def _to_obj_vertex(east_m, north_m, height_m):
    return (east_m, height_m, -north_m)


# ── polygon helpers ──────────────────────────────────────────────────────────

def _largest_poly(geom):
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def _exterior_coords(poly: Polygon):
    """Return (N, 2) array of exterior ring coords, without closing vertex."""
    coords = np.array(poly.exterior.coords[:-1])
    return coords


# ── OBJ writer ───────────────────────────────────────────────────────────────

class OBJWriter:
    def __init__(self):
        self.vertices  = []   # list of (x, y, z) tuples (OBJ convention)
        self.uvs       = []   # list of (u, v)
        self.objects   = []   # list of dicts with faces
        self._v_off  = 1      # OBJ is 1-indexed
        self._vt_off = 1

    def add_building(self, bid, name, ext_coords, height_m):
        """
        ext_coords : (N, 2) UTM local-origin coordinates (East, North)
        height_m   : building height in metres
        Returns facade_list: list of dicts describing each side face.
        """
        N = len(ext_coords)
        v_base = self._v_off
        vt_base = self._vt_off

        # ── vertices: bottom ring then top ring ──────────────────────────────
        for e, n in ext_coords:
            self.vertices.append(_to_obj_vertex(e, n, 0))          # bottom
        for e, n in ext_coords:
            self.vertices.append(_to_obj_vertex(e, n, height_m))   # top

        # ── UV layout for side faces:  u ∈ [0,1] along edge, v ∈ [0,1] height
        # We use per-face UVs (0,0)-(1,0)-(1,1)-(0,1) for each quad
        # (stored once; each face will reference the same 4 UVs by index)
        uv0_idx = self._vt_off
        self.uvs += [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self._vt_off += 4

        # top/bottom caps get their own set of UVs if needed (skip for now)

        # ── faces ─────────────────────────────────────────────────────────────
        faces  = []
        facade_list = []

        for i in range(N):
            j = (i + 1) % N
            # vertices in OBJ 1-indexed
            bl = v_base + i          # bottom-left
            br = v_base + j          # bottom-right
            tr = v_base + N + j      # top-right
            tl = v_base + N + i      # top-left

            # quad → 2 triangles  (counter-clockwise winding for outward normal)
            mat_name = f"mat_b{bid}_f{i}"
            faces.append((mat_name, [(bl, uv0_idx),   (br, uv0_idx+1),
                                      (tr, uv0_idx+2), (tl, uv0_idx+3)]))

            # Facade metadata (used by Step 5 for texture mapping)
            a = ext_coords[i]
            b = ext_coords[j]
            facade_list.append({
                "facade_idx":   i,
                "building_id":  bid,
                "mat_name":     mat_name,
                # bottom-left and bottom-right corners in local UTM (east, north)
                "corner_a_EN":  [float(a[0]), float(a[1])],
                "corner_b_EN":  [float(b[0]), float(b[1])],
                "height_m":     float(height_m),
                # outward normal direction (degrees from North, clockwise)
                "normal_deg":   _edge_normal_deg(a, b),
            })

        self._v_off += 2 * N

        self.objects.append({
            "name": f"building_{bid}",
            "label": name or f"building_{bid}",
            "faces": faces,
        })
        return facade_list

    def write(self, obj_path: str, mtl_path: str):
        mtl_name = os.path.basename(mtl_path)

        with open(obj_path, "w") as f:
            f.write(f"# IC Campus LoD1 3D Model\n")
            f.write(f"# Coordinate system: East=+X, Height=+Y, South=+Z\n")
            f.write(f"# 1 unit = 1 metre\n\n")
            f.write(f"mtllib {mtl_name}\n\n")

            for vx, vy, vz in self.vertices:
                f.write(f"v {vx:.4f} {vy:.4f} {vz:.4f}\n")
            f.write("\n")

            for u, v in self.uvs:
                f.write(f"vt {u:.6f} {v:.6f}\n")
            f.write("\n")

            for obj in self.objects:
                f.write(f"o {obj['name']}\n")
                for mat_name, corners in obj["faces"]:
                    f.write(f"usemtl {mat_name}\n")
                    # Quad face: v1/vt1 v2/vt2 v3/vt3 v4/vt4
                    face_str = " ".join(f"{vi}/{vti}" for vi, vti in corners)
                    f.write(f"f {face_str}\n")
                f.write("\n")

        # MTL skeleton (textures will be filled in by Step 5)
        with open(mtl_path, "w") as f:
            f.write("# IC Campus Materials (textures added by step5_texture.py)\n\n")
            for obj in self.objects:
                for mat_name, _ in obj["faces"]:
                    f.write(f"newmtl {mat_name}\n")
                    f.write(f"Ka 1.0 1.0 1.0\n")
                    f.write(f"Kd 0.7 0.7 0.7\n")  # grey until textured
                    f.write(f"Ks 0.0 0.0 0.0\n")
                    f.write(f"d 1.0\n\n")


# ── utilities ────────────────────────────────────────────────────────────────

def _edge_normal_deg(a, b):
    """Outward-pointing normal of edge A→B (right-hand side), in degrees from North."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    # Rotate 90° clockwise (outward = right of walking A→B in CCW polygon)
    nx, ny = ey, -ex
    return float((np.degrees(np.arctan2(nx, ny)) + 360) % 360)


# ── main ─────────────────────────────────────────────────────────────────────

def build_model(buildings_gdf: gpd.GeoDataFrame):
    ox, oy = _campus_origin_utm()
    print(f"[Step 4] Campus UTM origin: E={ox:.1f}, N={oy:.1f}")

    writer = OBJWriter()
    all_facades = []

    skipped = 0
    for _, row in buildings_gdf.iterrows():
        bid   = int(row["building_id"])
        name  = str(row.get("name", "")) if "name" in row.index else ""
        h     = float(row["height_m"])
        geom  = _largest_poly(row.geometry)

        ext = _exterior_coords(geom)
        if len(ext) < 3:
            skipped += 1
            continue

        # Convert UTM → local-origin metres
        ext_local = np.column_stack([
            ext[:, 0] - ox,
            ext[:, 1] - oy,
        ])

        facades = writer.add_building(bid, name, ext_local, h)
        for fac in facades:
            fac["building_name"] = name
        all_facades.extend(facades)

    os.makedirs(MODEL_DIR, exist_ok=True)
    obj_path = os.path.join(MODEL_DIR, "campus_lod1.obj")
    mtl_path = os.path.join(MODEL_DIR, "campus_lod1.mtl")
    writer.write(obj_path, mtl_path)

    # Save facade metadata for Step 5
    meta_path = os.path.join(MODEL_DIR, "building_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "utm_origin": {"east": ox, "north": oy},
            "utm_crs": UTM_CRS,
            "facades": all_facades,
        }, f, indent=2)

    print(f"[Step 4] Generated {len(all_facades)} building facades "
          f"({skipped} buildings skipped)")
    print(f"         OBJ  → {obj_path}")
    print(f"         MTL  → {mtl_path}")
    print(f"         Meta → {meta_path}")
    return all_facades


def run():
    buildings_gdf = gpd.read_file(os.path.join(OSM_DIR, "buildings.gpkg"))
    return build_model(buildings_gdf)


if __name__ == "__main__":
    run()
