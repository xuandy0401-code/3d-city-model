"""
Step 5 – Perspective-correct texture mapping of street-view images onto building facades.

For each building facade:
  1. Find all classified street-view images that look at this facade.
  2. Score and pick the best image (most frontal, reasonable distance).
  3. Generate a texture image (TEXTURE_W × TEXTURE_H) by projecting
     a regular grid of facade points through the camera model and
     sampling the street-view image (bilinear interpolation).
  4. Write the texture PNG and update the MTL file.

Camera model (Google Street View)
----------------------------------
  heading : degrees from North, clockwise  (0=N, 90=E, 180=S, 270=W)
  pitch   : degrees upward from horizontal (p=20 means tilted 20° up)
  FOV     : 90° horizontal  →  focal length f = W / (2·tan(45°)) = W/2

Coordinate frame for projection
---------------------------------
  World  ENU : +E (east), +N (north), +U (up)  — metres, local UTM origin
  Camera     : +X (right), +Y (up), +Z (forward / into scene)

Outputs
-------
  output/textures/b{bid}_f{fid}.png   — one texture per facade
  output/models/campus_lod1.mtl       — updated with texture references
  output/classification/texture_log.csv  — which image → which facade
"""

import os, sys, json
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    IMAGE_DIR, MODEL_DIR, CLASS_DIR, TEX_DIR,
    SV_IMAGE_WIDTH, SV_IMAGE_HEIGHT, SV_FOV_H_DEG,
    SV_CAMERA_HEIGHT_M, TEXTURE_W, TEXTURE_H,
    MAX_TEXTURE_DISTANCE_M,
    UTM_CRS,
)
from pyproj import Transformer

WGS_TO_UTM = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)

# focal length in pixels for 90° HFOV, 640 px wide
FOCAL_PX = (SV_IMAGE_WIDTH / 2) / np.tan(np.radians(SV_FOV_H_DEG / 2))


# ─────────────────────────────────────────────────────────────────────────────
# Camera model
# ─────────────────────────────────────────────────────────────────────────────

def _camera_axes(heading_deg: float, pitch_deg: float):
    """
    Return (right, up, forward) unit vectors in ENU coordinates.

    ENU: east=+x, north=+y, up=+z
    Heading: from North clockwise  (0=N, 90=E, 180=S, 270=W)
    Pitch:   upward from horizontal (positive = camera tilted up)
    """
    h = np.radians(heading_deg)
    p = np.radians(pitch_deg)

    forward = np.array([
        np.sin(h) * np.cos(p),   # east
        np.cos(h) * np.cos(p),   # north
        np.sin(p),                # up
    ])
    world_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, world_up)
    norm = np.linalg.norm(right)
    if norm < 1e-9:                       # degenerate (camera pointing straight up/down)
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= norm

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    return right, up, forward / np.linalg.norm(forward)


def project_point(world_pt_enu: np.ndarray,
                  cam_enu: np.ndarray,
                  heading_deg: float,
                  pitch_deg: float):
    """
    Project one 3-D ENU point through the camera.
    Returns (u, v) pixel coords or None if behind camera / outside image.
    """
    right, up, forward = _camera_axes(heading_deg, pitch_deg)
    rel = world_pt_enu - cam_enu

    z_cam = float(np.dot(rel, forward))   # depth along optical axis
    if z_cam <= 0.1:
        return None                        # behind or too close

    x_cam = float(np.dot(rel, right))
    y_cam = float(np.dot(rel, up))

    u = SV_IMAGE_WIDTH  / 2 + FOCAL_PX * x_cam / z_cam
    v = SV_IMAGE_HEIGHT / 2 - FOCAL_PX * y_cam / z_cam   # v flipped (image top-down)

    # Allow small margin outside image
    margin = 10
    if not (-margin <= u < SV_IMAGE_WIDTH  + margin and
            -margin <= v < SV_IMAGE_HEIGHT + margin):
        return None

    return float(u), float(v)


# ─────────────────────────────────────────────────────────────────────────────
# Image sampling (bilinear)
# ─────────────────────────────────────────────────────────────────────────────

def _bilinear_sample(img_arr: np.ndarray, u: float, v: float) -> np.ndarray:
    """Bilinear sample of img_arr (H×W×C) at float (u, v). Returns RGB pixel."""
    H, W = img_arr.shape[:2]
    u = float(np.clip(u, 0, W - 1.001))
    v = float(np.clip(v, 0, H - 1.001))

    x0, y0 = int(u), int(v)
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    fx, fy = u - x0, v - y0

    c00 = img_arr[y0, x0].astype(np.float32)
    c10 = img_arr[y0, x1].astype(np.float32)
    c01 = img_arr[y1, x0].astype(np.float32)
    c11 = img_arr[y1, x1].astype(np.float32)

    return ((c00 * (1 - fx) + c10 * fx) * (1 - fy) +
            (c01 * (1 - fx) + c11 * fx) * fy).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Facade scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_image_for_facade(row, facade: dict, cam_enu: np.ndarray) -> float:
    """
    Score how suitable a street-view image is for texturing a given facade.

    Score = cosine_similarity × (1 / distance_factor)
    Higher is better.  Returns -inf if unusable.
    """
    dist = float(row["distance_m"]) if pd.notna(row["distance_m"]) else 9999
    if dist > MAX_TEXTURE_DISTANCE_M:
        return -np.inf

    # Angle between camera heading and facade outward normal
    # (best when camera looks directly at the facade → angle ≈ 180°, i.e. facing it)
    heading = float(row["heading"])
    normal  = float(facade["normal_deg"])
    angle_diff = abs(((heading - normal + 180) % 360) - 180)
    if angle_diff > 80:       # camera is nearly parallel to facade → skip
        return -np.inf

    # Check facade midpoint visibility
    a  = np.array(facade["corner_a_EN"])
    b  = np.array(facade["corner_b_EN"])
    h  = facade["height_m"]
    mid_xy  = (a + b) / 2
    mid_enu = np.array([mid_xy[0], mid_xy[1], h / 2])

    uv = project_point(mid_enu, cam_enu, float(row["heading"]), float(row["pitch"]))
    if uv is None:
        return -np.inf

    frontality = np.cos(np.radians(angle_diff))  # 1 = perfectly frontal
    dist_factor = np.sqrt(max(dist, 5))
    return frontality / dist_factor


# ─────────────────────────────────────────────────────────────────────────────
# Texture generation  (fully vectorised with numpy – fast)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_texture_vectorized(facade: dict, img_row: pd.Series,
                                  img_cache: dict):
    """
    Produce a TEXTURE_H × TEXTURE_W × 3 uint8 array by back-projecting all
    texture pixels through the camera in one vectorised numpy pass.

    Texture UV convention:
      tx=0       → corner A of the facade edge
      tx=TEX_W-1 → corner B
      ty=0       → top of building (height=H)
      ty=TEX_H-1 → ground (height=0)
    """
    fp = img_row["filepath"]
    if fp not in img_cache:
        img_cache[fp] = np.array(Image.open(fp).convert("RGB"))
    src = img_cache[fp]               # (SV_H, SV_W, 3)

    a        = np.array(facade["corner_a_EN"], dtype=np.float64)  # (east, north)
    b        = np.array(facade["corner_b_EN"], dtype=np.float64)
    height_m = float(facade["height_m"])

    cam_e, cam_n = WGS_TO_UTM.transform(float(img_row["lon"]),
                                         float(img_row["lat"]))
    cam_enu  = np.array([cam_e, cam_n, SV_CAMERA_HEIGHT_M])
    heading  = float(img_row["heading"])
    pitch    = float(img_row["pitch"])
    right, up, forward = _camera_axes(heading, pitch)

    # ── build grid of 3-D facade points: shape (TEX_H, TEX_W, 3) ────────────
    # tx_frac: fraction along A→B edge
    tx_frac = (np.arange(TEXTURE_W, dtype=np.float64) + 0.5) / TEXTURE_W  # (W,)
    # ty_frac: fraction of height, 0=top, 1=bottom → height = (1-ty_frac)*H
    ty_frac = (np.arange(TEXTURE_H, dtype=np.float64) + 0.5) / TEXTURE_H  # (H,)
    z_vals  = (1.0 - ty_frac) * height_m                                   # (H,)

    # Horizontal positions on facade edge: (W, 2) east/north
    edge_xy = a[np.newaxis, :] + tx_frac[:, np.newaxis] * (b - a)[np.newaxis, :]  # (W,2)

    # Expand to full grid: east (H,W), north (H,W), up (H,W)
    east_grid  = np.broadcast_to(edge_xy[:, 0][np.newaxis, :], (TEXTURE_H, TEXTURE_W))
    north_grid = np.broadcast_to(edge_xy[:, 1][np.newaxis, :], (TEXTURE_H, TEXTURE_W))
    z_grid     = np.broadcast_to(z_vals[:, np.newaxis],         (TEXTURE_H, TEXTURE_W))

    # Relative vector from camera to each facade point: (H, W, 3)
    rel_e = east_grid  - cam_enu[0]
    rel_n = north_grid - cam_enu[1]
    rel_z = z_grid     - cam_enu[2]

    # Project onto camera axes
    # right/up/forward are (3,) ENU vectors
    x_cam = rel_e * right[0] + rel_n * right[1] + rel_z * right[2]   # (H,W)
    y_cam = rel_e * up[0]    + rel_n * up[1]    + rel_z * up[2]      # (H,W)
    z_cam = rel_e * forward[0] + rel_n * forward[1] + rel_z * forward[2]  # (H,W)

    valid_mask = z_cam > 0.1   # points in front of camera

    # Perspective projection → image coords
    with np.errstate(divide="ignore", invalid="ignore"):
        u_img = np.where(valid_mask,
                         SV_IMAGE_WIDTH  / 2 + FOCAL_PX * x_cam / z_cam,
                         -1.0)
        v_img = np.where(valid_mask,
                         SV_IMAGE_HEIGHT / 2 - FOCAL_PX * y_cam / z_cam,
                         -1.0)

    # Clip to image bounds
    in_bounds = (valid_mask &
                 (u_img >= 0) & (u_img < SV_IMAGE_WIDTH  - 1) &
                 (v_img >= 0) & (v_img < SV_IMAGE_HEIGHT - 1))

    # ── bilinear sample (vectorised) ─────────────────────────────────────────
    texture = np.zeros((TEXTURE_H, TEXTURE_W, 3), dtype=np.float32)

    if in_bounds.any():
        u_c = np.clip(u_img[in_bounds], 0, SV_IMAGE_WIDTH  - 1.001)
        v_c = np.clip(v_img[in_bounds], 0, SV_IMAGE_HEIGHT - 1.001)

        x0 = u_c.astype(np.int32);  x1 = x0 + 1
        y0 = v_c.astype(np.int32);  y1 = y0 + 1
        fx = (u_c - x0).astype(np.float32)[:, np.newaxis]   # (N,1)
        fy = (v_c - y0).astype(np.float32)[:, np.newaxis]

        x1 = np.minimum(x1, SV_IMAGE_WIDTH  - 1)
        y1 = np.minimum(y1, SV_IMAGE_HEIGHT - 1)

        # bilinear: (N,3)
        c00 = src[y0, x0].astype(np.float32)
        c10 = src[y0, x1].astype(np.float32)
        c01 = src[y1, x0].astype(np.float32)
        c11 = src[y1, x1].astype(np.float32)
        sampled = (c00*(1-fx) + c10*fx)*(1-fy) + (c01*(1-fx) + c11*fx)*fy

        texture[in_bounds] = sampled

    coverage = float(in_bounds.sum()) / (TEXTURE_W * TEXTURE_H)
    return texture.astype(np.uint8), coverage


# ─────────────────────────────────────────────────────────────────────────────
# MTL update
# ─────────────────────────────────────────────────────────────────────────────

def _update_mtl(mtl_path: str, texture_assignments: dict):
    """
    Rewrite the MTL file so each material points to its texture PNG.
    texture_assignments: {mat_name: rel_tex_path}
    """
    with open(mtl_path) as f:
        lines = f.readlines()

    out = []
    current_mat = None
    for line in lines:
        strip = line.strip()
        if strip.startswith("newmtl "):
            current_mat = strip.split()[1]
            out.append(line)
        elif strip.startswith("Kd") and current_mat in texture_assignments:
            out.append("Kd 1.0 1.0 1.0\n")
        else:
            out.append(line)
            if strip.startswith("Kd") and current_mat in texture_assignments:
                pass
        # Append map_Kd after the material block's Kd line
        if strip.startswith("Kd") and current_mat in texture_assignments:
            out.append(f"map_Kd {texture_assignments[current_mat]}\n")

    with open(mtl_path, "w") as f:
        f.writelines(out)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def apply_textures(class_df: pd.DataFrame, facades: list):
    os.makedirs(TEX_DIR, exist_ok=True)

    # Add filepath column if missing (reconstruct from filename)
    if "filepath" not in class_df.columns:
        class_df = class_df.copy()
        class_df["filepath"] = class_df["filename"].apply(
            lambda fn: os.path.join(IMAGE_DIR, fn))

    # Pre-compute cam_enu for classified images
    classified = class_df[class_df["building_id"].notna()].copy()
    classified["cam_e"] = classified.apply(
        lambda r: WGS_TO_UTM.transform(r["lon"], r["lat"])[0], axis=1)
    classified["cam_n"] = classified.apply(
        lambda r: WGS_TO_UTM.transform(r["lon"], r["lat"])[1], axis=1)

    img_cache = {}
    texture_assignments = {}   # mat_name → relative tex path
    log_rows = []

    mtl_path = os.path.join(MODEL_DIR, "campus_lod1.mtl")

    n_facades = len(facades)
    textured, skipped = 0, 0

    for fi, facade in enumerate(facades):
        if fi % 100 == 0:
            print(f"  Facade {fi+1}/{n_facades} …")

        bid = facade["building_id"]
        fid = facade["facade_idx"]
        mat = facade["mat_name"]

        # Candidate images: classified to this building
        candidates = classified[classified["building_id"] == bid].copy()
        if candidates.empty:
            skipped += 1
            continue

        # Read OSM UTM origin to convert facade local→ENU
        # Facades already stored as local-UTM metres
        # (No offset needed — corner_a/b_EN are already in local-UTM coordinates
        #  matching the cam_enu from WGS_TO_UTM, since we share the same UTM CRS)

        # Actually the facade corners are in LOCAL coordinates (shifted by campus origin).
        # We need absolute UTM coords for the projection.
        # Load meta to get origin.
        pass   # handled below when loading metadata

        # Score candidates
        scores = []
        for _, img_row in candidates.iterrows():
            cam_enu = np.array([img_row["cam_e"], img_row["cam_n"],
                                SV_CAMERA_HEIGHT_M])
            s = _score_image_for_facade(img_row, facade, cam_enu)
            scores.append(s)

        best_i = int(np.argmax(scores))
        best_score = scores[best_i]
        if best_score == -np.inf:
            skipped += 1
            continue

        best_row = candidates.iloc[best_i]
        tex, coverage = _generate_texture_vectorized(facade, best_row, img_cache)

        tex_fname = f"b{bid}_f{fid}.png"
        tex_path  = os.path.join(TEX_DIR, tex_fname)
        Image.fromarray(tex).save(tex_path)

        # MTL uses relative path from models/  to  textures/
        rel_path = os.path.join("..", "textures", tex_fname)
        texture_assignments[mat] = rel_path

        log_rows.append({
            "building_id": bid,
            "facade_idx":  fid,
            "mat_name":    mat,
            "src_image":   best_row["filename"],
            "score":       round(best_score, 4),
            "coverage":    round(coverage, 3),
            "tex_file":    tex_path,
        })
        textured += 1

    # Update MTL
    _update_mtl(mtl_path, texture_assignments)

    # Save log
    log_df = pd.DataFrame(log_rows)
    log_path = os.path.join(CLASS_DIR, "texture_log.csv")
    log_df.to_csv(log_path, index=False)

    print(f"[Step 5] Textured {textured} facades, skipped {skipped} "
          f"(no suitable image)")
    print(f"         Textures → {TEX_DIR}")
    print(f"         Log      → {log_path}")
    return log_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run():
    class_df = pd.read_csv(os.path.join(CLASS_DIR, "classification.csv"))
    meta_path = os.path.join(MODEL_DIR, "building_meta.json")

    with open(meta_path) as f:
        meta = json.load(f)

    utm_origin = meta["utm_origin"]
    ox, oy = utm_origin["east"], utm_origin["north"]
    facades = meta["facades"]

    # Adjust facade corner coords from local-origin → absolute UTM
    for fac in facades:
        a = fac["corner_a_EN"]
        b = fac["corner_b_EN"]
        fac["corner_a_EN"] = [a[0] + ox, a[1] + oy]
        fac["corner_b_EN"] = [b[0] + ox, b[1] + oy]

    return apply_textures(class_df, facades)


if __name__ == "__main__":
    run()
