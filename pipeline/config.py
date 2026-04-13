"""
Configuration for IC South Kensington 3D City Model Pipeline
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR   = os.path.join(BASE_DIR, "IC_campus_streetview")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")

META_DIR    = os.path.join(OUTPUT_DIR, "metadata")
OSM_DIR     = os.path.join(OUTPUT_DIR, "osm_data")
CLASS_DIR   = os.path.join(OUTPUT_DIR, "classification")
MODEL_DIR   = os.path.join(OUTPUT_DIR, "models")
TEX_DIR     = os.path.join(OUTPUT_DIR, "textures")
VIS_DIR     = os.path.join(OUTPUT_DIR, "visualization")

# ── Campus bounding box  (south, west, north, east) in WGS84 ─────────────────
CAMPUS_BBOX = (51.494, -0.181, 51.501, -0.173)   # ~800m × 600m area
CAMPUS_CENTER_WGS = (51.4988, -0.1749)            # approximate campus centroid

# ── Projected CRS  (UTM Zone 30N, suitable for London) ───────────────────────
UTM_CRS = "EPSG:32630"

# ── Street-view camera parameters ────────────────────────────────────────────
SV_IMAGE_WIDTH  = 640          # pixels
SV_IMAGE_HEIGHT = 640          # pixels
SV_FOV_H_DEG    = 90.0         # horizontal field of view
SV_CAMERA_HEIGHT_M = 2.5       # camera height above ground (Google car)
SV_PITCH_DEFAULT   = 20.0      # default pitch (degrees, up from horizontal)

# ── Building geometry defaults ────────────────────────────────────────────────
DEFAULT_BUILDING_HEIGHT_M = 15.0   # used when OSM has no height data
METERS_PER_FLOOR          = 3.5

# ── Classification parameters ─────────────────────────────────────────────────
MAX_RAY_DISTANCE_M   = 200.0   # maximum look-ahead distance for ray casting
MIN_FACADE_DISTANCE_M = 3.0    # ignore buildings closer than this (camera inside)

# ── Texture parameters ────────────────────────────────────────────────────────
TEXTURE_W = 512                # texture image width in pixels
TEXTURE_H = 512                # texture image height in pixels
MAX_TEXTURE_DISTANCE_M = 120.0 # only use images within this distance for textures
