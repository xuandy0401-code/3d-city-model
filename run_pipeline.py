"""
IC South Kensington 3D City Model – Full Pipeline Runner
=========================================================

Stages
------
  1. Parse street-view image metadata from filenames
  2. Fetch building footprints + heights from OpenStreetMap
  3. Classify each image to a building (geometric ray-casting)
  4. Generate LoD1 3D model (extruded footprints) as OBJ + MTL
  5. Apply perspective-correct textures from street-view images
  6. Visualise results (maps, 3-D preview, gallery, report)

Usage
-----
  python run_pipeline.py              # run all steps
  python run_pipeline.py --steps 1 2  # run only steps 1 and 2
  python run_pipeline.py --steps 3    # re-run classification only

Output
------
  output/
    metadata/        image_metadata.csv
    osm_data/        buildings.gpkg, buildings_map.png
    classification/  classification.csv, classification_map.png, texture_log.csv
    models/          campus_lod1.obj, campus_lod1.mtl, building_meta.json
    textures/        b{bid}_f{fid}.png  (one texture per facade)
    visualization/   campus_3d.png, building_gallery.png, coverage_map.png,
                     texture_preview.png, summary.txt
"""

import sys
import time
import argparse
import os

# ── allow importing pipeline modules ─────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))


def run_step(n: int, label: str, fn, *args, **kwargs):
    print(f"\n{'─'*60}")
    print(f"  STEP {n}  |  {label}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    print(f"  ✓ Done in {time.time()-t0:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="IC 3D City Model Pipeline")
    parser.add_argument("--steps", nargs="*", type=int, default=list(range(1, 7)),
                        help="Steps to run (1-6, default: all)")
    args = parser.parse_args()
    steps = set(args.steps)

    print("=" * 60)
    print("  IC South Kensington – 3D City Model Pipeline")
    print("=" * 60)
    print(f"  Running steps: {sorted(steps)}")

    results = {}

    # ── Step 1: Parse metadata ────────────────────────────────────────────────
    if 1 in steps:
        from step1_metadata import parse_metadata
        results["meta_df"] = run_step(1, "Parse image metadata", parse_metadata)

    # ── Step 2: Fetch OSM buildings ───────────────────────────────────────────
    if 2 in steps:
        from step2_osm import fetch_buildings
        results["buildings_gdf"] = run_step(2, "Fetch OSM building footprints",
                                            fetch_buildings)

    # ── Step 3: Classify images to buildings ──────────────────────────────────
    if 3 in steps:
        import pandas as pd
        import geopandas as gpd
        from config import META_DIR, OSM_DIR

        if "meta_df" not in results:
            results["meta_df"] = pd.read_csv(
                os.path.join(META_DIR, "image_metadata.csv"))
        if "buildings_gdf" not in results:
            results["buildings_gdf"] = gpd.read_file(
                os.path.join(OSM_DIR, "buildings.gpkg"))

        from step3_classify import classify_images
        results["class_df"] = run_step(
            3, "Classify images → buildings (ray-casting)",
            classify_images, results["meta_df"], results["buildings_gdf"])

    # ── Step 4: Generate 3D model ─────────────────────────────────────────────
    if 4 in steps:
        import geopandas as gpd
        from config import OSM_DIR
        if "buildings_gdf" not in results:
            results["buildings_gdf"] = gpd.read_file(
                os.path.join(OSM_DIR, "buildings.gpkg"))

        from step4_model import build_model
        results["facades"] = run_step(
            4, "Generate LoD1 3D model (OBJ + MTL)",
            build_model, results["buildings_gdf"])

    # ── Step 5: Texture mapping ───────────────────────────────────────────────
    if 5 in steps:
        import json, pandas as pd
        from config import CLASS_DIR, MODEL_DIR

        if "class_df" not in results:
            results["class_df"] = pd.read_csv(
                os.path.join(CLASS_DIR, "classification.csv"))

        if "facades" not in results:
            with open(os.path.join(MODEL_DIR, "building_meta.json")) as f:
                meta = json.load(f)
            ox = meta["utm_origin"]["east"]
            oy = meta["utm_origin"]["north"]
            facades = meta["facades"]
            for fac in facades:
                a, b = fac["corner_a_EN"], fac["corner_b_EN"]
                fac["corner_a_EN"] = [a[0] + ox, a[1] + oy]
                fac["corner_b_EN"] = [b[0] + ox, b[1] + oy]
            results["facades"] = facades

        from step5_texture import apply_textures
        results["tex_log"] = run_step(
            5, "Apply perspective-correct textures",
            apply_textures, results["class_df"], results["facades"])

    # ── Step 6: Visualise ─────────────────────────────────────────────────────
    if 6 in steps:
        from step6_visualize import run as vis_run
        run_step(6, "Generate visualizations & summary", vis_run)

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"  Output → {os.path.join(os.path.dirname(__file__), 'output')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
