"""
Step 6 – Visualization & summary report.

Produces:
  output/visualization/building_gallery.png  – mosaic: each building + best images
  output/visualization/coverage_map.png      – map coloured by texture coverage
  output/visualization/campus_3d.png         – matplotlib 3D preview of the model
  output/visualization/summary.txt           – text summary
"""

import os, sys, json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from shapely.geometry import MultiPolygon

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    IMAGE_DIR, MODEL_DIR, CLASS_DIR, OSM_DIR, VIS_DIR, TEX_DIR,
    UTM_CRS,
)


# ── 3-D preview ──────────────────────────────────────────────────────────────

def _plot_3d(buildings_gdf: gpd.GeoDataFrame, meta: dict):
    from pyproj import Transformer
    WGS_TO_UTM = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)
    from config import CAMPUS_CENTER_WGS
    ox_lon, ox_lat = CAMPUS_CENTER_WGS[1], CAMPUS_CENTER_WGS[0]
    ox, oy = WGS_TO_UTM.transform(ox_lon, ox_lat)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    colours = plt.cm.tab20.colors
    all_heights = []

    for ci, (_, row) in enumerate(buildings_gdf.iterrows()):
        geom = row.geometry
        h = float(row["height_m"])
        all_heights.append(h)

        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)

        coords = np.array(geom.exterior.coords[:-1])
        xs = coords[:, 0] - ox
        ys = coords[:, 1] - oy
        N = len(xs)

        col = colours[ci % len(colours)]
        alpha = 0.7

        # side faces
        for i in range(N):
            j = (i + 1) % N
            verts = [[xs[i], ys[i], 0], [xs[j], ys[j], 0],
                     [xs[j], ys[j], h], [xs[i], ys[i], h]]
            poly = Poly3DCollection([verts], alpha=alpha,
                                    facecolor=col, edgecolor="white", linewidth=0.2)
            ax.add_collection3d(poly)

        # top face
        top_verts = [[xs[i], ys[i], h] for i in range(N)]
        poly_top = Poly3DCollection([top_verts], alpha=0.9,
                                    facecolor=col, edgecolor="white", linewidth=0.3)
        ax.add_collection3d(poly_top)

    # auto-set limits
    all_coords = []
    for _, row in buildings_gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)
        c = np.array(geom.exterior.coords)
        all_coords.append(c)

    all_coords = np.vstack(all_coords)
    xs_all = all_coords[:, 0] - ox
    ys_all = all_coords[:, 1] - oy

    margin = 30
    ax.set_xlim(xs_all.min() - margin, xs_all.max() + margin)
    ax.set_ylim(ys_all.min() - margin, ys_all.max() + margin)
    ax.set_zlim(0, max(all_heights) * 1.1)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Height (m)")
    ax.set_title("IC South Kensington – LoD1 3D Building Model", fontsize=12)
    ax.view_init(elev=35, azim=-60)

    out = os.path.join(VIS_DIR, "campus_3d.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  3-D preview → {out}")


# ── building gallery ──────────────────────────────────────────────────────────

def _plot_gallery(class_df: pd.DataFrame, buildings_gdf: gpd.GeoDataFrame,
                  n_cols=5, max_buildings=40):
    """Show the best street-view image for each named / textured building."""
    from PIL import Image as PILImage

    classified = class_df[class_df["building_id"].notna()].copy()
    bids = classified["building_id"].unique()[:max_buildings]

    n_rows = int(np.ceil(len(bids) / n_cols))
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.0))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    building_meta = {int(r["building_id"]): r
                     for _, r in buildings_gdf.iterrows()
                     if "building_id" in buildings_gdf.columns}

    for idx, bid in enumerate(bids):
        row_i, col_i = divmod(idx, n_cols)
        ax = axes[row_i][col_i]

        subset = classified[classified["building_id"] == bid]
        # Pick the image with smallest distance
        best = subset.loc[subset["distance_m"].idxmin()]

        img_path = os.path.join(IMAGE_DIR, best["filename"])
        try:
            img = PILImage.open(img_path)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "No image", ha="center", va="center",
                    transform=ax.transAxes)

        brow = building_meta.get(int(bid), {})
        label = str(brow.get("name", f"Bldg {bid}"))[:28]
        h = float(brow.get("height_m", 0))
        ax.set_title(f"{label}\n{len(subset)} imgs | {h:.0f}m",
                     fontsize=6.5, pad=2)

    fig.suptitle("IC Campus – Best Street View per Building", fontsize=13)
    plt.tight_layout()

    out = os.path.join(VIS_DIR, "building_gallery.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Gallery     → {out}")


# ── coverage map ──────────────────────────────────────────────────────────────

def _plot_coverage(class_df: pd.DataFrame, buildings_gdf: gpd.GeoDataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    gdf_wgs = buildings_gdf.to_crs("EPSG:4326")
    counts  = class_df[class_df["building_id"].notna()
                       ].groupby("building_id").size().to_dict()
    gdf_wgs["img_count"] = gdf_wgs["building_id"].map(counts).fillna(0)

    # Left: image count per building
    ax = axes[0]
    gdf_wgs.plot(ax=ax, column="img_count", cmap="YlGnBu",
                 edgecolor="#444", linewidth=0.5, legend=True,
                 legend_kwds={"label": "# images", "shrink": 0.6})
    ax.set_title("Images per Building")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    # Right: headings distribution
    ax2 = axes[1]
    by_heading = class_df[class_df["building_id"].notna()
                          ].groupby("heading").size()
    angles = np.radians(by_heading.index.values)
    values = by_heading.values

    ax2.remove()
    ax_polar = fig.add_subplot(1, 2, 2, projection="polar")
    ax_polar.bar(angles, values, width=np.radians(45), align="center",
                 alpha=0.7, color="steelblue", edgecolor="white")
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_title("Camera Heading Distribution\n(classified images)",
                       pad=15, fontsize=10)
    tick_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ax_polar.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax_polar.set_xticklabels(tick_labels, fontsize=8)

    plt.suptitle("IC Campus – Street-View Coverage Analysis", fontsize=13)
    plt.tight_layout()

    out = os.path.join(VIS_DIR, "coverage_map.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Coverage    → {out}")


# ── texture preview ───────────────────────────────────────────────────────────

def _plot_texture_preview(n_show=12):
    """Show a sample of generated texture images in a grid."""
    tex_files = sorted([f for f in os.listdir(TEX_DIR) if f.endswith(".png")])[:n_show]
    if not tex_files:
        return

    n_cols = 4
    n_rows = int(np.ceil(len(tex_files) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 2.5))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for i, fname in enumerate(tex_files):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        img = Image.open(os.path.join(TEX_DIR, fname))
        ax.imshow(img)
        ax.set_title(fname.replace(".png", ""), fontsize=6.5)

    fig.suptitle("Sample Generated Facade Textures", fontsize=12)
    plt.tight_layout()

    out = os.path.join(VIS_DIR, "texture_preview.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Texture preview → {out}")


# ── summary report ────────────────────────────────────────────────────────────

def _write_summary(class_df: pd.DataFrame, buildings_gdf: gpd.GeoDataFrame):
    lines = [
        "=" * 60,
        "  IC South Kensington – 3D City Model Pipeline Summary",
        "=" * 60,
        "",
        f"  Street-view images   : {len(class_df)}",
        f"  Camera locations     : {class_df['location_id'].nunique()}",
        f"  Heading directions   : {sorted(class_df['heading'].unique())}",
        "",
        f"  OSM buildings found  : {len(buildings_gdf)}",
        f"  Named buildings      : {buildings_gdf['name'].notna().sum() if 'name' in buildings_gdf.columns else 'N/A'}",
        f"  Height range         : {buildings_gdf['height_m'].min():.1f} – "
            f"{buildings_gdf['height_m'].max():.1f} m",
        "",
    ]

    classified = class_df[class_df["building_id"].notna()]
    lines += [
        f"  Images classified    : {len(classified)} / {len(class_df)} "
            f"({len(classified)/len(class_df)*100:.1f}%)",
        f"  Buildings w/ images  : {classified['building_id'].nunique()}",
        "",
    ]

    tex_log_path = os.path.join(CLASS_DIR, "texture_log.csv")
    if os.path.exists(tex_log_path):
        log = pd.read_csv(tex_log_path)
        lines += [
            f"  Textured facades     : {len(log)}",
            f"  Mean texture cov.    : {log['coverage'].mean()*100:.1f}%",
        ]

    lines += [
        "",
        "  Output files:",
        f"    models/campus_lod1.obj  – 3D building geometry",
        f"    models/campus_lod1.mtl  – material library with textures",
        f"    textures/               – facade texture PNG images",
        f"    visualization/          – maps and previews",
        "=" * 60,
    ]

    txt = "\n".join(lines)
    out = os.path.join(VIS_DIR, "summary.txt")
    with open(out, "w") as f:
        f.write(txt)
    print(txt)
    print(f"\n  Summary saved → {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def run():
    os.makedirs(VIS_DIR, exist_ok=True)

    class_df = pd.read_csv(os.path.join(CLASS_DIR, "classification.csv"))
    buildings_gdf = gpd.read_file(os.path.join(OSM_DIR, "buildings.gpkg"))

    meta_path = os.path.join(MODEL_DIR, "building_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    print("[Step 6] Generating visualizations …")
    _plot_3d(buildings_gdf, meta)
    _plot_gallery(class_df, buildings_gdf)
    _plot_coverage(class_df, buildings_gdf)
    _plot_texture_preview()
    _write_summary(class_df, buildings_gdf)


if __name__ == "__main__":
    run()
