"""
Step 1 – Parse street-view image metadata from filenames.

Filename format: {lat}_{lon}_h{heading}_p{pitch}.jpg
Output: output/metadata/image_metadata.csv
"""

import os
import re
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import IMAGE_DIR, META_DIR

FNAME_RE = re.compile(
    r"^(?P<lat>-?[\d.]+)_(?P<lon>-?[\d.]+)_h(?P<heading>\d+)_p(?P<pitch>-?\d+)\.jpg$"
)


def parse_metadata() -> pd.DataFrame:
    records = []
    for fname in sorted(os.listdir(IMAGE_DIR)):
        m = FNAME_RE.match(fname)
        if not m:
            continue
        records.append({
            "filename":    fname,
            "filepath":    os.path.join(IMAGE_DIR, fname),
            "lat":         float(m.group("lat")),
            "lon":         float(m.group("lon")),
            "heading":     int(m.group("heading")),
            "pitch":       int(m.group("pitch")),
        })

    df = pd.DataFrame(records)

    # Unique location ID (same lat/lon share a location)
    loc_keys = df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    loc_keys["location_id"] = loc_keys.index
    df = df.merge(loc_keys, on=["lat", "lon"])

    os.makedirs(META_DIR, exist_ok=True)
    out = os.path.join(META_DIR, "image_metadata.csv")
    df.to_csv(out, index=False)

    print(f"[Step 1] Parsed {len(df)} images across {df['location_id'].nunique()} locations")
    print(f"         Headings: {sorted(df['heading'].unique())}")
    print(f"         Lat {df['lat'].min():.6f}–{df['lat'].max():.6f},  "
          f"Lon {df['lon'].min():.6f}–{df['lon'].max():.6f}")
    print(f"         Saved → {out}")
    return df


if __name__ == "__main__":
    parse_metadata()
