"""Detect tanks on the Hardisty AOI at 0.55 and 0.8 m/px, before and after the merge pass.

Writes scratch/hardisty/detections.csv (merged boxes) and detections_raw.csv (after NMS,
before merge), and prints counts and CPU runtime.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oiltank.detect import merge_close_boxes, tile_detect
from oiltank.geo import pixel_to_lonlat, true_gsd
from oiltank.model import load_checkpoint

LAT = 52.65455
IMAGES = {"0.55": ROOT / "data/hardisty/khtt_055.tif", "0.80": ROOT / "data/hardisty/khtt_080.tif"}
OUT = ROOT / "scratch/hardisty"
CONF, OVERLAP = 0.5, 0.3


def rows_for(res, boxes, scores, counts, transform, gsd):
    rows = []
    for i, ((x1, y1, x2, y2), s, n) in enumerate(zip(boxes, scores, counts)):
        lon, lat = pixel_to_lonlat(transform, (x1 + x2) / 2, (y1 + y2) / 2)
        rows.append({"gsd_m": res, "det_id": f"D{res}_{i + 1}", "score": round(float(s), 4),
                     "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                     "x2": round(float(x2), 1), "y2": round(float(y2), 1),
                     "lon": round(lon, 7), "lat": round(lat, 7),
                     "diameter_m": round(float((x2 - x1 + y2 - y1) / 2 * gsd), 2),
                     "n_fragments": n})
    return rows


def main():
    model = load_checkpoint(ROOT / "checkpoints/best.pth")
    merged_rows, raw_rows, summary = [], [], {}
    for res, path in IMAGES.items():
        with rasterio.open(path) as src:
            img = Image.fromarray(np.transpose(src.read([1, 2, 3]), (1, 2, 0)))
            T = src.transform
        gsd = true_gsd(T.a, LAT)
        t0 = time.time()
        boxes, scores, info = tile_detect(model, img, conf=CONF, overlap=OVERLAP)
        mb, ms, counts = merge_close_boxes(boxes, scores)
        seconds = time.time() - t0
        raw_rows += rows_for(res, boxes, scores, [1] * len(boxes), T, gsd)
        merged_rows += rows_for(res, mb, ms, counts, T, gsd)
        summary[res] = {"image_px": img.size, "gsd_m": gsd, "tiles": info["tiles"],
                        "raw_boxes": info["raw"], "after_nms": info["after_nms"],
                        "after_merge": len(mb), "cpu_seconds": seconds}
        print(f"{res} m/px: {img.size[0]}x{img.size[1]} px, {info['tiles']} tiles, "
              f"raw {info['raw']} -> NMS {info['after_nms']} -> merge {len(mb)}  ({seconds:.1f} s CPU)")
    for name, rows in (("detections.csv", merged_rows), ("detections_raw.csv", raw_rows)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(merged_rows[0] if merged_rows else raw_rows[0]))
            w.writeheader()
            w.writerows(rows)
    json.dump(summary, open(OUT / "detection_summary.json", "w"), indent=1)
    print("wrote", OUT / "detections.csv")


if __name__ == "__main__":
    main()
