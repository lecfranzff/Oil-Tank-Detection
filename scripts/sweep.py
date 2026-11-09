"""How the Hardisty result changes with the confidence threshold and with image resolution.

1. Threshold sweep: at 0.55 and 0.80 m/px, detect once keeping every box scored >= 0.05,
   then for each threshold keep the boxes at or above it, run the merge pass and score.
   (NMS can only remove a box in favour of a higher-scoring one, so applying the
   threshold after NMS gives the same boxes as applying it before.)
2. Resolution sweep: resample the zoom-18 image to each resolution, detect at conf 0.5
   and score. The same model and ground truth are used throughout.

Writes results/hardisty/sweeps.json. About 1 minute on CPU.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oiltank.detect import merge_close_boxes, tile_detect
from oiltank.geo import haversine_m, resample_geotiff, true_gsd
from oiltank.model import load_checkpoint
from run_detection import LAT, OVERLAP, rows_for
from score import score

OUT = ROOT / "results/hardisty"
WORK = ROOT / "scratch/sweep"
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
RESOLUTIONS = [0.40, 0.48, 0.55, 0.64, 0.72, 0.80, 0.90, 1.00, 1.20]
KEEP = ["tp", "fp", "fn", "precision", "recall", "f1", "fixed_correctly_ignored", "fixed_tanks",
        "detections_ignored_edge", "detected_count", "diameter_mean_signed_vs_gt_m", "fn_tank_ids"]


def load_gt():
    gt = list(csv.DictReader(open(OUT / "ground_truth.csv")))
    for g in gt:
        g["lat"], g["lon"], g["diameter_m"] = float(g["lat"]), float(g["lon"]), float(g["diameter_m"])
    return gt


def open_image(path):
    with rasterio.open(path) as src:
        img = Image.fromarray(np.transpose(src.read([1, 2, 3]), (1, 2, 0)))
        transform = src.transform
    return img, transform, true_gsd(transform.a, LAT)


def nearest_tank(gt, lat, lon):
    best = gt[0]
    for g in gt:
        if haversine_m(lat, lon, g["lat"], g["lon"]) < haversine_m(lat, lon, best["lat"], best["lon"]):
            best = g
    return best


def score_boxes(gt, name, boxes, scores, transform, gsd):
    """Merge the boxes, turn them into detection rows and score them like score.py does."""
    mb, ms, counts = merge_close_boxes(boxes, scores)
    rows = rows_for(name, mb, ms, counts, transform, gsd)
    result = score(gt, rows)
    summary = {k: result[k] for k in KEEP}
    summary["false_positives"] = []              # where each false positive is
    for d in rows:
        if d["det_id"] in result["fp_det_ids"]:
            nearest = nearest_tank(gt, d["lat"], d["lon"])
            summary["false_positives"].append({
                "score": d["score"], "lon": d["lon"], "lat": d["lat"], "nearest_tank": nearest["tank_id"],
                "distance_m": round(haversine_m(d["lat"], d["lon"], nearest["lat"], nearest["lon"]), 1),
                "box_side_m": d["diameter_m"]})
    return summary


def main():
    gt = load_gt()
    model = load_checkpoint(ROOT / "checkpoints/best.pth")
    WORK.mkdir(parents=True, exist_ok=True)
    sweeps = {"thresholds": {}, "resolutions": []}

    for name in ("0.55", "0.80"):
        img, T, gsd = open_image(ROOT / f"data/hardisty/khtt_{name.replace('.', '')}.tif")
        boxes, scores, _ = tile_detect(model, img, conf=min(THRESHOLDS), overlap=OVERLAP)
        rows = []
        for t in THRESHOLDS:
            keep = scores >= t
            row = {"conf": t, **score_boxes(gt, name, boxes[keep], scores[keep], T, gsd)}
            rows.append(row)
            print(f"{name} m/px  conf {t:<4}  TP {row['tp']:2d}  FP {row['fp']:2d}  FN {row['fn']:2d}  "
                  f"fixed ignored {row['fixed_correctly_ignored']}/{row['fixed_tanks']}")
        sweeps["thresholds"][name] = rows

    for res in RESOLUTIONS:
        path = WORK / f"khtt_{res:.2f}.tif"
        resample_geotiff(ROOT / "data/hardisty/khtt_z18.tif", path, res, LAT)
        img, T, gsd = open_image(path)
        boxes, scores, info = tile_detect(model, img, conf=0.5, overlap=OVERLAP)
        row = {"gsd_m": gsd, "image_px": list(img.size), "tiles": info["tiles"],
               **score_boxes(gt, f"{res:.2f}", boxes, scores, T, gsd)}
        sweeps["resolutions"].append(row)
        print(f"{res:.2f} m/px  TP {row['tp']:2d}  FP {row['fp']:2d}  FN {row['fn']:2d}  "
              f"fixed ignored {row['fixed_correctly_ignored']}/{row['fixed_tanks']}")

    json.dump(sweeps, open(OUT / "sweeps.json", "w"), indent=1)
    print("wrote", OUT / "sweeps.json")


if __name__ == "__main__":
    main()
