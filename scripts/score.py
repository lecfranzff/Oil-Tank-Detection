"""Score detections against the hand-verified ground truth.

Rules (fixed before detection was run):
- A detection matches a GT tank if its centre lies within 0.5 x GT radius; one-to-one,
  closest pairs first.
- Targets = floating_sunken + floating_near_full. Matches to targets are TP; unmatched
  targets are FN; every other non-ignored detection is FP.
- Detections whose centre lies within the edge_excluded tank's radius are ignored.
- A fixed/internal-roof tank is "correctly ignored" when no detection centre falls inside
  its full radius.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oiltank.geo import haversine_m

OUT = ROOT / "results/hardisty"
TARGETS = ("floating_sunken", "floating_near_full")
# Published Phase C diameters (Matrix Service project sheet). Assigned to tanks by the
# size groups read from the image (3 small / 3 medium / 2 large) -- see metrics.md note.
SPEC = {"W1": 55.17, "W7": 55.17, "W8": 55.17,
        "W2": 65.23, "W3": 65.23, "W5": 65.23,
        "W4": 69.80, "W6": 69.80}


def score(gt, dets):
    ignore_zones = [g for g in gt if g["type"] == "edge_excluded"]
    kept, ignored = [], []
    for d in dets:
        inside = any(haversine_m(d["lat"], d["lon"], z["lat"], z["lon"]) <= z["diameter_m"] / 2
                     for z in ignore_zones)
        (ignored if inside else kept).append(d)

    scored_gt = [g for g in gt if g["type"] != "edge_excluded"]
    pairs = sorted((haversine_m(d["lat"], d["lon"], g["lat"], g["lon"]), i, j)
                   for i, d in enumerate(kept) for j, g in enumerate(scored_gt))
    match_d, match_g = {}, {}
    for dist, i, j in pairs:
        if i in match_d or j in match_g or dist > 0.25 * scored_gt[j]["diameter_m"]:
            continue
        match_d[i], match_g[j] = j, i

    targets = [j for j, g in enumerate(scored_gt) if g["type"] in TARGETS]
    tp = [j for j in targets if j in match_g]
    fn = [j for j in targets if j not in match_g]
    fp = []                                   # unmatched, or matched to a non-target tank
    for i in range(len(kept)):
        if i not in match_d or scored_gt[match_d[i]]["type"] not in TARGETS:
            fp.append(i)
    n_tp, n_fp, n_fn = len(tp), len(fp), len(fn)
    precision = n_tp / (n_tp + n_fp) if n_tp + n_fp else float("nan")
    recall = n_tp / (n_tp + n_fn) if n_tp + n_fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if n_tp else 0.0

    def recall_of(kind):
        ids = [j for j in targets if scored_gt[j]["type"] == kind]
        return (sum(j in match_g for j in ids), len(ids))

    fixed = [g for g in scored_gt if g["type"] == "fixed_or_internal_roof"]
    fixed_hits = {g["tank_id"]: [d["det_id"] for d in kept
                                 if haversine_m(d["lat"], d["lon"], g["lat"], g["lon"]) <= g["diameter_m"] / 2]
                  for g in fixed}
    n_ignored_ok = sum(not v for v in fixed_hits.values())

    per_tank, err_gt, err_spec = [], [], []
    for j in tp:
        g, d = scored_gt[j], kept[match_g[j]]
        row = {"tank_id": g["tank_id"], "det_id": d["det_id"], "score": d["score"],
               "centre_offset_m": haversine_m(d["lat"], d["lon"], g["lat"], g["lon"]),
               "gt_d_m": g["diameter_m"], "det_d_m": d["diameter_m"],
               "n_fragments": d["n_fragments"], "spec_d_m": SPEC.get(g["tank_id"])}
        err_gt.append(abs(d["diameter_m"] - g["diameter_m"]))
        if row["spec_d_m"]:
            err_spec.append(abs(d["diameter_m"] - row["spec_d_m"]))
        per_tank.append(row)
    return {
        "detections_total": len(dets), "detections_ignored_edge": len(ignored),
        "ignored_det_ids": [d["det_id"] for d in ignored],
        "tp": n_tp, "fp": n_fp, "fn": n_fn, "precision": precision, "recall": recall, "f1": f1,
        "fp_det_ids": [kept[i]["det_id"] for i in fp],
        "fn_tank_ids": [scored_gt[j]["tank_id"] for j in fn],
        "recall_sunken": recall_of("floating_sunken"), "recall_near_full": recall_of("floating_near_full"),
        "fixed_tanks": len(fixed), "fixed_correctly_ignored": n_ignored_ok,
        "fixed_hits": fixed_hits,
        "diameter_mae_vs_gt_m": float(np.mean(err_gt)) if err_gt else float("nan"),
        "diameter_mean_signed_vs_gt_m": float(np.mean([r["det_d_m"] - r["gt_d_m"] for r in per_tank])) if per_tank else float("nan"),
        "diameter_mae_vs_spec_m": float(np.mean(err_spec)) if err_spec else float("nan"),
        "n_spec_compared": len(err_spec),
        "detected_count": len(kept), "gt_target_count": len(targets),
        "per_tank": per_tank,
    }


def main():
    gt = list(csv.DictReader(open(OUT / "ground_truth.csv")))
    for g in gt:
        g["lat"], g["lon"], g["diameter_m"] = float(g["lat"]), float(g["lon"]), float(g["diameter_m"])
    dets = list(csv.DictReader(open(OUT / "detections.csv")))
    for d in dets:
        for k in ("lat", "lon", "diameter_m", "score"):
            d[k] = float(d[k])
        d["n_fragments"] = int(d["n_fragments"])
    summary = json.load(open(OUT / "detection_summary.json"))
    results = {}
    for res in ("0.55", "0.80"):
        r = score(gt, [d for d in dets if d["gsd_m"] == res])
        r.update({k: summary[res][k] for k in ("raw_boxes", "after_nms", "after_merge", "cpu_seconds", "tiles", "gsd_m")})
        results[res] = r
        print(f"\n=== {res} m/px ===")
        for k, v in r.items():
            if k != "per_tank":
                print(f"  {k}: {v}")
        for row in r["per_tank"]:
            print("   ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
    json.dump(results, open(OUT / "metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
