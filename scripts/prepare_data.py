"""Step 4a: label statistics + scene-level train/val/test split -> data/splits.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oiltank.data import box_stats, load_coco, patches_for, scene_id, split_scenes

LABELS = ROOT / "data/raw/Oil Tanks/labels_coco.json"
TRAIN_KEEP_NEGATIVES = 0.25          # train split only; val/test keep every patch


def main():
    boxes = load_coco(LABELS)
    stats = box_stats(boxes)
    print("label statistics (all 100 scenes):")
    for k, v in stats.items():
        print(f"  {k:20s} {v}")

    scenes = split_scenes([scene_id(n) for n in boxes], seed=0)
    patches = {
        "train": patches_for(scenes["train"], boxes, keep_negatives=TRAIN_KEEP_NEGATIVES, seed=0),
        "val": patches_for(scenes["val"], boxes),
        "test": patches_for(scenes["test"], boxes),
    }
    print("\nsplit by scene (seed 0):")
    for k in ("train", "val", "test"):
        names = patches[k]
        pos = sum(len(boxes[n]) > 0 for n in names)
        n_boxes = sum(len(boxes[n]) for n in names)
        print(f"  {k:5s}: {len(scenes[k]):3d} scenes | {len(names):5d} patches "
              f"({pos} with tanks, {len(names) - pos} without) | {n_boxes} boxes")
    out = ROOT / "data/splits.json"
    json.dump({"seed": 0, "train_keep_negatives": TRAIN_KEEP_NEGATIVES,
               "scenes": scenes, "patches": patches, "label_stats": stats},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
