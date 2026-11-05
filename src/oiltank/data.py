"""Kaggle Oil Storage Tanks data: read COCO labels, split by scene, serve training patches.

Patch file names look like `01_3_7.jpg` = scene 01, row 3, col 7. The 100 patches of a
scene overlap each other, so patches from one scene must never be split across
train/val/test, or near-duplicate patches would leak into evaluation.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF


def scene_id(file_name):
    """'01_3_7.jpg' -> '01'."""
    return Path(file_name).name.split("_")[0]


def load_coco(path, category_name="Floating Head Tank"):
    """Return {file_name: array of boxes [x1, y1, x2, y2]} for every patch (empty if no tanks).

    COCO stores boxes as [x_min, y_min, width, height]; torchvision wants corners.
    """
    coco = json.load(open(path))
    cat_ids = {c["id"] for c in coco["categories"] if c["name"] == category_name}
    names = {im["id"]: Path(im["file_name"]).name for im in coco["images"]}
    boxes = {name: [] for name in names.values()}
    for a in coco["annotations"]:
        if a["category_id"] not in cat_ids:
            continue
        x, y, w, h = a["bbox"]
        if w < 1 or h < 1:                        # skip degenerate boxes
            continue
        boxes[names[a["image_id"]]].append([x, y, x + w, y + h])
    return {k: np.array(v, dtype=np.float32).reshape(-1, 4) for k, v in boxes.items()}


def box_stats(boxes_by_patch):
    """Counts and box side length (mean of width and height) in pixels."""
    sides = np.concatenate([(b[:, 2] - b[:, 0] + b[:, 3] - b[:, 1]) / 2
                            for b in boxes_by_patch.values() if len(b)])
    return {
        "patches": len(boxes_by_patch),
        "patches_with_tanks": sum(len(b) > 0 for b in boxes_by_patch.values()),
        "boxes": len(sides),
        "side_median_px": float(np.median(sides)),
        "side_p75_px": float(np.percentile(sides, 75)),
        "side_max_px": float(sides.max()),
    }


def split_scenes(scenes, seed=0, fractions=(0.70, 0.15, 0.15)):
    """Shuffle scene ids with a fixed seed and cut them into train/val/test lists."""
    scenes = sorted(set(scenes))
    random.Random(seed).shuffle(scenes)
    n_train = round(fractions[0] * len(scenes))
    n_val = round(fractions[1] * len(scenes))
    return {"train": sorted(scenes[:n_train]),
            "val": sorted(scenes[n_train:n_train + n_val]),
            "test": sorted(scenes[n_train + n_val:])}


def patches_for(split_scene_ids, boxes_by_patch, keep_negatives=1.0, seed=0):
    """Patch names belonging to the given scenes.

    keep_negatives < 1 keeps only that fraction of tank-free patches (used for train only,
    so the CPU spends its time on patches that actually contain tanks).
    """
    rng = random.Random(seed)
    wanted = set(split_scene_ids)
    out = []
    for name in sorted(boxes_by_patch):
        if scene_id(name) not in wanted:
            continue
        if len(boxes_by_patch[name]) == 0 and rng.random() >= keep_negatives:
            continue
        out.append(name)
    return out


class TankPatches(torch.utils.data.Dataset):
    """Returns (image tensor, target dict) in the format torchvision detectors expect."""

    def __init__(self, image_dir, names, boxes_by_patch, train=False):
        self.image_dir = Path(image_dir)
        self.names = names
        self.boxes = boxes_by_patch
        self.train = train                         # random flips only when training

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = TF.to_tensor(Image.open(self.image_dir / name).convert("RGB"))
        # torch.tensor makes a copy; editing a view of the stored labels would flip them
        # for good, and the next epoch would pair an unflipped image with flipped boxes
        boxes = torch.tensor(self.boxes[name]).reshape(-1, 4)
        w, h = img.shape[2], img.shape[1]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)   # some labels run past the patch
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)   # edge; keep them inside it
        if self.train:
            if random.random() < 0.5:              # horizontal flip
                img = img.flip(2)
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if random.random() < 0.5:              # vertical flip
                img = img.flip(1)
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        target = {"boxes": boxes,
                  "labels": torch.ones(len(boxes), dtype=torch.int64)}   # class 1 = tank
        return img, target


def collate(batch):
    """Detection batches are lists (images can hold different numbers of boxes)."""
    return tuple(zip(*batch))
