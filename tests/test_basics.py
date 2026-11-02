import json
import math
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from oiltank.data import split_scenes, scene_id
from oiltank.evaluate import average_precision, iou, precision_recall_at
from oiltank.geo import lonlat_to_merc, lonlat_to_pixel, pixel_to_lonlat, true_gsd

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits.json"
PATCHES = Path(__file__).resolve().parents[1] / "data" / "raw" / "Oil Tanks" / "image_patches"


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_ap_perfect_and_with_false_positive():
    gt = [np.array([[0, 0, 10, 10]], dtype=float)]
    perfect = [(np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9]))]
    assert average_precision(perfect, gt) == pytest.approx(1.0)
    # a higher-scoring false positive halves precision at the only recall step
    with_fp = [(np.array([[50, 50, 60, 60], [0, 0, 10, 10]], dtype=float), np.array([0.95, 0.9]))]
    assert average_precision(with_fp, gt) == pytest.approx(0.5)
    pr = precision_recall_at(with_fp, gt, conf=0.5)
    assert (pr["tp"], pr["fp"], pr["fn"]) == (1, 1, 0)


def test_scene_split_is_disjoint_and_complete():
    scenes = [f"{i:02d}" for i in range(1, 101)]
    s = split_scenes(scenes, seed=0)
    sets = [set(s[k]) for k in ("train", "val", "test")]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
    assert set().union(*sets) == set(scenes)
    assert scene_id("07_3_9.jpg") == "07"


@pytest.mark.skipif(not SPLITS.exists(), reason="data/splits.json not created yet")
def test_saved_split_has_no_scene_in_two_splits():
    s = json.load(open(SPLITS))
    seen = {}
    for split, names in s["patches"].items():
        for name in names:
            sid = scene_id(name)
            assert seen.setdefault(sid, split) == split, f"scene {sid} is in {seen[sid]} and {split}"


def test_true_gsd_hardisty():
    assert true_gsd(0.5972, 52.7) == pytest.approx(0.3620, abs=1e-3)


def test_pixel_lonlat_round_trip():
    x0, y0 = lonlat_to_merc(-111.28, 52.66)
    T = from_origin(x0, y0, 0.9065, 0.9065)
    lon, lat = pixel_to_lonlat(T, 120.0, 340.0)
    col, row = lonlat_to_pixel(T, lon, lat)
    assert (col, row) == (pytest.approx(120.0, abs=1e-6), pytest.approx(340.0, abs=1e-6))
    assert lon > -111.28 and lat < 52.66          # moved east and south of the origin


def test_checkpoint_reload_gives_identical_outputs(tmp_path):
    """Save -> strict reload -> same boxes and scores on one patch."""
    patch = PATCHES / "01_3_3.jpg"
    if not patch.exists():
        pytest.skip("needs the Kaggle patches in data/raw (a random image gives no boxes)")
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from oiltank.model import build_model, load_checkpoint, save_checkpoint

    torch.manual_seed(0)
    model = build_model(pretrained=True).eval()       # COCO weights: produces real boxes
    path = tmp_path / "m.pth"
    save_checkpoint(model, path)
    reloaded = load_checkpoint(path)                  # load_state_dict is strict
    image = TF.to_tensor(Image.open(patch).convert("RGB"))
    with torch.no_grad():
        a, b = model([image])[0], reloaded([image])[0]
    assert len(a["boxes"]) > 0                        # compare real detections, not two empty lists
    assert torch.allclose(a["boxes"], b["boxes"]) and torch.allclose(a["scores"], b["scores"])


def test_merge_joins_fragments_but_keeps_neighbours_apart():
    from oiltank.detect import merge_close_boxes

    # one 120 px tank split into three overlapping fragments
    frags = [[100, 100, 180, 220], [150, 100, 220, 220], [100, 160, 220, 220]]
    b, s, n = merge_close_boxes(frags, [0.9, 0.8, 0.7])
    assert len(b) == 1 and n == [3]
    assert list(b[0]) == [100, 100, 220, 220] and s[0] == pytest.approx(0.9)
    # two whole 120 px tanks whose centres are 180 px apart (like W6/W8: 99 m at 0.55 m/px)
    b, s, n = merge_close_boxes([[0, 0, 120, 120], [180, 0, 300, 120]], [0.9, 0.9])
    assert len(b) == 2 and n == [1, 1]
