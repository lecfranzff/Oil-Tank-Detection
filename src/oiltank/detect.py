"""Run the detector over a large image: overlapping tiles -> one global NMS -> merge pass."""
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.ops import nms


def tile_detect(model, img, conf=0.5, tile=512, overlap=0.3, nms_iou=0.5, device="cpu"):
    """Detect tanks on a PIL image of any size.

    Tiles overlap so every tank appears whole in at least one tile (overlap in px must be
    larger than the biggest tank). Boxes from all tiles are shifted to full-image
    coordinates and de-duplicated with ONE global NMS at the end.
    Returns boxes (N x 4 numpy), scores (N), and a dict of counts/timing.
    """
    stride = max(1, int(round(tile * (1.0 - overlap))))
    W, H = img.size
    xs = list(range(0, max(1, W - tile + 1), stride)) or [0]
    ys = list(range(0, max(1, H - tile + 1), stride)) or [0]
    if xs[-1] + tile < W:
        xs.append(W - tile)                       # last column flush with the right edge
    if ys[-1] + tile < H:
        ys.append(H - tile)

    boxes, scores = [], []
    t0 = time.time()
    model.eval()
    for top in ys:
        for left in xs:
            crop = img.crop((left, top, left + tile, top + tile))
            with torch.no_grad():
                o = model([TF.to_tensor(crop).to(device)])[0]
            keep = o["scores"] >= conf
            b, s = o["boxes"][keep].cpu(), o["scores"][keep].cpu()
            if len(b):
                b[:, [0, 2]] += left
                b[:, [1, 3]] += top
                boxes.append(b)
                scores.append(s)
    info = {"tiles": len(xs) * len(ys), "seconds": time.time() - t0, "raw": 0, "after_nms": 0}
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), info
    B, S = torch.cat(boxes), torch.cat(scores)
    keep = nms(B, S, nms_iou)
    info["raw"], info["after_nms"] = len(B), len(keep)
    return B[keep].numpy(), S[keep].numpy(), info


def box_centre(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def box_side(b):
    """Mean of width and height."""
    return (b[2] - b[0] + b[3] - b[1]) / 2


def find_close_pair(boxes, factor):
    """First pair (i, j) whose centres are closer than factor x the larger box's side."""
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            distance = np.linalg.norm(box_centre(boxes[i]) - box_centre(boxes[j]))
            if distance < factor * max(box_side(boxes[i]), box_side(boxes[j])):
                return i, j
    return None


def merge_close_boxes(boxes, scores, factor=0.5):
    """Union boxes that are fragments of one tank.

    While some pair of boxes is close (see find_close_pair), replace the pair by one box
    covering both, keeping the higher score. Returns boxes, scores, and how many original
    boxes went into each merged box.
    """
    boxes = [np.asarray(b, dtype=float) for b in boxes]
    scores = [float(s) for s in scores]
    counts = [1] * len(boxes)
    pair = find_close_pair(boxes, factor)
    while pair is not None:
        i, j = pair
        a, b = boxes[i], boxes[j]
        boxes[i] = np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])])
        scores[i] = max(scores[i], scores[j])
        counts[i] += counts[j]
        del boxes[j], scores[j], counts[j]
        pair = find_close_pair(boxes, factor)
    return np.array(boxes).reshape(-1, 4), np.array(scores), counts
