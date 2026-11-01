"""Metrics: IoU, AP@0.5 and precision/recall at a confidence threshold.

Written out by hand (no pycocotools) so every step can be explained.
"""
import numpy as np


def iou(a, b):
    """IoU of two boxes [x1, y1, x2, y2]."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_detections(preds, gts, iou_thr=0.5):
    """Greedy matching over a dataset.

    preds: list over images of (boxes Nx4, scores N); gts: list over images of boxes Mx4.
    Returns (scores, is_tp) for every prediction, plus the number of GT boxes.
    Highest-scoring predictions claim GT boxes first; each GT box can be claimed once.
    """
    all_scores, all_tp = [], []
    n_gt = 0
    for (boxes, scores), gt in zip(preds, gts):
        n_gt += len(gt)
        used = [False] * len(gt)
        for k in np.argsort(-np.asarray(scores)):
            best, best_j = 0.0, -1
            for j, g in enumerate(gt):
                o = iou(boxes[k], g)
                if not used[j] and o > best:
                    best, best_j = o, j
            hit = best >= iou_thr
            if hit:
                used[best_j] = True
            all_scores.append(float(scores[k]))
            all_tp.append(hit)
    return np.array(all_scores), np.array(all_tp, dtype=bool), n_gt


def average_precision(preds, gts, iou_thr=0.5):
    """Area under the precision-recall curve (all-point interpolation, as in VOC2010+)."""
    scores, tp, n_gt = match_detections(preds, gts, iou_thr)
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-scores)
    tp = tp[order]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(~tp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    # make precision non-increasing from right to left, then sum the area under it
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    steps = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[steps + 1] - r[steps]) * p[steps + 1]))


def precision_recall_at(preds, gts, conf=0.5, iou_thr=0.5):
    """TP/FP/FN, precision and recall keeping only predictions with score >= conf."""
    kept = [(b[np.asarray(s) >= conf], np.asarray(s)[np.asarray(s) >= conf]) for b, s in preds]
    scores, tp, n_gt = match_detections(kept, gts, iou_thr)
    n_tp = int(tp.sum())
    n_fp = int((~tp).sum())
    return {"tp": n_tp, "fp": n_fp, "fn": n_gt - n_tp,
            "precision": n_tp / (n_tp + n_fp) if n_tp + n_fp else float("nan"),
            "recall": n_tp / n_gt if n_gt else float("nan")}
