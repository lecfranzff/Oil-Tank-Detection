"""Figures: annotated full images, crop sheet, one-page summary.png.

Colours: GT targets green, detections red, fixed-roof tanks grey (labelled whether they
were correctly ignored), edge-excluded tank dark grey.
"""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oiltank.geo import lonlat_to_pixel, true_gsd

LAT = 52.65455
OUT = ROOT / "results/hardisty"
FIGS = ROOT / "docs/figures"
IMAGES = {"0.55": ROOT / "data/hardisty/khtt_055.tif", "0.80": ROOT / "data/hardisty/khtt_080.tif"}
GREEN, RED, GREY, DARK = (0, 230, 0), (255, 40, 40), (200, 200, 200), (90, 90, 90)


def load():
    gt = list(csv.DictReader(open(OUT / "ground_truth.csv")))
    dets = list(csv.DictReader(open(OUT / "detections.csv")))
    metrics = json.load(open(OUT / "metrics.json"))
    return gt, dets, metrics


def trunc2(v):
    """Score shown with 2 decimals, truncated (0.999 -> 0.99, never rounded up to 1.00)."""
    return f"{math.floor(float(v) * 100 + 1e-9) / 100:.2f}"


def label(d, xy, text, fill, size):
    """Draw text clamped so it stays inside the image."""
    x = min(max(xy[0], 2), size[0] - 7 * len(text) - 2)
    y = min(max(xy[1], 2), size[1] - 14)
    d.text((x, y), text, fill=fill)


def fixed_roof_circles(gt, T, gsd):
    """(col, row, radius) in image px for every fixed/internal-roof tank."""
    out = []
    for g in gt:
        if g["type"] == "fixed_or_internal_roof":
            c, r = lonlat_to_pixel(T, float(g["lon"]), float(g["lat"]))
            out.append((c, r, float(g["diameter_m"]) / gsd / 2, g["tank_id"]))
    return out


def annotate(res, gt, dets, metrics, width=4, draw_fixed=True):
    with rasterio.open(IMAGES[res]) as src:
        img = Image.fromarray(np.transpose(src.read([1, 2, 3]), (1, 2, 0)))
        T = src.transform
    gsd = true_gsd(T.a, LAT)
    d = ImageDraw.Draw(img)
    hits = metrics[res]["fixed_hits"]
    for g in gt:
        c, r = lonlat_to_pixel(T, float(g["lon"]), float(g["lat"]))
        rad = float(g["diameter_m"]) / gsd / 2
        if g["type"] in ("floating_sunken", "floating_near_full"):
            colour, text = GREEN, g["tank_id"]
        elif g["type"] == "edge_excluded":
            colour, text = DARK, f'{g["tank_id"]} edge-excluded'
        else:
            if not draw_fixed:
                continue
            colour = GREY
            text = f'{g["tank_id"]} fixed: ' + ("ignored" if not hits.get(g["tank_id"]) else "FP")
        d.ellipse([c - rad, r - rad, c + rad, r + rad], outline=colour, width=width)
        if colour == GREY:   # white roof: dark text at the roof centre, clear of any detection box
            label(d, (c - 3.5 * len(text), r - 7), text, (30, 30, 30), img.size)
        else:
            label(d, (c - 3.5 * len(text), r - rad + 6), text, colour, img.size)
    for x in dets:
        if x["gsd_m"] != res:
            continue
        b = [float(x[k]) for k in ("x1", "y1", "x2", "y2")]
        d.rectangle(b, outline=RED, width=max(2, width - 1))
        label(d, (b[0] + 4, b[3] - 15), trunc2(x["score"]), RED, img.size)   # inside the box
    return img, T, gsd


def crop_sheet(img, T, gsd, gt, extra_crops=(), cell=300, window_m=95):
    items = [(g["tank_id"], float(g["lon"]), float(g["lat"])) for g in gt] + list(extra_crops)
    cols = 4
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), "black")
    half = window_m / gsd / 2
    for i, (name, lon, lat) in enumerate(items):
        c, r = lonlat_to_pixel(T, lon, lat)
        tile = img.crop((int(c - half), int(r - half), int(c + half), int(r + half))).resize((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + 18)
        sheet.paste(tile, (x, y + 18))
        ImageDraw.Draw(sheet).text((x + 4, y + 3), name, fill="white")
    return sheet


def summary_png(img, metrics, path, fixed=()):
    m55, m80 = metrics["0.55"], metrics["0.80"]

    def f(v, n=3):
        """Truncate (never round up) to n decimals."""
        if v is None:
            return "n/a"
        return f"{math.floor(v * 10 ** n + 1e-9) / 10 ** n:.{n}f}" if isinstance(v, float) else str(v)

    def frac(t):
        return f"{t[0]}/{t[1]}" if t[1] else "n/a (0 tanks)"

    rows = [
        ["TP / FP / FN", f'{m55["tp"]} / {m55["fp"]} / {m55["fn"]}', f'{m80["tp"]} / {m80["fp"]} / {m80["fn"]}'],
        ["Precision", f(m55["precision"]), f(m80["precision"])],
        ["Recall", f(m55["recall"]), f(m80["recall"])],
        ["F1", f(m55["f1"]), f(m80["f1"])],
        ["Recall, roof sunken", frac(m55["recall_sunken"]), frac(m80["recall_sunken"])],
        ["Recall, roof near full", frac(m55["recall_near_full"]), frac(m80["recall_near_full"])],
        ["Fixed roofs ignored", f'{m55["fixed_correctly_ignored"]}/{m55["fixed_tanks"]}',
         f'{m80["fixed_correctly_ignored"]}/{m80["fixed_tanks"]}'],
        ["Diameter MAE vs GT (m)", f(m55["diameter_mae_vs_gt_m"], 2), f(m80["diameter_mae_vs_gt_m"], 2)],
        ["Detected / GT tanks", f'{m55["detected_count"]} / {m55["gt_target_count"]}',
         f'{m80["detected_count"]} / {m80["gt_target_count"]}'],
        ["Boxes before / after merge", f'{m55["after_nms"]} / {m55["after_merge"]}',
         f'{m80["after_nms"]} / {m80["after_merge"]}'],
        ["CPU time (s)", f(m55["cpu_seconds"], 1), f(m80["cpu_seconds"], 1)],
    ]
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Oil-tank detector on an unseen Alberta terminal (Keystone Hardisty, AB)",
                 fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.02, 0.06, 0.48, 0.86])
    ax.imshow(img)
    for c, r, rad, _ in fixed:                     # fixed roofs: dark dashed outline
        ax.add_patch(plt.Circle((c, r), rad, fill=False, edgecolor="#202020",
                                linestyle=(0, (4, 3)), linewidth=1.4))
    ax.set_axis_off()
    ax.set_title("0.55 m/px: green = hand-labelled floating roof, red = detection,\n"
                 "dark dashed = fixed roof (should be ignored)", fontsize=9)
    tx = fig.add_axes([0.53, 0.30, 0.45, 0.60])
    tx.set_axis_off()
    t = tx.table(cellText=rows, colLabels=["Metric", "0.55 m/px", "0.80 m/px"],
                 loc="upper center", cellLoc="left", colWidths=[0.48, 0.26, 0.26])
    fig.text(0.53, 0.29, "Decimals truncated, not rounded.", fontsize=8)
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    t.scale(1, 1.55)
    note = ("Faster R-CNN (MobileNetV3-FPN) trained only on the Kaggle Oil Storage Tanks\n"
            "dataset (scene-level split; test mAP@0.5 on unseen Kaggle scenes = "
            f'{math.floor(metrics["model_test_ap50"] * 1000) / 1000:.3f}).\n'
            "Ground truth labelled by hand before training. Esri World Imagery,\n"
            "WorldView-3, captured 2022-06-08. One site, 15 tanks: see metrics.md.")
    fig.text(0.53, 0.10, note, fontsize=8.5, va="bottom")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    gt, dets, metrics = load()
    metrics["model_test_ap50"] = json.load(open(ROOT / "checkpoints/results.json"))["test_ap50"]
    img55, T55, gsd55 = annotate("0.55", gt, dets, metrics)
    img55.save(FIGS / "annotated.jpg", quality=90)
    img80, T80, gsd80 = annotate("0.80", gt, dets, metrics, width=3)
    img80.save(FIGS / "annotated_080.jpg", quality=90)

    fp = [x for x in dets if x["gsd_m"] == "0.80" and x["det_id"] in metrics["0.80"]["fp_det_ids"]]
    sheet = crop_sheet(img55, T55, gsd55, gt)
    extra = [(f'{x["det_id"]} (FP at 0.80)', float(x["lon"]), float(x["lat"])) for x in fp]
    if extra:
        sheet80 = crop_sheet(img80, T80, gsd80, [], extra)
        combined = Image.new("RGB", (sheet.width, sheet.height + sheet80.height), "black")
        combined.paste(sheet, (0, 0))
        combined.paste(sheet80, (0, sheet.height))
        sheet = combined
    sheet.save(FIGS / "crop_sheet.jpg", quality=90)

    plain, _, _ = annotate("0.55", gt, dets, metrics, draw_fixed=False)
    small = plain.resize((plain.width // 2, plain.height // 2), Image.LANCZOS)
    fixed = [(c / 2, r / 2, rad / 2, k) for c, r, rad, k in fixed_roof_circles(gt, T55, gsd55)]
    summary_png(small, metrics, FIGS / "summary.png", fixed)

    print("wrote annotated.jpg, annotated_080.jpg, crop_sheet.jpg, summary.png to", FIGS)


if __name__ == "__main__":
    main()
