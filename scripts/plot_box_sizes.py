"""Histogram of training box sizes, with the Hardisty floating-roof tanks marked in pixels
at both test resolutions -> docs/figures/box_sizes.png."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oiltank.data import load_coco

LABELS = ROOT / "data/raw/Oil Tanks/labels_coco.json"
GT = ROOT / "results/hardisty/ground_truth.csv"
OUT = ROOT / "docs/figures/box_sizes.png"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    boxes = load_coco(LABELS)
    b = np.concatenate([v for v in boxes.values() if len(v)])
    sides = (b[:, 2] - b[:, 0] + b[:, 3] - b[:, 1]) / 2
    p50, p75 = np.median(sides), np.percentile(sides, 75)
    gt = [float(r["diameter_m"]) for r in csv.DictReader(open(GT))
          if r["type"] == "floating_sunken"]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hist(sides, bins=np.arange(0, 192, 4), color=BLUE, edgecolor=SURFACE, linewidth=1)
    top = ax.get_ylim()[1]
    for x, name, side in ((p50, "median", "right"), (p75, "75th pct", "left")):
        ax.axvline(x, color=MUTED, linewidth=1, linestyle="--")
        ax.text(x + (1.5 if side == "left" else -1.5), top * 0.97, f"{name} {x:.1f}",
                color=MUTED, fontsize=8, va="top", ha=side)
    for gsd, colour, marker, y in ((0.55, ORANGE, "o", 0.86), (0.80, AQUA, "s", 0.78)):
        px = [d / gsd for d in gt]
        ax.scatter(px, [top * y] * len(px), color=colour, marker=marker, s=40,
                   edgecolor=SURFACE, linewidth=1, zorder=3,
                   label=f"Hardisty floating roofs at {gsd:.2f} m/px")
    ax.set_xlabel("box side in training labels (px, mean of width and height)", color=INK)
    ax.set_ylabel("boxes", color=INK)
    ax.set_title(f"Training box sizes ({len(sides):,} boxes) vs Hardisty tank rims",
                 color=INK, loc="left", fontsize=11)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.72))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT}")
    for gsd in (0.55, 0.80):
        px = np.array([d / gsd for d in gt])
        share = float((sides >= px.min()).mean())
        print(f"  {gsd} m/px: Hardisty rims {px.min():.1f}-{px.max():.1f} px; "
              f"training boxes at least that large: {share}")


if __name__ == "__main__":
    main()
