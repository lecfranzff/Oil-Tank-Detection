"""Measure the ground resolution of Kaggle training scenes by matching them to Esri imagery.

The Kaggle data has no published resolution. For each scene: rebuild the 4800 px image
from its 100 patches, fetch Esri World Imagery around the scene centre (known m/px), then
try many candidate m/px values. For each one, shrink the scene to Esri's pixel size and
slide it over the Esri image; the candidate with the best match is the scene's resolution.
Matching uses normalised cross-correlation of edge strength, so colour and season
differences between the two images matter less.

usage: python scripts/measure_training_gsd.py [scene ids, default 01 40 50 70 16 13 28 36]
(about 20 min on CPU for the 8 default scenes; Esri tiles are cached in scratch/gsd_check)
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oiltank.fetch_esri import fetch
from oiltank.geo import parse_dms, true_gsd

RAW = ROOT / "data/raw/Oil Tanks"
CACHE = ROOT / "scratch/gsd_check"
ZOOM = 16
CANDIDATES = np.arange(0.30, 0.95, 0.0025)


def stitch(sid):
    """Paste the 10 x 10 patches of one scene back into a 4800 x 4800 grey image."""
    big = np.zeros((4800, 4800), np.float32)
    for r in range(10):
        for c in range(10):
            y, x = round(r * 4288 / 9), round(c * 4288 / 9)
            patch = Image.open(RAW / "image_patches" / f"{sid}_{r}_{c}.jpg").convert("L")
            big[y:y + 512, x:x + 512] = np.asarray(patch, np.float32)
    return Image.fromarray(big)


def edges(a):
    gy, gx = np.gradient(a)
    return np.hypot(gx, gy)


def slide_dot(image, template):
    """Dot product of the template with every same-sized window of the image.

    Looping over every position would take hours for these image sizes. Multiplying in
    the frequency domain (np.fft) gives the result for all positions at once.
    """
    H, W = image.shape
    h, w = template.shape
    flipped = template[::-1, ::-1]
    full = np.fft.irfft2(np.fft.rfft2(image) * np.fft.rfft2(flipped, (H, W)), (H, W))
    return full[h - 1:, w - 1:]                  # keep positions where the template fits


def window_sums(a, h, w):
    """Sum of every h x w window of `a`, from a table of running totals (4 lookups each)."""
    totals = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return totals[h:, w:] - totals[:-h, w:] - totals[h:, :-w] + totals[:-h, :-w]


def best_match(image, template):
    """Best normalised cross-correlation (1 = identical pattern) of template over image."""
    h, w = template.shape
    t = template - template.mean()
    t /= np.sqrt((t ** 2).sum())                # template now has mean 0 and length 1
    dot = slide_dot(image, t)
    n = h * w
    spread = window_sums(image ** 2, h, w) - window_sums(image, h, w) ** 2 / n
    return float((dot / np.sqrt(np.maximum(spread, 1e-6))).max())


def measure(scene):
    lat, lon = parse_dms(scene["Latitude"]), parse_dms(scene["Longitude"])
    sid = scene["Filename"][:2]
    half_lat = 2.6 / 111.0                                   # about 2.6 km each way
    half_lon = half_lat / math.cos(math.radians(lat))
    tif = CACHE / f"esri_{sid}_z{ZOOM}.tif"
    if not tif.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        fetch([lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat], ZOOM, tif, max_tiles=600)
    with rasterio.open(tif) as src:
        esri = edges(src.read().mean(0).astype(np.float32))
        esri_gsd = true_gsd(src.res[0], lat)
    img = stitch(sid)
    scores = []
    for g in CANDIDATES:
        n = round(4800 * g / esri_gsd)
        if n < min(esri.shape):
            tpl = edges(np.asarray(img.resize((n, n), Image.BILINEAR), np.float32))
            scores.append((best_match(esri, tpl), g))
    scores.sort(reverse=True)
    runner_up = max(s for s, g in scores if abs(g - scores[0][1]) > 0.05)
    return scores[0][1], scores[0][0], runner_up


def main():
    ids = sys.argv[1:] or ["01", "40", "50", "70", "16", "13", "28", "36"]
    rows = {r["Filename"][:2]: r for r in csv.DictReader(open(RAW / "large_image_data.csv", encoding="utf-8-sig"))}
    print("scene  eye alt (m)  measured m/px  match  best match > 0.05 m/px away  place")
    for sid in ids:
        r = rows[sid]
        g, score, runner_up = measure(r)
        print(f"{sid:>5}  {r['Eye Alt (m)']:>11}  {g:13.4f}  {score:.3f}  {runner_up:26.3f}  {r['General Location']}")


if __name__ == "__main__":
    main()
