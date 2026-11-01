"""Labeling aid for blind ground truth: numbered overview + crop sheet with a fitted circle.

Reads a draft CSV of tank centres (pixel col/row in the 0.55 m/px image) and diameters,
draws each candidate circle on a crop of the native zoom-18 image so the fit can be
checked by eye, and prints centre lon/lat. No detector is involved.

usage: python scripts/make_gt_sheet.py DRAFT.csv OUT_DIR [--ids E1,E2,E3 --window 160 --no-circle]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oiltank.geo import lonlat_to_pixel, pixel_to_lonlat, true_gsd

LAT = 52.65455                                    # AOI centre latitude
IMG_055 = "data/hardisty/khtt_055.tif"
IMG_Z18 = "data/hardisty/khtt_z18.tif"


def read_rgb(path):
    with rasterio.open(path) as src:
        return np.transpose(src.read([1, 2, 3]), (1, 2, 0)), src.transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("draft")
    ap.add_argument("out_dir")
    ap.add_argument("--ids", default="")
    ap.add_argument("--window", type=float, default=110, help="crop side in metres")
    ap.add_argument("--cell", type=int, default=380, help="crop side on the sheet, px")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--no-circle", action="store_true")
    ap.add_argument("--name", default="crop_sheet_draft.png")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(a.draft)))
    if a.ids:
        rows = [r for r in rows if r["tank_id"] in a.ids.split(",")]
    img055, T055 = read_rgb(IMG_055)
    z18, T18 = read_rgb(IMG_Z18)
    gsd18 = true_gsd(T18.a, LAT)

    # --- numbered overview on the 0.55 image ---
    ov = Image.fromarray(img055)
    d = ImageDraw.Draw(ov)
    gsd055 = true_gsd(T055.a, LAT)
    for r in rows:
        c, rw = float(r["col_055"]), float(r["row_055"])
        rad = float(r["diameter_m"]) / gsd055 / 2
        d.ellipse([c - rad, rw - rad, c + rad, rw + rad], outline="yellow", width=3)
        d.text((c + rad + 4, rw - 8), r["tank_id"], fill="yellow")
    ov.save(out / "overview_numbered.jpg", quality=90)

    # --- crop sheet from the native zoom-18 image ---
    n = len(rows)
    ncol = min(a.cols, n)
    nrow = (n + ncol - 1) // ncol
    sheet = Image.new("RGB", (ncol * a.cell, nrow * (a.cell + 20)), "black")
    half = a.window / gsd18 / 2                   # crop half-side in z18 px
    k = a.cell / (2 * half)                        # z18 px -> sheet px
    print(f"z18 gsd {gsd18:.4f} m/px; sheet scale {gsd18 / k:.4f} m per sheet px")
    print("tank_id,lon,lat,diameter_m,type")
    for i, r in enumerate(rows):
        lon, lat = pixel_to_lonlat(T055, float(r["col_055"]), float(r["row_055"]))
        print(f'{r["tank_id"]},{lon:.6f},{lat:.6f},{r["diameter_m"]},{r["type"]}')
        cx, cy = lonlat_to_pixel(T18, lon, lat)
        x0, y0 = int(round(cx - half)), int(round(cy - half))
        size = int(round(2 * half))
        crop = np.zeros((size, size, 3), np.uint8)       # pad if the crop runs off the image
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + size, z18.shape[1]), min(y0 + size, z18.shape[0])
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = z18[sy0:sy1, sx0:sx1]
        tile = Image.fromarray(crop).resize((a.cell, a.cell), Image.LANCZOS)
        td = ImageDraw.Draw(tile)
        mid = a.cell / 2
        if not a.no_circle:
            rad = float(r["diameter_m"]) / gsd18 / 2 * k
            td.ellipse([mid - rad, mid - rad, mid + rad, mid + rad], outline=(0, 255, 0), width=1)
            td.line([mid - 6, mid, mid + 6, mid], fill=(0, 255, 0))
            td.line([mid, mid - 6, mid, mid + 6], fill=(0, 255, 0))
        # 5 m ticks along the top and left edges, long tick every 25 m
        step = 5 / gsd18 * k
        j = 0
        while j * step < a.cell:
            L = 10 if j % 5 == 0 else 4
            td.line([j * step, 0, j * step, L], fill="yellow")
            td.line([0, j * step, L, j * step], fill="yellow")
            j += 1
        sx, sy = (i % ncol) * a.cell, (i // ncol) * (a.cell + 20)
        sheet.paste(tile, (sx, sy + 20))
        ImageDraw.Draw(sheet).text((sx + 4, sy + 4),
                                   f'{r["tank_id"]}  d={r["diameter_m"]} m  {r["type"]}', fill="white")
    sheet.save(out / a.name)
    print("wrote", out / "overview_numbered.jpg", "and", out / a.name)


if __name__ == "__main__":
    main()
