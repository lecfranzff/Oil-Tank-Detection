"""Measure each hand-picked tank's diameter from colour profiles on the zoom-18 image.

Why the NE-SW axis: in this image the sun casts each tank's shadow to the NW and the
off-nadir view shows the shell wall on the SE side. Both stretch the tank along NW-SE,
so we measure the width across the perpendicular axis (NE-SW, +-15 degrees), where
neither effect reaches the edge.

For each ray: tank colour = median at 0.5-0.8 R, ground colour = median at 1.2-1.4 R
(R = draft radius). The edge is the first point past 0.8 R that is closer in colour to
ground than to tank. Edge points on the NE and SW arcs are fitted with a least-squares circle
(centre and radius), re-run 5 times so the colour samples follow the refined centre.

usage: python scripts/measure_tanks.py DRAFT.csv  (prints refined rows; does not edit the file)
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oiltank.geo import lonlat_to_pixel, pixel_to_lonlat, true_gsd

LAT = 52.65455


def ray(img, cx, cy, ang_deg, r_max, step=0.5):
    """Sample RGB along a ray from (cx, cy). Angle 0 = east, 90 = north (image y is flipped)."""
    a = math.radians(ang_deg)
    ts = np.arange(0, r_max, step)
    xs = np.clip(np.round(cx + ts * math.cos(a)).astype(int), 0, img.shape[1] - 1)
    ys = np.clip(np.round(cy - ts * math.sin(a)).astype(int), 0, img.shape[0] - 1)
    return ts, img[ys, xs].astype(float)


def edge_radius(img, cx, cy, ang, R):
    ts, rgb = ray(img, cx, cy, ang, 1.45 * R)
    tank = np.median(rgb[(ts > 0.5 * R) & (ts < 0.8 * R)], axis=0)
    ground = np.median(rgb[(ts > 1.2 * R) & (ts < 1.4 * R)], axis=0)
    d_tank = np.linalg.norm(rgb - tank, axis=1)
    d_ground = np.linalg.norm(rgb - ground, axis=1)
    out = np.where((ts > 0.8 * R) & (d_ground < d_tank))[0]
    return ts[out[0]] if len(out) else float("nan")


def fit_circle(img, cx, cy, R):
    """Least-squares circle through edge points on the NE and SW arcs (shadow/lean free)."""
    angles = list(range(15, 76, 10)) + list(range(195, 256, 10))
    pts = []
    for a in angles:
        t = edge_radius(img, cx, cy, a, R)
        if not math.isnan(t):
            pts.append((cx + t * math.cos(math.radians(a)), cy - t * math.sin(math.radians(a))))
    x, y = np.array(pts).T
    # Kasa fit: x^2 + y^2 + D x + E y + F = 0 is linear in D, E, F
    A = np.column_stack([x, y, np.ones_like(x)])
    D, E, F = np.linalg.lstsq(A, -(x ** 2 + y ** 2), rcond=None)[0]
    ncx, ncy = -D / 2, -E / 2
    nR = math.sqrt(ncx ** 2 + ncy ** 2 - F)
    resid = float(np.sqrt(np.mean((np.hypot(x - ncx, y - ncy) - nR) ** 2)))
    return ncx, ncy, nR, resid


def main():
    with rasterio.open("data/hardisty/khtt_z18.tif") as s:
        img = np.transpose(s.read([1, 2, 3]), (1, 2, 0))
        T18 = s.transform
    with rasterio.open("data/hardisty/khtt_055.tif") as s:
        T055 = s.transform
    gsd = true_gsd(T18.a, LAT)
    print("tank_id,col_055,row_055,lon,lat,diameter_m,fit_rms_m")
    for r in csv.DictReader(open(sys.argv[1])):
        if r["type"] == "edge_excluded":
            continue
        lon, lat = pixel_to_lonlat(T055, float(r["col_055"]), float(r["row_055"]))
        cx, cy = lonlat_to_pixel(T18, lon, lat)
        R = float(r["diameter_m"]) / gsd / 2
        for _ in range(5):                       # refine centre and radius a few times
            cx, cy, R, resid = fit_circle(img, cx, cy, R)
        col, row = lonlat_to_pixel(T055, *pixel_to_lonlat(T18, cx, cy))
        lon, lat = pixel_to_lonlat(T18, cx, cy)
        print(f'{r["tank_id"]},{col:.1f},{row:.1f},{lon:.6f},{lat:.6f},{2 * R * gsd:.1f},{resid * gsd:.2f}')


if __name__ == "__main__":
    main()
