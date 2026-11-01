"""Fetch one AOI from Esri World Imagery as a georeferenced GeoTIFF, and VERIFY it.

Attribution: imagery (c) Esri, Maxar, Earthstar Geographics.

The verification half matters as much as the download: Esri serves a flat grey
"Map data not yet available" tile at zooms it lacks coverage for, and that placeholder
is a perfectly valid PNG. We check per-channel statistics and reject flat tiles.
"""
from __future__ import annotations

import argparse
import io
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
TS = 256
ORIGIN = 20037508.342789244          # web-mercator half-extent, metres
UA = {"User-Agent": "Mozilla/5.0 (research; oil-tank-detection)"}


def deg2tile(lat, lon, z):
    n = 2.0 ** z
    xt = (lon + 180.0) / 360.0 * n
    la = math.radians(lat)
    yt = (1.0 - math.asinh(math.tan(la)) / math.pi) / 2.0 * n
    return xt, yt


def fetch(bbox, zoom, out_path, timeout=30, max_tiles=400):
    """bbox = [west, south, east, north] in EPSG:4326."""
    w, s, e, n = bbox
    x0f, y0f = deg2tile(s, w, zoom)                   # west, south
    x1f, y1f = deg2tile(n, e, zoom)                   # east, north
    # tile y grows southward: north edge -> smaller y
    xt0, xt1 = int(math.floor(min(x0f, x1f))), int(math.ceil(max(x0f, x1f)))
    yt0, yt1 = int(math.floor(min(y0f, y1f))), int(math.ceil(max(y0f, y1f)))
    nx, ny = xt1 - xt0, yt1 - yt0
    print(f"[fetch] zoom {zoom}: tiles x {xt0}..{xt1-1} y {yt0}..{yt1-1}  ({nx}x{ny} = {nx*ny} tiles)")
    if nx * ny > max_tiles:
        raise SystemExit(f"refusing to fetch {nx*ny} tiles; shrink the bbox")

    canvas = Image.new("RGB", (nx * TS, ny * TS))
    n_fail = 0
    for j in range(ny):
        for i in range(nx):
            url = TILE_URL.format(z=zoom, y=yt0 + j, x=xt0 + i)
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    im = Image.open(io.BytesIO(r.read())).convert("RGB")
                canvas.paste(im, (i * TS, j * TS))
            except Exception as ex:
                n_fail += 1
                print(f"[fetch]   tile {zoom}/{yt0+j}/{xt0+i} FAILED: {ex}")
        print(f"[fetch]   row {j+1}/{ny}")
    if n_fail:
        print(f"[fetch] {n_fail} tile(s) failed")

    res = 2 * ORIGIN / (TS * 2 ** zoom)               # merc metres per px
    west_m = xt0 * TS * res - ORIGIN
    north_m = ORIGIN - yt0 * TS * res
    transform = from_origin(west_m, north_m, res, res)

    arr = np.asarray(canvas).astype("uint8")          # 3-band, 8-bit, no rescaling
    with rasterio.open(out_path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=3, dtype="uint8",
                       crs="EPSG:3857", transform=transform, compress="deflate") as dst:
        for b in range(3):
            dst.write(arr[:, :, b], b + 1)
    return out_path, arr, res, n_fail


def verify(arr, res, lat):
    """Return (ok, reason, true_gsd). Rejects flat/placeholder imagery."""
    print(f"\n[verify] shape {arr.shape}  dtype {arr.dtype}  bands {arr.shape[2]}")
    ok = True
    reasons = []
    for b, name in enumerate("RGB"):
        ch = arr[:, :, b].astype(np.float64)
        print(f"[verify]   {name}: mean {ch.mean():7.2f}  std {ch.std():6.2f}  "
              f"min {ch.min():3.0f}  max {ch.max():3.0f}")
        if ch.std() < 8:
            ok = False
            reasons.append(f"{name} std {ch.std():.2f} < 8 (flat)")
    # a grey placeholder has near-identical channels everywhere
    rg = float(np.mean(np.abs(arr[:, :, 0].astype(int) - arr[:, :, 1].astype(int))))
    gb = float(np.mean(np.abs(arr[:, :, 1].astype(int) - arr[:, :, 2].astype(int))))
    print(f"[verify]   mean |R-G| {rg:.2f}   mean |G-B| {gb:.2f}  (near 0 on both => greyscale placeholder)")
    if rg < 1.0 and gb < 1.0:
        ok = False
        reasons.append("channels near-identical (greyscale placeholder)")
    uniq = len(np.unique(arr[::7, ::7, 0]))
    print(f"[verify]   distinct R values (subsampled): {uniq}")
    if uniq < 20:
        ok = False
        reasons.append(f"only {uniq} distinct values")

    true_gsd = res * math.cos(math.radians(lat))
    print(f"[verify]   merc res {res:.4f} m/px  ->  TRUE gsd at lat {lat:.4f} "
          f"= {true_gsd:.4f} m/px")
    return ok, "; ".join(reasons), true_gsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--west", type=float, required=True)
    ap.add_argument("--south", type=float, required=True)
    ap.add_argument("--east", type=float, required=True)
    ap.add_argument("--north", type=float, required=True)
    ap.add_argument("--zoom", type=int, default=18)
    ap.add_argument("--min-zoom", type=int, default=17)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    bbox = [a.west, a.south, a.east, a.north]
    lat = (a.south + a.north) / 2
    print(f"[aoi] bbox {bbox}")

    for z in range(a.zoom, a.min_zoom - 1, -1):
        path, arr, res, n_fail = fetch(bbox, z, a.out)
        ok, why, gsd = verify(arr, res, lat)
        if ok and n_fail == 0:
            Image.fromarray(arr).resize(
                (900, int(900 * arr.shape[0] / arr.shape[1]))
            ).save(str(Path(a.out).with_name(Path(a.out).stem + "_preview.jpg")), quality=90)
            print(f"\n[OK] valid imagery at zoom {z}")
            print(f"[OK] {arr.shape[1]}x{arr.shape[0]} px, true gsd {gsd:.4f} m/px")
            print(f"[OK] -> {path}")
            return 0
        print(f"\n[reject] zoom {z}: {why or 'tile fetch failures'} -- dropping a zoom level")

    print("\n[FAIL] no zoom produced valid imagery")
    return 1


if __name__ == "__main__":
    sys.exit(main())
