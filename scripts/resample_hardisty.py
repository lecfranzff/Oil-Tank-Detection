"""Resample the zoom-18 Hardisty GeoTIFF to 0.55 and 0.80 m/px ground resolution.

usage: python scripts/resample_hardisty.py [--src data/hardisty/khtt_z18.tif --out-dir data/hardisty]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oiltank.geo import resample_geotiff, true_gsd

LAT = 52.65455
TARGETS = {"khtt_055.tif": 0.55, "khtt_080.tif": 0.80}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/hardisty/khtt_z18.tif"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/hardisty"))
    a = ap.parse_args()
    for name, gsd in TARGETS.items():
        dst = Path(a.out_dir) / name
        merc_res = resample_geotiff(a.src, dst, gsd, LAT)
        print(f"{dst}: ground resolution {true_gsd(merc_res, LAT)} m/px")


if __name__ == "__main__":
    main()
