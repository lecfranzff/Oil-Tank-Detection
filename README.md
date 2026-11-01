# Oil Tank Detection

Can a detector trained only on the Kaggle Oil Storage Tanks dataset find and count the
external floating-roof tanks at a real, unseen Alberta terminal, and does it correctly
ignore fixed-roof tanks?

![summary](docs/figures/summary.png)

## Result (Keystone Hardisty terminal, Hardisty AB; Esri imagery captured 2022-06-08)

| | 0.55 m/px | 0.80 m/px |
|---|---|---|
| Floating-roof tanks found (recall) | 11/11 | 11/11 |
| False positives | 0 | 1 |
| Fixed-roof tanks correctly ignored | 4/4 | 3/4 |

All numbers, the method and the limitations are in
[scratch/hardisty/metrics.md](scratch/hardisty/metrics.md). In short: this is one site,
one image and 15 tanks, so it shows the model transfers to this terminal, not how
accurate it is across Alberta. Coverage of all Alberta hubs is work in progress.

## Method

1. **Site and imagery**: located the terminal from TC Energy's legal land description and
   cross-checked tank sizes against the published Phase C specification. Fetched Esri World
   Imagery at zoom 18 (0.36 m/px) and resampled to 0.55 and 0.80 m/px.
2. **Blind ground truth**: labelled every tank (centre, rim diameter, roof type) by hand
   *before* any model was trained (`ground_truth.csv`, with its SHA-256 hash).
3. **Model**: Faster R-CNN with a COCO-pretrained MobileNetV3-FPN backbone, fine-tuned on
   Kaggle patches split by **scene** (70/15/15), so overlapping patches never leak between
   splits. Test mAP@0.5 on unseen Kaggle scenes: 0.9059.
4. **Detection**: 512 px tiles with 30% overlap, one global NMS, then a merge pass that
   joins fragments of large tanks.
5. **Scoring**: one-to-one centre matching (within 0.5 × GT radius) against the ground truth.

## How to run

Requires Python with torch, torchvision, rasterio, numpy, pillow, matplotlib, pytest and the kaggle CLI.

```bash
python scripts/prepare_data.py          # label stats + scene split -> data/splits.json
python scripts/build_kaggle_job.py      # bundle training into one Kaggle GPU script
kaggle kernels push -p scratch/kaggle_train   # train; download best.pth to checkpoints/
python -m oiltank.fetch_esri --west -111.2790 --south 52.6490 --east -111.2650 \
       --north 52.6600 --zoom 18 --out data/hardisty/khtt_z18.tif   # (PYTHONPATH=src)
python scripts/run_detection.py         # detections at 0.55 and 0.80 m/px
python scripts/score.py                 # metrics.json
python scripts/write_metrics.py         # metrics.md
python scripts/make_figures.py          # annotated images, crop sheet, summary.png
python -m pytest
```

Resampling to 0.55/0.80 m/px uses `oiltank.geo.resample_geotiff`. The labelling aids are
`scripts/make_gt_sheet.py` and `scripts/measure_tanks.py`.

## Layout

```
src/oiltank/   data, model, training, detection, metrics, geo helpers, Esri fetcher
scripts/       one script per step
tests/         pytest (scene-split leakage, AP, IoU, merge pass, geo, checkpoint reload)
scratch/hardisty/  ground truth, detections, metrics.md, figures
```

## Future work

Estimate how full each floating-roof tank is from the shadow the rim casts on the roof
(inner vs outer shadow, S_in / S_out). This is not implemented yet.

Imagery © Esri, Vantor, Earthstar Geographics. Training data: Kaggle `towardsentropy/oil-storage-tanks`.
