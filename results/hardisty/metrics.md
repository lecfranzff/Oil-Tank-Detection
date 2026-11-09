# Keystone Hardisty (AB): ground-truthed detector evaluation

Site: Keystone Hardisty Tank Terminal area (TC Energy), Hardisty, Alberta. AOI W -111.2790, S 52.6490, E -111.2650, N 52.6600 (1.147 km²). Esri World Imagery, WorldView-3, captured 2022-06-08, fetched at zoom 18 (true GSD 0.3623 m/px) and resampled to 0.55 and 0.80 m/px. Nearest Kaggle training scene: 1834.5 km away.

## Results

| Metric | 0.55 m/px | 0.80 m/px |
|---|---|---|
| TP / FP / FN | 11 / 0 / 0 | 11 / 1 / 0 |
| Precision | 1.0000 (11/11) | 0.9166 (11/12) |
| Recall | 1.0000 (11/11) | 1.0000 (11/11) |
| F1 | 1.0000 | 0.9565 |
| Recall, floating roof sunken | 11/11 | 11/11 |
| Recall, floating roof near full | n/a (0 tanks) | n/a (0 tanks) |
| Fixed/internal-roof tanks correctly ignored | 4/4 (1.0000) | 3/4 (0.7500) |
| Diameter MAE vs GT rim (m), TPs | 7.50 | 7.50 |
| Mean signed diameter error vs GT (m) | +7.50 | +7.50 |
| Diameter MAE vs published spec (m) | 6.45 (n=8) | 6.75 (n=8) |
| Detected tanks / GT floating-roof tanks | 11 / 11 | 12 / 11 |
| Boxes: raw / after NMS / after merge | 25 / 12 / 11 | 36 / 14 / 13 |
| Detections ignored (edge tank X1) | 0 | 1 |
| CPU detection time (s) | 8.4 (35 tiles) | 1.8 (20 tiles) |

Decimals are truncated, not rounded. Matching: a detection counts as a hit when its centre is within 0.5 × the GT radius, one-to-one. Targets = floating_sunken + floating_near_full. Detection: confidence ≥ 0.5, 512 px tiles with 30% overlap, NMS IoU 0.5, then a merge pass joining boxes whose centres are closer than 0.5 × the larger box side.

Only false positive: D0.80_12 at 0.80 m/px, a thin box on the dark shadow crescent along the north rim of fixed-roof tank S2. At 0.55 m/px there were no false positives.

## Summary

Trained only on the Kaggle Oil Storage Tanks data, the detector found all 11 floating-roof tanks at this unseen Alberta terminal at 0.55 m/px with no false positives, and ignored all 4 fixed-roof tanks. At 0.80 m/px it still found all 11 but fired once on a fixed roof's shadow crescent (3/4 ignored). Its boxes wrap the whole visible tank, including the leaning shell wall and cast shadow, so they are larger than the rim on every hit (mean +7.50 m at 0.55 m/px). The near-full hypothesis could not be tested here: every floating roof in this 2022-06-08 image sits well below its rim. This is one site, one image date and 15 tanks, so these numbers show the detector transfers to this terminal; they do not estimate accuracy across Alberta.

## Model (trained for this evaluation)

- Faster R-CNN, MobileNetV3-Large FPN, COCO-pretrained, 2 classes. Trained on 1 × Tesla T4 (Kaggle) for 10 epochs, batch 4, SGD lr 0.01: 646.8 s. Best epoch by val AP@0.5: 10.
- Split by scene (70/15/15 scenes). Patches after removing 3 of 4 tank-free patches from train only: train 2567 (1153 with tanks), val 1500 (231), test 1500 (211).
- Test scenes: mAP@0.5 0.9059; at conf 0.5, TP 734 / FP 217 / FN 78, precision 0.7718, recall 0.9039. This is a patch-level metric (patches overlap within a scene).

## Per-tank (0.55 m/px)

| Tank | Score | Centre offset (m) | GT rim d (m) | Box d (m) | Box − GT (m) | Spec d (m) | Fragments merged |
|---|---|---|---|---|---|---|---|
| W1 | 0.9836 | 2.51 | 57.0 | 60.55 | +3.55 | 55.17 | 1 |
| W2 | 0.9786 | 2.75 | 66.0 | 75.22 | +9.22 | 65.23 | 1 |
| W3 | 0.5368 | 7.39 | 67.0 | 67.39 | +0.39 | 65.23 | 1 |
| W4 | 0.9859 | 3.58 | 69.5 | 78.33 | +8.83 | 69.8 | 1 |
| W5 | 0.9883 | 2.30 | 65.3 | 69.97 | +4.67 | 65.23 | 1 |
| W6 | 0.9720 | 2.64 | 70.3 | 78.13 | +7.83 | 69.8 | 1 |
| W7 | 0.9769 | 1.99 | 55.7 | 58.94 | +3.24 | 55.17 | 1 |
| W8 | 0.9909 | 3.76 | 56.7 | 63.89 | +7.19 | 55.17 | 1 |
| E1 | 0.9976 | 1.40 | 63.3 | 74.17 | +10.87 | not published | 1 |
| E2 | 0.9978 | 6.82 | 63.8 | 77.81 | +14.01 | not published | 2 |
| E3 | 0.9960 | 6.33 | 63.9 | 76.68 | +12.78 | not published | 1 |

Recomputed from these 11 rows: diameter MAE 7.50 m, mean signed error +7.50 m (box larger than rim on 11/11 tanks).

## Per-tank (0.80 m/px)

| Tank | Score | Centre offset (m) | GT rim d (m) | Box d (m) | Box − GT (m) | Spec d (m) | Fragments merged |
|---|---|---|---|---|---|---|---|
| W1 | 0.9746 | 0.98 | 57.0 | 63.23 | +6.23 | 55.17 | 1 |
| W2 | 0.9730 | 4.92 | 66.0 | 73.76 | +7.76 | 65.23 | 1 |
| W3 | 0.9322 | 3.66 | 67.0 | 70.36 | +3.36 | 65.23 | 1 |
| W4 | 0.9573 | 4.32 | 69.5 | 79.08 | +9.58 | 69.8 | 1 |
| W5 | 0.9846 | 1.92 | 65.3 | 72.64 | +7.34 | 65.23 | 1 |
| W6 | 0.9809 | 1.82 | 70.3 | 78.54 | +8.24 | 69.8 | 1 |
| W7 | 0.9623 | 4.09 | 55.7 | 58.19 | +2.49 | 55.17 | 1 |
| W8 | 0.9601 | 3.66 | 56.7 | 59.01 | +2.31 | 55.17 | 1 |
| E1 | 0.9901 | 5.98 | 63.3 | 76.11 | +12.81 | not published | 1 |
| E2 | 0.9943 | 6.30 | 63.8 | 73.64 | +9.84 | not published | 1 |
| E3 | 0.9865 | 5.52 | 63.9 | 76.5 | +12.60 | not published | 2 |

Recomputed from these 11 rows: diameter MAE 7.50 m, mean signed error +7.50 m (box larger than rim on 11/11 tanks).

## Sweeps (same model, same ground truth)

Made by scripts/sweep.py (sweeps.json). These were run after the main result to see how sensitive it is; the reported setting stays conf 0.5. The conf 0.5 rows below reproduce the main table exactly.

**Confidence threshold**

| Threshold | 0.55 m/px TP / FP / FN | Fixed roofs ignored | 0.80 m/px TP / FP / FN | Fixed roofs ignored |
|---|---|---|---|---|
| 0.05 | 11 / 0 / 0 | 4/4 | 11 / 3 / 0 | 2/4 |
| 0.1 | 11 / 0 / 0 | 4/4 | 11 / 1 / 0 | 3/4 |
| 0.2 | 11 / 0 / 0 | 4/4 | 11 / 1 / 0 | 3/4 |
| 0.3 | 11 / 0 / 0 | 4/4 | 11 / 1 / 0 | 3/4 |
| 0.4 | 11 / 0 / 0 | 4/4 | 11 / 1 / 0 | 3/4 |
| 0.5 | 11 / 0 / 0 | 4/4 | 11 / 1 / 0 | 3/4 |
| 0.6 | 10 / 0 / 1 | 4/4 | 11 / 1 / 0 | 3/4 |
| 0.7 | 10 / 0 / 1 | 4/4 | 11 / 0 / 0 | 4/4 |
| 0.8 | 10 / 0 / 1 | 4/4 | 11 / 0 / 0 | 4/4 |
| 0.9 | 10 / 0 / 1 | 4/4 | 11 / 0 / 0 | 4/4 |
| 0.95 | 10 / 0 / 1 | 4/4 | 10 / 0 / 1 | 4/4 |

**Resolution** (conf 0.5)

| Ground resolution (m/px) | Image (px) | Tiles | TP / FP / FN | Fixed roofs ignored | Missed tanks | Mean box − rim (m) |
|---|---|---|---|---|---|---|
| 0.4000 | 2550 × 3246 | 63 | 8 / 0 / 3 | 4/4 | W3, W4, W6 | −0.29 |
| 0.4800 | 2125 × 2705 | 48 | 11 / 1 / 0 | 4/4 | none | +7.62 |
| 0.5499 | 1855 × 2361 | 35 | 11 / 0 / 0 | 4/4 | none | +7.50 |
| 0.6399 | 1594 × 2029 | 30 | 11 / 0 / 0 | 4/4 | none | +7.33 |
| 0.7199 | 1417 × 1803 | 20 | 11 / 0 / 0 | 4/4 | none | +5.73 |
| 0.8000 | 1275 × 1623 | 20 | 11 / 1 / 0 | 3/4 | none | +7.50 |
| 0.9003 | 1133 × 1443 | 12 | 11 / 1 / 0 | 3/4 | none | +6.39 |
| 1.0000 | 1020 × 1298 | 12 | 11 / 0 / 0 | 4/4 | none | +6.56 |
| 1.2001 | 850 × 1082 | 6 | 11 / 0 / 0 | 4/4 | none | +6.11 |

- All 11 floating roofs are found at every resolution from 0.48 to 1.20 m/px. At 0.40 m/px the three largest (W3, W4, W6; 167–176 px across) are missed: only 2 of the 5,820 training boxes are that large.
- At 0.72 m/px, the measured resolution of 95 training scenes, the result is 11 / 0 / 0 with all 4 fixed roofs ignored.
- False positives, checked by eye: at 0.80 m/px the shadow crescent inside fixed-roof tank S2's rim (at every threshold up to 0.6; at 0.05 the pipe below and fixed-roof S1 add one each); at 0.90 m/px the same kind of rim shadow on fixed-roof S1; at 0.48 m/px a white bent pipe about 170 m from any tank, near the south edge of the image.
- At 0.55 m/px any threshold above 0.5 loses W3 (score 0.5368). At 0.80 m/px thresholds 0.7–0.9 remove the S2 false positive and keep all 11 tanks; at 0.95 one tank is lost (W3).
- One image and 15 tanks: these show sensitivity at this site, not a tuned operating point.

## Limitations and notes

- One site, one image (2022-06-08), 11 floating-roof + 4 fixed-roof tanks. Small numbers: one error changes precision or recall by several points.
- Ground truth was labelled by one person (types confirmed by the author) before the model was trained. SHA-256 of ground_truth.csv is in ground_truth.sha256.
- **Rim diameters were read with the published spec sizes already known, so agreement with the spec is supporting evidence, not independent validation.** Spec sizes (Matrix Service, Phase C: 3 × 55.17 m, 3 × 65.23 m, 2 × 69.80 m) were assigned to W1–W8 by their measured size group. E1–E3 have no published diameter.
- Rim diameters are ±3 m (by eye) for floating roofs; the fixed roofs come from a circle fit.
- Tank count: 11 floating-roof tanks matches 8 Phase C tanks (Matrix Service) plus 3 earlier TransCanada tanks (RBN Energy), but that published count was not verified for this exact fence line.
- Near-full recall is n/a (0 tanks), so the near-full hypothesis is untested.
- Fragile hit: W3 scored 0.5368 at 0.55 m/px, just above the 0.5 threshold (all other hits ≥ 0.97); a slightly higher threshold would have missed it.
- The edge tank X1 (cut by the AOI boundary) is excluded; detections centred inside it are ignored (0 at 0.55 m/px, 1 at 0.80 m/px).
- Training settings (10 epochs, batch 4, lr 0.01) were chosen, not tuned; the best epoch was picked by val AP@0.5, and val AP was still rising at epoch 10.
- NaN training loss appeared in 92 of 330 logged 20-iteration windows from epoch 2 onward. Nothing skipped those steps: training ran in plain fp32 with no GradScaler, autocast or isfinite check, so optimizer.step() ran on every one of the 6,420 iterations (0 skipped). The number of individual NaN iterations is unknown, because only 20-iteration means were logged. Because one step with NaN gradients would have made the weights NaN for good, and all 284 checkpoint tensors are finite, the NaN was most likely in a loss term whose gradient stayed finite (an inference, not verified). A forward-only re-check of all 2,567 train patches with the final weights (no flips, no updates) gave 0 non-finite losses, so the cause was not reproduced and remains unverified. The checkpoint was kept as-is. After training, two guards were added to the code (not used for this checkpoint): boxes are clipped to the patch, and any step with a non-finite loss is skipped and counted. The same change fixed a bug where random flips edited the stored labels in place; with the 2 data-loader workers used on Kaggle every epoch started from unedited labels (checked), so this checkpoint was not affected.
- Esri imagery is a mosaic; it was resampled by area averaging to 0.55 and 0.80 m/px. 0.55 m/px was the assumed training resolution; measured afterwards (docs/dataset.md), 95 of the 100 training scenes are 0.72 m/px and 5 are 0.48 m/px, so 0.80 m/px is the closer match.

## Sources

- TC Energy, Hardisty Geographic Response Plan (legal land description NW-29-042-09-W4 / SW-32-042-09-W4): https://www.tcenergy.com/siteassets/pdfs/sustainability/safety/emergency-response-plans/tcenergy-hardisty-geographic-response-plan.pdf
- Matrix Service, KHTT Phase C project sheet: https://www.matrixservice.com/wp-content/uploads/2021/01/MS_TransCanada_KHTT-Phase-C_Hardisty-Alberta-CAN.pdf
- RBN Energy, TransCanada's three Hardisty tanks: https://rbnenergy.com/storing-crude-oil-in-harmony-transcanada-and-gibson-at-hardisty
- Imagery capture date: Esri World_Imagery MapServer identify at the AOI centre (Vantor Vivid, WV03, 2022-06-08).
- Training data: Kaggle towardsentropy/oil-storage-tanks.
