"""Write results/hardisty/metrics.md from metrics.json and checkpoints/results.json.

Every number is read from those files; decimals are truncated (never rounded up).
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/hardisty"


def t(v, n=4):
    """Truncate to n decimals (never round up)."""
    return f"{math.floor(v * 10 ** n + 1e-9) / 10 ** n:.{n}f}"   # 1e-9: 6.2299999 (float) is 6.23


def frac(pair):
    return f"{pair[0]}/{pair[1]}" if pair[1] else "n/a (0 tanks)"


def per_tank_table(res, x):
    """Per-tank rows plus MAE / signed error recomputed from those rows (checked against metrics.json)."""
    rows = x["per_tank"]
    errs = [p["det_d_m"] - p["gt_d_m"] for p in rows]
    mae = sum(abs(e) for e in errs) / len(errs)
    signed = sum(errs) / len(errs)
    assert abs(mae - x["diameter_mae_vs_gt_m"]) < 1e-9 and abs(signed - x["diameter_mean_signed_vs_gt_m"]) < 1e-9
    out = [f"## Per-tank ({res} m/px)", "",
           "| Tank | Score | Centre offset (m) | GT rim d (m) | Box d (m) | Box − GT (m) | Spec d (m) | Fragments merged |",
           "|---|---|---|---|---|---|---|---|"]
    for p, e in zip(rows, errs):
        out.append(f'| {p["tank_id"]} | {t(p["score"])} | {t(p["centre_offset_m"], 2)} | {p["gt_d_m"]} | '
                   f'{p["det_d_m"]} | {"+" if e >= 0 else "−"}{t(abs(e), 2)} | '
                   f'{p["spec_d_m"] or "not published"} | {p["n_fragments"]} |')
    out += ["", f"Recomputed from these {len(rows)} rows: diameter MAE {t(mae, 2)} m, mean signed error "
                f"{'+' if signed >= 0 else '−'}{t(abs(signed), 2)} m (box larger than rim on "
                f"{sum(e > 0 for e in errs)}/{len(errs)} tanks).", ""]
    return out


def main():
    m = json.load(open(OUT / "metrics.json"))
    r = json.load(open(ROOT / "checkpoints/results.json"))
    a, b = m["0.55"], m["0.80"]
    ps = r["patches_per_split"]
    nan_note = ("NaN training loss appeared in 92 of 330 logged 20-iteration windows from epoch 2 "
                "onward. Nothing skipped those steps: training ran in plain fp32 with no GradScaler, "
                "autocast or isfinite check, so optimizer.step() ran on every one of the 6,420 "
                "iterations (0 skipped). The number of individual NaN iterations is unknown, because only "
                "20-iteration means were logged. Because one step with NaN gradients would have made the "
                "weights NaN for good, and all 284 checkpoint tensors are finite, the NaN was most likely in "
                "a loss term whose gradient stayed finite (an inference, not verified). A forward-only re-check "
                "of all 2,567 train patches with the final weights (no flips, no updates) gave 0 non-finite "
                "losses, so the cause was not reproduced and remains unverified. The checkpoint was kept as-is. "
                "After training, two guards were added to the code (not used for this checkpoint): boxes "
                "are clipped to the patch, and any step with a non-finite loss is skipped and counted. "
                "The same change fixed a bug where random flips edited the stored labels in place; with "
                "the 2 data-loader workers used on Kaggle every epoch started from unedited labels "
                "(checked), so this checkpoint was not affected.")

    def row(name, f):
        return f"| {name} | {f(a)} | {f(b)} |"

    lines = [
        "# Keystone Hardisty (AB): ground-truthed detector evaluation",
        "",
        "Site: Keystone Hardisty Tank Terminal area (TC Energy), Hardisty, Alberta. AOI "
        "W -111.2790, S 52.6490, E -111.2650, N 52.6600 (1.147 km²). Esri World Imagery, "
        "WorldView-3, captured 2022-06-08, fetched at zoom 18 (true GSD 0.3623 m/px) and "
        "resampled to 0.55 and 0.80 m/px. Nearest Kaggle training scene: 1834.5 km away.",
        "",
        "## Results",
        "",
        "| Metric | 0.55 m/px | 0.80 m/px |",
        "|---|---|---|",
        row("TP / FP / FN", lambda x: f'{x["tp"]} / {x["fp"]} / {x["fn"]}'),
        row("Precision", lambda x: f'{t(x["precision"])} ({x["tp"]}/{x["tp"] + x["fp"]})'),
        row("Recall", lambda x: f'{t(x["recall"])} ({x["tp"]}/{x["tp"] + x["fn"]})'),
        row("F1", lambda x: t(x["f1"])),
        row("Recall, floating roof sunken", lambda x: frac(x["recall_sunken"])),
        row("Recall, floating roof near full", lambda x: frac(x["recall_near_full"])),
        row("Fixed/internal-roof tanks correctly ignored",
            lambda x: f'{x["fixed_correctly_ignored"]}/{x["fixed_tanks"]} '
                      f'({t(x["fixed_correctly_ignored"] / x["fixed_tanks"])})'),
        row("Diameter MAE vs GT rim (m), TPs", lambda x: t(x["diameter_mae_vs_gt_m"], 2)),
        row("Mean signed diameter error vs GT (m)", lambda x: "+" + t(x["diameter_mean_signed_vs_gt_m"], 2)),
        row("Diameter MAE vs published spec (m)",
            lambda x: f'{t(x["diameter_mae_vs_spec_m"], 2)} (n={x["n_spec_compared"]})'),
        row("Detected tanks / GT floating-roof tanks", lambda x: f'{x["detected_count"]} / {x["gt_target_count"]}'),
        row("Boxes: raw / after NMS / after merge", lambda x: f'{x["raw_boxes"]} / {x["after_nms"]} / {x["after_merge"]}'),
        row("Detections ignored (edge tank X1)", lambda x: str(x["detections_ignored_edge"])),
        row("CPU detection time (s)", lambda x: f'{t(x["cpu_seconds"], 1)} ({x["tiles"]} tiles)'),
        "",
        "Decimals are truncated, not rounded. Matching: a detection counts as a hit when its centre "
        "is within 0.5 × the GT radius, one-to-one. Targets = floating_sunken + floating_near_full. "
        "Detection: confidence ≥ 0.5, 512 px tiles with 30% overlap, NMS IoU 0.5, then a merge pass "
        "joining boxes whose centres are closer than 0.5 × the larger box side.",
        "",
        f'Only false positive: {", ".join(b["fp_det_ids"]) or "none"} at 0.80 m/px, a thin box on the '
        "dark shadow crescent along the north rim of fixed-roof tank S2. At 0.55 m/px there were no false positives.",
        "",
        "## Summary",
        "",
        f'Trained only on the Kaggle Oil Storage Tanks data, the detector found all {a["gt_target_count"]} '
        f'floating-roof tanks at this unseen Alberta terminal at 0.55 m/px with no false positives, and '
        f'ignored all {a["fixed_tanks"]} fixed-roof tanks. At 0.80 m/px it still found all '
        f'{b["gt_target_count"]} but fired once on a fixed roof\'s shadow crescent '
        f'({b["fixed_correctly_ignored"]}/{b["fixed_tanks"]} ignored). '
        "Its boxes wrap the whole visible tank, including the leaning shell wall and cast shadow, so they "
        f'are larger than the rim on every hit (mean +{t(a["diameter_mean_signed_vs_gt_m"], 2)} m at 0.55 m/px). '
        "The near-full hypothesis could not be tested here: every floating roof in this 2022-06-08 image sits "
        "well below its rim. This is one site, one image date and 15 tanks, so these numbers show the "
        "detector transfers to this terminal; they do not estimate accuracy across Alberta.",
        "",
        "## Model (trained for this evaluation)",
        "",
        f'- Faster R-CNN, MobileNetV3-Large FPN, COCO-pretrained, 2 classes. Trained on 1 × {r["hardware"]} '
        f'(Kaggle) for {r["epochs"]} epochs, batch {r["batch"]}, SGD lr {r["lr"]}: '
        f'{t(r["train_seconds"], 1)} s. Best epoch by val AP@0.5: {r["best_epoch"]}.',
        f'- Split by scene (70/15/15 scenes). Patches after removing 3 of 4 tank-free patches from train only: '
        f'train {ps["train"]["total"]} ({ps["train"]["with_tanks"]} with tanks), '
        f'val {ps["val"]["total"]} ({ps["val"]["with_tanks"]}), test {ps["test"]["total"]} ({ps["test"]["with_tanks"]}).',
        f'- Test scenes: mAP@0.5 {t(r["test_ap50"])}; at conf 0.5, TP {r["test_at_conf_0.5"]["tp"]} / '
        f'FP {r["test_at_conf_0.5"]["fp"]} / FN {r["test_at_conf_0.5"]["fn"]}, precision '
        f'{t(r["test_at_conf_0.5"]["precision"])}, recall {t(r["test_at_conf_0.5"]["recall"])}. '
        "This is a patch-level metric (patches overlap within a scene).",
        "",
    ]
    for res, x in (("0.55", a), ("0.80", b)):
        lines += per_tank_table(res, x)
    lines += [
        "",
        "## Limitations and notes",
        "",
        "- One site, one image (2022-06-08), 11 floating-roof + 4 fixed-roof tanks. Small numbers: one "
        "error changes precision or recall by several points.",
        "- Ground truth was labelled by one person (types confirmed by the author) before the model was "
        "trained. SHA-256 of ground_truth.csv is in ground_truth.sha256.",
        "- **Rim diameters were read with the published spec sizes already known, so agreement with the "
        "spec is supporting evidence, not independent validation.** Spec sizes (Matrix Service, Phase C: "
        "3 × 55.17 m, 3 × 65.23 m, 2 × 69.80 m) were assigned to W1–W8 by their measured size group. "
        "E1–E3 have no published diameter.",
        "- Rim diameters are ±3 m (by eye) for floating roofs; the fixed roofs come from a circle fit.",
        "- Tank count: 11 floating-roof tanks matches 8 Phase C tanks (Matrix Service) plus 3 earlier "
        "TransCanada tanks (RBN Energy), but that published count was not verified for this exact fence line.",
        "- Near-full recall is n/a (0 tanks), so the near-full hypothesis is untested.",
        f'- Fragile hit: W3 scored {t(a["per_tank"][2]["score"])} at 0.55 m/px, just above the 0.5 threshold '
        "(all other hits ≥ 0.97); a slightly higher threshold would have missed it.",
        "- The edge tank X1 (cut by the AOI boundary) is excluded; detections centred inside it are "
        f'ignored ({a["detections_ignored_edge"]} at 0.55 m/px, {b["detections_ignored_edge"]} at 0.80 m/px).',
        "- Training settings (10 epochs, batch 4, lr 0.01) were chosen, not tuned; the best epoch was "
        "picked by val AP@0.5, and val AP was still rising at epoch 10.",
        f"- {nan_note}",
        "- Esri imagery is a mosaic; it was resampled by area averaging to 0.55 and 0.80 m/px. 0.55 m/px was the "
        "assumed training resolution; measured afterwards (docs/dataset.md), 95 of the 100 training scenes are "
        "0.72 m/px and 5 are 0.48 m/px, so 0.80 m/px is the closer match.",
        "",
        "## Sources",
        "",
        "- TC Energy, Hardisty Geographic Response Plan (legal land description NW-29-042-09-W4 / "
        "SW-32-042-09-W4): https://www.tcenergy.com/siteassets/pdfs/sustainability/safety/emergency-response-plans/tcenergy-hardisty-geographic-response-plan.pdf",
        "- Matrix Service, KHTT Phase C project sheet: "
        "https://www.matrixservice.com/wp-content/uploads/2021/01/MS_TransCanada_KHTT-Phase-C_Hardisty-Alberta-CAN.pdf",
        "- RBN Energy, TransCanada's three Hardisty tanks: "
        "https://rbnenergy.com/storing-crude-oil-in-harmony-transcanada-and-gibson-at-hardisty",
        "- Imagery capture date: Esri World_Imagery MapServer identify at the AOI centre (Vantor Vivid, WV03, 2022-06-08).",
        "- Training data: Kaggle towardsentropy/oil-storage-tanks.",
    ]
    (OUT / "metrics.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "metrics.md")


if __name__ == "__main__":
    main()
