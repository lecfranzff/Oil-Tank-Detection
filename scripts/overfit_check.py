"""Overfit check: the model should memorise 16 training patches; if not, the training loop is broken.

Also reports seconds per training iteration on this machine.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageDraw
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oiltank.data import TankPatches, collate, load_coco
from oiltank.evaluate import average_precision, precision_recall_at
from oiltank.model import build_model
from oiltank.train import predict, train_one_epoch

IMAGES = ROOT / "data/raw/Oil Tanks/image_patches"
OUT = ROOT / "scratch/overfit_check"


def main(epochs=40, n=16):
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    boxes = load_coco(ROOT / "data/raw/Oil Tanks/labels_coco.json")
    splits = json.load(open(ROOT / "data/splits.json"))
    names = [p for p in splits["patches"]["train"] if len(boxes[p])][:n]
    print(f"{len(names)} patches, {sum(len(boxes[p]) for p in names)} tanks")

    ds = TankPatches(IMAGES, names, boxes, train=False)       # no flips: pure memorisation
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate)
    eval_loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate)
    model = build_model()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)

    preds, gts = predict(model, eval_loader, "cpu")
    print(f"before training: AP@0.5 {average_precision(preds, gts):.4f}")
    losses, t0 = [], time.time()
    for ep in range(epochs):
        losses += train_one_epoch(model, loader, opt, "cpu", log_every=0)
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep + 1}: loss {np.mean(losses[-4:]):.4f}  ({time.time() - t0:.0f}s)")
    sec_per_iter = (time.time() - t0) / len(losses)
    preds, gts = predict(model, eval_loader, "cpu")
    print(f"after {epochs} epochs ({len(losses)} iters, {time.time() - t0:.0f}s, "
          f"{sec_per_iter:.2f} s/iter at batch 4): AP@0.5 {average_precision(preds, gts):.4f}")
    print("at conf 0.5:", precision_recall_at(preds, gts, conf=0.5))

    plt.figure(figsize=(6, 3.5))
    plt.plot(losses)
    plt.xlabel("iteration (batch of 4)")
    plt.ylabel("total loss")
    plt.title(f"Overfitting {len(names)} patches")
    plt.tight_layout()
    plt.savefig(OUT / "loss.png", dpi=120)

    # contact sheet: GT green, predictions (conf >= 0.5) red
    sheet = None
    for i, ((b, s), g) in enumerate(zip(preds, gts)):
        img = TF.to_pil_image(ds[i][0]).resize((256, 256))
        d = ImageDraw.Draw(img)
        for x1, y1, x2, y2 in g / 2:
            d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        for (x1, y1, x2, y2), sc in zip(b / 2, s):
            if sc >= 0.5:
                d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=1)
        if sheet is None:
            from PIL import Image
            sheet = Image.new("RGB", (256 * 4, 256 * 4))
        sheet.paste(img, ((i % 4) * 256, (i // 4) * 256))
    sheet.save(OUT / "predictions.jpg", quality=90)
    print(f"wrote {OUT / 'loss.png'} and {OUT / 'predictions.jpg'}")


if __name__ == "__main__":
    main()
