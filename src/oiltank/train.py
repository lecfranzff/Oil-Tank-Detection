"""A plain training loop: forward -> loss -> backward -> optimizer step."""
import time

import numpy as np
import torch

from oiltank.evaluate import average_precision
from oiltank.model import save_checkpoint


@torch.no_grad()
def predict(model, loader, device):
    """Run the model on every batch; return predictions and GT boxes as numpy arrays."""
    model.eval()
    preds, gts = [], []
    for images, targets in loader:
        outputs = model([im.to(device) for im in images])
        for o, t in zip(outputs, targets):
            preds.append((o["boxes"].cpu().numpy(), o["scores"].cpu().numpy()))
            gts.append(t["boxes"].numpy())
    return preds, gts


def train_one_epoch(model, loader, optimizer, device, log_every=20, max_iters=None):
    """One pass over the training data. Returns the loss of every step that was used,
    and how many steps were skipped because the loss was NaN or infinite."""
    model.train()
    losses = []
    skipped = 0
    t0 = time.time()
    for it, (images, targets) in enumerate(loader):
        if max_iters is not None and it >= max_iters:
            break
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)        # in train mode the model returns its losses
        loss = sum(loss_dict.values())            # classifier + box regression + RPN losses
        if not torch.isfinite(loss):              # a NaN step would ruin the weights: skip it
            skipped += 1
            parts = {name: value.item() for name, value in loss_dict.items()}
            print(f"  iter {it + 1}  skipped, non-finite loss: {parts}")
            continue
        optimizer.zero_grad()
        loss.backward()                           # gradients of the loss w.r.t. every weight
        optimizer.step()                          # nudge the weights downhill
        losses.append(loss.item())
        if log_every and (it + 1) % log_every == 0:
            print(f"  iter {it + 1}  loss {loss.item():.4f}  "
                  f"{(time.time() - t0) / (it + 1):.2f} s/iter")
    return losses, skipped


def fit(model, train_loader, val_loader, epochs, device, ckpt_path, lr=0.01):
    """Train for `epochs`, keep the checkpoint with the best validation AP@0.5."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    best_ap, history = -1.0, []
    for epoch in range(epochs):
        t0 = time.time()
        losses, skipped = train_one_epoch(model, train_loader, optimizer, device)
        preds, gts = predict(model, val_loader, device)
        ap = average_precision(preds, gts)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_ap50": ap,
                        "skipped_steps": skipped, "seconds": time.time() - t0,
                        "step_losses": losses})
        print(f"epoch {epoch + 1}: mean loss {np.mean(losses):.4f}  val AP@0.5 {ap:.4f}  "
              f"skipped steps {skipped}  ({time.time() - t0:.0f}s)")
        if ap > best_ap:
            best_ap = ap
            save_checkpoint(model, ckpt_path)
            print(f"  saved best -> {ckpt_path}")
    return history
