"""The detector: Faster R-CNN with a MobileNetV3 backbone, retargeted to one class (tank)."""
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(pretrained=True):
    # COCO-pretrained Faster R-CNN; the MobileNetV3 backbone already encodes edges and
    # textures, and the FPN handles objects at several scales.
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT" if pretrained else None,
        weights_backbone=None,      # never download separate ImageNet weights; the full
                                    # COCO model (or our checkpoint) supplies every weight
        # keep 512 px patches at native scale (default would upsample to 800 px,
        # changing apparent tank size relative to the evaluation imagery)
        min_size=512,
        max_size=512,
    )
    # replace the 91-class COCO head with background + tank
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    return model


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(path, device="cpu"):
    """Build the model and load trained weights. Fails loudly if any key mismatches."""
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))   # strict=True by default
    return model.to(device).eval()
