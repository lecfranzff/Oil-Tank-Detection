"""The detector: Faster R-CNN with a MobileNetV3 backbone, retargeted to one class (tank)."""
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(pretrained=True):
    # 1. Start from a Faster R-CNN already trained on COCO (everyday photos, 91 classes).
    #    Its backbone (MobileNetV3, a small fast CNN) already knows edges, circles and
    #    textures, so we only have to teach it what an oil tank looks like.
    #    FPN = feature pyramid network: lets it find both small and large objects.
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT" if pretrained else None,
        weights_backbone=None,      # never download separate ImageNet weights; the full
                                    # COCO model (or our checkpoint) supplies every weight
        # 2. By default torchvision resizes every image so its short side is 800 px.
        #    Our patches are 512 px; upscaling them would change the apparent tank size
        #    (and be slower). Fixing min/max size at 512 keeps 1 image px = 1 input px.
        min_size=512,
        max_size=512,
    )
    # 3. Replace the final layer ("box predictor"). COCO's has 91 class scores; ours
    #    needs 2: class 0 = background, class 1 = tank. It outputs a score per class
    #    and a box adjustment per class for every candidate region.
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
