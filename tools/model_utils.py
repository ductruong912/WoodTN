"""Model construction and checkpoint contracts shared by evaluation tools."""

from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

ARCHITECTURES = ("efficientnet_v2_s", "mvitv2_small", "vit_b_16")
DEFAULT_CLASSES = Path(__file__).resolve().parents[1] / "classes.txt"


def read_classes(path):
    classes = Path(path).read_text(encoding="utf-8").splitlines()
    if not classes or any(not name.strip() for name in classes):
        raise ValueError("Class file must have one nonempty label per line.")
    if len(set(classes)) != len(classes):
        raise ValueError("Class file contains duplicate labels.")
    return classes


def build_model(architecture, num_classes):
    if architecture == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == "mvitv2_small":
        import timm

        model = timm.create_model(
            architecture, pretrained=False, num_classes=num_classes
        )
    elif architecture == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model


def load_model(checkpoint_path, architecture, classes, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        if checkpoint.get("classes") is not None and checkpoint["classes"] != classes:
            raise ValueError("Checkpoint class order does not match the class file.")
        state = checkpoint["model_state"]
    else:
        state = checkpoint
    model = build_model(architecture, len(classes))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def evaluation_transform(architecture, image_size=224, resize_size=None):
    if image_size <= 0:
        raise ValueError("Image size must be positive.")
    if architecture in ("vit_b_16", "mvitv2_small") and image_size != 224:
        raise ValueError(f"{architecture} requires the 224px configuration used here.")
    if resize_size is None:
        resize_size = 256 if architecture == "mvitv2_small" else int(image_size * 1.15)
    if resize_size < image_size:
        raise ValueError("Resize size must be at least the crop size.")
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
