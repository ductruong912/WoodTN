"""Export an EfficientNetV2-S or MViTv2 state dictionary to ONNX."""

import argparse
from pathlib import Path

import onnx
import torch

from tools.model_utils import DEFAULT_CLASSES, load_model, read_classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("efficientnet_v2_s", "mvitv2_small"), required=True
    )
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    if args.image_size <= 0 or (
        args.architecture == "mvitv2_small" and args.image_size != 224
    ):
        parser.error(
            "Image size must be positive; this MViTv2 configuration requires 224."
        )
    if args.output.exists():
        parser.error("Output already exists; choose a new file.")
    classes = read_classes(args.classes)
    model = load_model(args.checkpoint, args.architecture, classes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, args.image_size, args.image_size)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(args.output))
    print(
        f"Exported and checked {args.output}; preprocessing is external to the model."
    )


if __name__ == "__main__":
    main()
