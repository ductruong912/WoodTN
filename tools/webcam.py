"""Run a local ONNX webcam demo with deterministic preprocessing."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from tools.model_utils import DEFAULT_CLASSES, evaluation_transform, read_classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("efficientnet_v2_s", "mvitv2_small"), required=True
    )
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int)
    args = parser.parse_args()
    classes = read_classes(args.classes)
    transform = evaluation_transform(
        args.architecture, args.image_size, args.resize_size
    )
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    camera = cv2.VideoCapture(args.camera)
    try:
        if not camera.isOpened():
            raise RuntimeError("Cannot open camera.")
        print("Press Esc to exit.")
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Cannot read a frame from the camera.")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = transform(image).unsqueeze(0).numpy()
            logits = session.run(None, {input_name: tensor})[0]
            if logits.shape != (1, len(classes)):
                raise ValueError("Model output does not match the class file.")
            label = classes[int(np.argmax(logits[0]))]
            cv2.putText(
                frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.imshow("WoodTN", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
