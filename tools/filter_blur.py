"""Copy sharp images using the Laplacian variance of a center crop."""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tools.data_utils import image_paths, validate_output


def blur_score(image, crop_size=300):
    height, width = image.shape[:2]
    if min(height, width) < crop_size:
        return 0.0
    top, left = (height - crop_size) // 2, (width - crop_size) // 2
    crop = image[top : top + crop_size, left : left + crop_size]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def filter_images(source, output, threshold=20, crop_size=300):
    if threshold < 0 or crop_size <= 0:
        raise ValueError(
            "Threshold must be nonnegative and crop size must be positive."
        )
    source, output = validate_output(source, output)
    kept = 0
    for path in tqdm(image_paths(source), desc="Checking sharpness"):
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        if (
            min(image.shape[:2]) >= crop_size
            and blur_score(image, crop_size) >= threshold
        ):
            destination = output / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            kept += 1
    print(f"Copied {kept} sharp images to {output}; source images are unchanged.")
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=20)
    parser.add_argument("--crop-size", type=int, default=300)
    args = parser.parse_args()
    try:
        filter_images(args.source, args.output, args.threshold, args.crop_size)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
