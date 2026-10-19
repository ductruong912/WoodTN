"""Copy one image per perceptual-hash group, preserving class directories."""

import argparse
import shutil
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

from tools.data_utils import image_paths, validate_output


def deduplicate(source, output, method="phash", threshold=6):
    if threshold < 0:
        raise ValueError("Threshold must be nonnegative.")
    source, output = validate_output(source, output)
    hash_image = {"phash": imagehash.phash, "dhash": imagehash.dhash}[method]
    kept = 0
    # Compare within each class; similar textures in different classes retain labels.
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        groups = []
        for path in tqdm(image_paths(class_dir), desc=class_dir.name):
            try:
                with Image.open(path) as image:
                    image_hash = hash_image(image.convert("RGB"))
                    resolution = image.width * image.height
            except (OSError, ValueError) as error:
                raise ValueError(f"Cannot read image: {path}") from error
            for group in groups:
                if image_hash - group[0] <= threshold:
                    if resolution > group[2]:
                        group[1], group[2] = path, resolution
                    break
            else:
                groups.append([image_hash, path, resolution])
        for _, path, _ in groups:
            destination = output / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            kept += 1
    print(f"Copied {kept} images to {output}")
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Directory with one subfolder per class"
    )
    parser.add_argument("output", type=Path, help="New directory for selected images")
    parser.add_argument("--method", choices=("phash", "dhash"), default="phash")
    parser.add_argument("--threshold", type=int, default=6)
    args = parser.parse_args()
    try:
        deduplicate(args.source, args.output, args.method, args.threshold)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
