"""Resize the short edge and center-crop images into a separate directory."""

import argparse
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

from tools.data_utils import image_paths, validate_output


def resize_images(source, output, size=224):
    if size <= 0:
        raise ValueError("Size must be positive.")
    source, output = validate_output(source, output)
    count = 0
    for path in tqdm(image_paths(source), desc="Resizing"):
        destination = output / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
            image.save(destination)
        count += 1
    print(f"Saved {count} images to {output}")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    try:
        resize_images(args.source, args.output, args.size)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
