"""Shared filesystem rules for image preprocessing."""

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_paths(root):
    return sorted(
        path
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_output(source, output):
    """Require a separate, new destination so inputs cannot be overwritten."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f"Input directory does not exist: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Input and output directories must not overlap.")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    return source, output
