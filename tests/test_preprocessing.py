import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.data_utils import validate_output
from tools.deduplicate_images import deduplicate
from tools.filter_blur import filter_images
from tools.resize_images import resize_images
from tools.split_dataset import split_dataset


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()

    def snapshot(self):
        return {
            path.relative_to(self.source).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in self.source.rglob("*")
            if path.is_file()
        }

    def save(self, relative, pixels):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)
        return path

    def test_rejects_overlapping_or_existing_outputs(self):
        for output in (self.source, self.source / "nested", self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                validate_output(self.source, output)
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(ValueError):
            validate_output(self.source, existing)

    def test_deduplication_keeps_largest_copy_and_does_not_merge_labels(self):
        pixels = np.full((32, 32, 3), 120, dtype=np.uint8)
        self.save("wood/small.png", pixels)
        self.save("wood/large.png", np.repeat(np.repeat(pixels, 2, axis=0), 2, axis=1))
        self.save("other/same.png", pixels)
        before = self.snapshot()
        output = self.root / "dedup"
        self.assertEqual(deduplicate(self.source, output, threshold=0), 2)
        self.assertTrue((output / "wood/large.png").is_file())
        self.assertFalse((output / "wood/small.png").exists())
        self.assertTrue((output / "other/same.png").is_file())
        self.assertEqual(self.snapshot(), before)

    def test_dhash_preserves_nested_paths_with_the_same_filename(self):
        rng = np.random.default_rng(42)
        self.save("wood/a/image.png", rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        self.save("wood/b/image.png", rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        output = self.root / "dhash"
        self.assertEqual(
            deduplicate(self.source, output, method="dhash", threshold=0), 2
        )
        self.assertTrue((output / "wood/a/image.png").exists())
        self.assertTrue((output / "wood/b/image.png").exists())

    def test_blur_filter_copies_sharp_images_and_preserves_unicode_paths(self):
        rng = np.random.default_rng(42)
        self.save("gỗ/sharp.png", rng.integers(0, 256, (80, 80, 3), dtype=np.uint8))
        self.save("gỗ/blur.png", np.zeros((80, 80, 3), dtype=np.uint8))
        self.save("gỗ/small.png", rng.integers(0, 256, (8, 8, 3), dtype=np.uint8))
        before = self.snapshot()
        output = self.root / "sharp"
        self.assertEqual(
            filter_images(self.source, output, threshold=20, crop_size=32), 1
        )
        self.assertTrue((output / "gỗ/sharp.png").is_file())
        self.assertEqual(self.snapshot(), before)

    def test_resize_keeps_class_layout_and_source_bytes(self):
        self.save("wood/nested/image.png", np.zeros((40, 80, 3), dtype=np.uint8))
        before = self.snapshot()
        output = self.root / "resized"
        self.assertEqual(resize_images(self.source, output, size=24), 1)
        with Image.open(output / "wood/nested/image.png") as image:
            self.assertEqual(image.size, (24, 24))
        self.assertEqual(self.snapshot(), before)

    def test_split_is_reproducible_disjoint_and_keeps_all_source_images(self):
        for label in ("wood", "other"):
            for index in range(20):
                self.save(
                    f"{label}/{index:02d}.png",
                    np.full((8, 8, 3), index, dtype=np.uint8),
                )
        before = self.snapshot()
        first, second = self.root / "first", self.root / "second"
        split_dataset(self.source, first)
        split_dataset(self.source, second)
        manifests = []
        for output in (first, second):
            manifest = {}
            for split, count in (("train", 15), ("val", 3), ("test", 2)):
                for label in ("wood", "other"):
                    paths = list((output / split / label).glob("*.png"))
                    self.assertEqual(len(paths), count)
                    for path in paths:
                        key = f"{label}/{path.name}"
                        self.assertNotIn(key, manifest)
                        manifest[key] = split
            self.assertEqual(set(manifest), set(before))
            manifests.append(manifest)
        self.assertEqual(manifests[0], manifests[1])
        self.assertEqual(self.snapshot(), before)

    def test_invalid_parameters_fail_before_writing(self):
        output = self.root / "invalid"
        for action in (
            lambda: resize_images(self.source, output, size=0),
            lambda: filter_images(self.source, output, threshold=-1),
            lambda: filter_images(self.source, output, crop_size=0),
            lambda: deduplicate(self.source, output, threshold=-1),
        ):
            with self.subTest(action=action), self.assertRaises(ValueError):
                action()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
