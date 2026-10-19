import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from tools.model_utils import (
    ARCHITECTURES,
    build_model,
    evaluation_transform,
    load_model,
    read_classes,
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_each_architecture_produces_one_logit_per_class(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = build_model(architecture, 2).eval()
                with torch.inference_mode():
                    output = model(torch.zeros(1, 3, 224, 224))
                self.assertEqual(output.shape, (1, 2))
                self.assertTrue(torch.isfinite(output).all())
                del model

    def test_checkpoint_roundtrip_preserves_predictions_and_checks_class_order(self):
        classes = ["other", "wood"]
        model = build_model("efficientnet_v2_s", len(classes)).eval()
        inputs = torch.zeros(1, 3, 224, 224)
        with torch.inference_mode():
            expected = model(inputs)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            torch.save(
                {"model_state": model.state_dict(), "classes": classes}, checkpoint
            )
            restored = load_model(checkpoint, "efficientnet_v2_s", classes)
            with torch.inference_mode():
                torch.testing.assert_close(restored(inputs), expected)
            with self.assertRaises(ValueError):
                load_model(checkpoint, "efficientnet_v2_s", list(reversed(classes)))
            torch.save(model.state_dict(), checkpoint)
            restored = load_model(checkpoint, "efficientnet_v2_s", classes)
            with torch.inference_mode():
                torch.testing.assert_close(restored(inputs), expected)

    def test_inference_transform_is_deterministic_and_handles_rgb_images(self):
        image = Image.new("RGB", (640, 480), color=(80, 100, 120))
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                transform = evaluation_transform(architecture)
                first, second = transform(image), transform(image)
                self.assertEqual(first.shape, (3, 224, 224))
                torch.testing.assert_close(first, second)

    def test_incompatible_transform_dimensions_are_rejected(self):
        for architecture, size, resize in (
            ("vit_b_16", 384, None),
            ("mvitv2_small", 300, None),
            ("efficientnet_v2_s", 0, None),
            ("efficientnet_v2_s", 224, 100),
        ):
            with self.subTest(architecture=architecture), self.assertRaises(ValueError):
                evaluation_transform(architecture, size, resize)

    def test_class_file_rejects_empty_or_duplicate_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classes.txt"
            for content in ("", "wood\n\n", "wood\nwood\n"):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    read_classes(path)


if __name__ == "__main__":
    unittest.main()
