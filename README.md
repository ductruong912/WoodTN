# WoodTN — Wood image classification

WoodTN is a computer vision research project for classifying wood images into
15 wood labels and an `other` category. Some labels distinguish the same wood
type by collection region; these are dataset labels, not 15 distinct species.

The repository covers image quality filtering, perceptual deduplication,
EfficientNetV2-S transfer learning, checkpoint evaluation, and ONNX
export for a local webcam demo.

## Workflow

```text
Class-folder images → deduplicate → filter blur → train/val/test split
                                                ↓
                               EfficientNetV2-S
                                                ↓
                      Held-out evaluation → ONNX export → webcam demo
```

The [EfficientNetV2-S notebook](notebooks/efficientnet-v2-s.ipynb) uses head
warmup, full fine-tuning, balanced sampling, Mixup/CutMix, and early stopping.

The real-image evaluation set is included in [test_data/](test_data/): 143 images
across the 16 labels. A historical Excel result is retained at
[best_model_epoch14.xlsx](results/best_model_epoch14.xlsx). The trained
EfficientNetV2-S checkpoint is included at
[Go_v4_epoch_14.pth](save_model/Go_v4_epoch_14.pth), so you can evaluate the real
images without retraining. Training data and other internal reports are
excluded. This release does not claim an independently reproduced accuracy
score.

## Setup

Use Python 3.11. Create and activate a virtual environment from the repository
root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with `source .venv/bin/activate`.

Install a matching PyTorch and torchvision pair for your platform using the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then
install the project dependencies:

```sh
python -m pip install -r requirements.txt
```

The dependency ranges allow platform-specific PyTorch builds; they are not a
frozen training environment. Pretrained ImageNet weights download on the first
training run. Evaluation and export can use the included trained checkpoint.

## Prepare data

Organize images into one directory per label, using the spelling in
[classes.txt](classes.txt). `ImageFolder` assigns indices by sorted folder name.
Keep this ordering identical across splits and checkpoints.

```text
data/raw/
├── Bach dan Lang Son/
│   └── image-001.jpg
├── Bach dan Thai Nguyen/
│   └── image-002.jpg
├── ...
└── other/
    └── image-003.jpg
```

Run commands from the repository root. Each preprocessing command copies to a
new destination and leaves the source unchanged:

```sh
python -m tools.deduplicate_images data/raw data/deduplicated --method phash --threshold 6
python -m tools.filter_blur data/deduplicated data/clear --threshold 20 --crop-size 300
python -m tools.split_dataset data/clear data_split --seed 42
```

The split ratio is 75% train, 15% validation, and 10% test, with rounding per
class. Tune the blur threshold on your acquisition setup; images smaller than
the specified center crop are excluded from the copied output. Deduplication
compares images within each class and keeps the largest image per hash group.
It is a heuristic and has quadratic comparison cost.

For a dataset with multiple photos of one physical wood sample, create splits
by sample or capture session yourself. The random file split does not guarantee
sample independence. Review near duplicates and cross-class label conflicts
before training. Augment only after splitting.

An optional resize tool is available for preparing separate image copies:

```sh
python -m tools.resize_images data/clear data/resized --size 300
```

The notebooks already resize images at runtime; this step is optional.

## Train

```sh
jupyter lab
```

Open `notebooks/efficientnet-v2-s.ipynb` and run its cells in order. Start Jupyter
from the repository root. Defaults are `data_split/` for data and
`checkpoints/` for model output. Override them before launching Jupyter:

```powershell
$env:WOODTN_DATA_DIR = "D:\datasets\wood-split"
$env:WOODTN_OUTPUT_DIR = "D:\experiments\wood"
jupyter lab
```

The notebook uses 224 × 224 inputs. Validation resizes the short edge to 257
pixels. Normalization
uses ImageNet mean and standard deviation. The default notebook worker count is
zero for Windows compatibility. Adjust batch size to fit your hardware.

Checkpoints contain `model_state` and the ordered `classes` list. The notebook
writes `best.pth` under `checkpoints/efficientnet_v2_s/`, selecting the weights
with the lowest validation loss. Evaluate the selected checkpoint on separate
test images after training.

## Evaluate and export

Evaluate a checkpoint on held-out class folders:

```sh
python -m tools.evaluate --architecture efficientnet_v2_s --checkpoint checkpoints/efficientnet_v2_s/best.pth --data data_split/test --output results/efficientnet-test
```

To evaluate the included real-image set with the bundled EfficientNetV2-S
checkpoint and its 300-pixel preprocessing:

```sh
python -m tools.evaluate --architecture efficientnet_v2_s --checkpoint save_model/Go_v4_epoch_14.pth --data test_data --image-size 300 --resize-size 300 --output results/real-wood-test
```

The command writes `report.json` with precision, recall, F1, accuracy, and
preprocessing settings, `confusion_matrix.csv`, and one `evaluation.xlsx`
workbook with `Classification_Report` and `Confusion_Matrix` sheets.
It verifies dataset and
checkpoint class order against `classes.txt`. For another label set, pass
`--classes path/to/classes.txt` containing one label per line in training order.
The historical workbook remains unchanged when you run a new evaluation.

Export the bundled EfficientNetV2-S to ONNX, then run the webcam demo:

```sh
python -m tools.export_onnx --architecture efficientnet_v2_s --checkpoint save_model/Go_v4_epoch_14.pth --image-size 300 --output checkpoints/efficientnet-v2-s.onnx
python -m tools.webcam --architecture efficientnet_v2_s --model checkpoints/efficientnet-v2-s.onnx --image-size 300 --resize-size 300
```

Export checks the ONNX graph. Preprocessing and class labels remain external to
the graph: use the same architecture, input size, resize size, and class file
for evaluation and the demo. For an older EfficientNet checkpoint trained at
300 pixels, explicitly pass `--image-size 300` to export and
`--image-size 300 --resize-size 300` to evaluation or the demo if those match
its training pipeline. Press Esc to close the webcam window.

## Verification

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
ruff check tools tests notebooks
ruff format --check tools tests notebooks
```

Tests use synthetic images and temporary checkpoints; they do not measure wood
classification accuracy. Full training, camera hardware, and real-data model
quality require separate validation.

## Scope and limitations

- Visually similar wood types and collection regions can be hard to separate.
- Lighting, magnification, surface preparation, and backgrounds can change
  predictions. An `other` category does not guarantee rejection of every
  unknown input.
- The source release contains research workflows and a local demonstration.
  It does not include a mobile app, hosted API, or deployment service.
- See [data and model notes](docs/data-and-models.md) for class mapping,
  evaluation boundaries, and artifact handling.
