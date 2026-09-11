"""Train YOLO11n on the RAD Road Anomaly Detection dataset.

Run manually from the RoadSense project root:
    python train_rad.py
"""

from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).parent
DATASET_YAML = Path(r"D:\datasetRad\data.yaml")
TRAINING_PROJECT = PROJECT_ROOT / "training"
RUN_NAME = "rad_yolo11n"
RUN_DIRECTORY = TRAINING_PROJECT / RUN_NAME


def main():
    """Start the explicitly configured RAD training run."""
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(f"RAD dataset YAML was not found: {DATASET_YAML}")
    if RUN_DIRECTORY.exists():
        raise FileExistsError(
            f"Training output already exists: {RUN_DIRECTORY}\n"
            "Rename or move that run before starting a new one."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This training configuration requires device=0.")

    print("RoadSense RAD training configuration")
    print(f"Dataset YAML: {DATASET_YAML}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output: {RUN_DIRECTORY}")
    print("Model: yolo11n.pt | Epochs: 30 | Image size: 640 | Batch: 4")

    # This creates a new RAD-specific model run; model/best.pt is never touched.
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATASET_YAML),
        epochs=30,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        cache=False,
        patience=8,
        pretrained=True,
        project=str(TRAINING_PROJECT),
        name=RUN_NAME,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
