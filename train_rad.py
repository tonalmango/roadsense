"""Train YOLO11m on the RAD Road Anomaly Detection dataset.

Run manually from the RoadSense project root:
    python train_rad.py
"""

from pathlib import Path
import argparse

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).parent
DATASET_YAML = PROJECT_ROOT / "rad_yolo11m_data.yaml"
TRAINING_PROJECT = PROJECT_ROOT / "training"
RUN_NAME = "rad_yolo11m"
RUN_DIRECTORY = TRAINING_PROJECT / RUN_NAME


def main():
    """Start the explicitly configured RAD training run."""
    parser = argparse.ArgumentParser(description="Train or resume RoadSense RAD YOLO11m.")
    parser.add_argument("--resume", action="store_true", help="Resume from this run's last.pt checkpoint.")
    arguments = parser.parse_args()
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(f"RAD dataset YAML was not found: {DATASET_YAML}")
    last_checkpoint = RUN_DIRECTORY / "weights" / "last.pt"
    if RUN_DIRECTORY.exists() and not arguments.resume:
        raise FileExistsError(
            f"Training output already exists: {RUN_DIRECTORY}\n"
            "Use --resume to continue its last.pt checkpoint."
        )
    if arguments.resume and not last_checkpoint.is_file():
        raise FileNotFoundError(f"Cannot resume; checkpoint not found: {last_checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This training configuration requires device=0.")

    print("RoadSense RAD YOLO11m training configuration")
    print(f"Dataset YAML: {DATASET_YAML}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output: {RUN_DIRECTORY}")
    print("Model: yolo11m.pt | Epochs: 40 | Image size: 416 | Batch: 1")

    # YOLO11m is deliberately trained in a new run; model/best.pt is untouched.
    # Batch 1 and 416px are conservative for the RTX 2050's 4 GB VRAM.
    model = YOLO(str(last_checkpoint) if arguments.resume else "yolo11m.pt")
    model.train(
        data=str(DATASET_YAML),
        epochs=40,
        imgsz=416,
        batch=1,
        device=0,
        workers=2,
        cache=False,
        patience=10,
        pretrained=not arguments.resume,
        project=str(TRAINING_PROJECT),
        name=RUN_NAME,
        exist_ok=arguments.resume,
        resume=arguments.resume,
    )


if __name__ == "__main__":
    main()
