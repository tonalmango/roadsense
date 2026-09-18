"""Train or resume the dedicated RDD2022 India one-class pothole detector.

The source dataset at D:\\RoadSense_Pothole_India is read-only for training.
This script writes only the new project-local training run and never loads the
old RAD checkpoint.
"""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


PROJECT_ROOT = Path(__file__).parent
DATA_YAML = Path(r"D:\RoadSense_Pothole_India\data.yaml")
RUN_PROJECT = PROJECT_ROOT / "training"
RUN_NAME = "pothole_yolo11n"
RUN_DIRECTORY = RUN_PROJECT / RUN_NAME


def _validate_dataset():
    """Verify the external prepared dataset before consuming GPU time."""
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Prepared pothole YAML was not found: {DATA_YAML}")
    data = check_det_dataset(str(DATA_YAML))
    names = data.get("names", {})
    if dict(names) != {0: "pothole"}:
        raise ValueError(f"Expected exactly one class {{0: 'pothole'}}; found {names}")
    for split in ("train", "val"):
        image_directory = Path(data[split])
        label_directory = image_directory.parents[1] / "labels" / image_directory.name
        image_stems = {path.stem for path in image_directory.glob("*.*")}
        label_stems = {path.stem for path in label_directory.glob("*.txt")}
        if not image_stems or image_stems != label_stems:
            raise ValueError(f"Image/label mismatch in prepared pothole {split} split.")
    return data


def _remove_failed_empty_run():
    """Remove only a failed run that has no checkpoint to preserve."""
    if not RUN_DIRECTORY.exists():
        return
    weights = RUN_DIRECTORY / "weights"
    if any((weights / name).is_file() for name in ("best.pt", "last.pt")):
        raise RuntimeError(
            "An interrupted pothole run already has checkpoints. Preserve it and use --resume."
        )
    shutil.rmtree(RUN_DIRECTORY)


def _train(batch, resume=False):
    checkpoint = RUN_DIRECTORY / "weights" / "last.pt"
    model = YOLO(str(checkpoint) if resume else "yolo11n.pt")
    return model.train(
        data=str(DATA_YAML),
        epochs=40,
        imgsz=512,
        batch=batch,
        device=0,
        workers=2,
        cache=False,
        patience=8,
        pretrained=not resume,
        project=str(RUN_PROJECT),
        name=RUN_NAME,
        exist_ok=resume,
        resume=resume,
    )


def main():
    parser = argparse.ArgumentParser(description="Train the dedicated RoadSense YOLO11n pothole model.")
    parser.add_argument("--resume", action="store_true", help="Resume from training/pothole_yolo11n/weights/last.pt.")
    arguments = parser.parse_args()
    data = _validate_dataset()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this configuration requires device=0.")
    checkpoint = RUN_DIRECTORY / "weights" / "last.pt"
    if arguments.resume and not checkpoint.is_file():
        raise FileNotFoundError(f"Cannot resume; checkpoint not found: {checkpoint}")
    if RUN_DIRECTORY.exists() and not arguments.resume:
        raise FileExistsError(f"Training output already exists: {RUN_DIRECTORY}. Use --resume if appropriate.")

    print("RoadSense dedicated pothole YOLO11n training")
    print(f"Dataset: {DATA_YAML}")
    print(f"Train images: {len(list(Path(data['train']).glob('*.*')))} | Validation images: {len(list(Path(data['val']).glob('*.*')))}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Model: yolo11n.pt | Epochs: 40 | Image size: 512 | Initial batch: 4")
    for batch in (4, 2, 1):
        try:
            _train(batch, resume=arguments.resume)
            return
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or arguments.resume or batch == 1:
                raise
            print(f"CUDA out of memory at batch={batch}; retrying with a smaller batch.")
            torch.cuda.empty_cache()
            _remove_failed_empty_run()


if __name__ == "__main__":
    main()
