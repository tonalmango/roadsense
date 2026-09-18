"""Train or resume the dedicated four-class RDD2022 YOLO11n damage model."""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


PROJECT_ROOT = Path(__file__).parent
DATA_YAML = Path(r"D:\RoadSense_RDD2022_4Class\data.yaml")
RUN_DIRECTORY = PROJECT_ROOT / "training" / "rdd_yolo11n"
EXPECTED_NAMES = {0: "Pothole", 1: "Longitudinal Crack", 2: "Transverse Crack", 3: "Alligator Crack"}


def validate_training_data():
    data = check_det_dataset(str(DATA_YAML))
    if dict(data.get("names", {})) != EXPECTED_NAMES:
        raise ValueError(f"Unexpected RDD class mapping: {data.get('names')}")
    for split in ("train", "val"):
        images = Path(data[split])
        labels = images.parents[1] / "labels" / images.name
        image_stems = {path.stem for path in images.glob("*.*")}
        label_stems = {path.stem for path in labels.glob("*.txt")}
        if not image_stems or image_stems != label_stems:
            raise ValueError(f"RDD image/label mismatch in {split}.")
    return data


def remove_failed_empty_run():
    if not RUN_DIRECTORY.exists():
        return
    if any((RUN_DIRECTORY / "weights" / name).is_file() for name in ("best.pt", "last.pt")):
        raise RuntimeError("Existing RDD run has checkpoints. Resume it instead of overwriting.")
    shutil.rmtree(RUN_DIRECTORY)


def train(arguments):
    checkpoint = RUN_DIRECTORY / "weights" / "last.pt"
    model = YOLO(str(checkpoint) if arguments.resume else "yolo11n.pt")
    return model.train(
        data=str(DATA_YAML), epochs=arguments.epochs, imgsz=arguments.imgsz, batch=arguments.batch,
        device=0, workers=2, cache=False, patience=arguments.patience, pretrained=not arguments.resume,
        optimizer="auto", degrees=0.0, flipud=0.0, fliplr=0.5, translate=0.05, scale=0.30,
        mosaic=0.5, close_mosaic=10, copy_paste=0.0,
        project=str(PROJECT_ROOT / "training"), name="rdd_yolo11n", exist_ok=arguments.resume, resume=arguments.resume,
    )


def main():
    parser = argparse.ArgumentParser(description="Train RoadSense four-class RDD2022 YOLO11n model.")
    parser.add_argument("--epochs", type=int, default=60, help="60 balances overnight completion with validation; increase only if time permits.")
    parser.add_argument("--imgsz", type=int, default=512, help="512 is selected for reliable 4 GB RTX 2050 training.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    data = validate_training_data()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is required for this local training configuration.")
    if RUN_DIRECTORY.exists() and not arguments.resume:
        raise FileExistsError(f"Run already exists: {RUN_DIRECTORY}. Use --resume to preserve its checkpoints.")
    if arguments.resume and not (RUN_DIRECTORY / "weights" / "last.pt").is_file():
        raise FileNotFoundError("Cannot resume because training/rdd_yolo11n/weights/last.pt is missing.")
    print(f"GPU: {torch.cuda.get_device_name(0)} | train={len(list(Path(data['train']).glob('*.*')))} | val={len(list(Path(data['val']).glob('*.*')))}")
    for batch in (arguments.batch, 2, 1):
        if batch > arguments.batch:
            continue
        try:
            arguments.batch = batch
            train(arguments)
            return
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or arguments.resume or batch == 1:
                raise
            print(f"CUDA out of memory at batch={batch}; retrying with a smaller batch.")
            torch.cuda.empty_cache()
            remove_failed_empty_run()


if __name__ == "__main__":
    main()
