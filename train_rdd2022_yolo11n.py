"""Manually train YOLO11n on the prepared RDD2022 YOLO dataset."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).parent
DATA_YAML = Path(r"D:\21431547_yolo\data.yaml")
RUN_PROJECT = PROJECT_ROOT / "training"
RUN_NAME = "rdd2022_yolo11n"


def train(batch, resume=False):
    checkpoint = RUN_PROJECT / RUN_NAME / "weights" / "last.pt"
    model = YOLO(str(checkpoint) if resume else "yolo11n.pt")
    return model.train(
        data=str(DATA_YAML), epochs=40, imgsz=512, batch=batch, device=0,
        workers=2, cache=False, patience=10, pretrained=not resume,
        project=str(RUN_PROJECT), name=RUN_NAME, exist_ok=resume, resume=resume,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a separate YOLO11n RDD2022 RoadSense model.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Prepared YOLO YAML not found: {DATA_YAML}. Run prepare_rdd2022_yolo.py first.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this RTX 2050 training setup requires device=0.")
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Model: yolo11n.pt | Image size: 512 | Batch: {args.batch} | Epochs: 40")
    try:
        train(args.batch, resume=args.resume)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower() or args.batch <= 1 or args.resume:
            raise
        print("CUDA out of memory at batch", args.batch, "— retrying once with batch=1.")
        torch.cuda.empty_cache()
        train(1, resume=False)


if __name__ == "__main__":
    main()
