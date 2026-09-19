"""Trustworthy evaluation for the RoadSense four-class RDD model.

This utility is evaluation-only. It never trains, changes model weights, edits
labels, or changes RoadSense production thresholds. Ultralytics supplies the
official mAP metrics; this module independently matches validation labels and
predictions for transparent TP/FP/FN, threshold, IoU, confusion, and error
analysis.

Example:
    python evaluate_rdd_model.py --data D:/RoadSense_RDD2022_4Class/data.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "training" / "rdd_yolo11n" / "weights" / "best.pt"
DEFAULT_DATA_CANDIDATES = (
    PROJECT_ROOT / "training" / "rdd_yolo11n" / "data.yaml",
    Path("D:/RoadSense_RDD2022_4Class/data.yaml"),
    Path("D:/RoadSense_RDD2022_4Class"),
)
CLASS_NAMES = (
    "Pothole",
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
)
DEFAULT_CONFIDENCE_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.10, 0.71, 0.05))
DEFAULT_IOU_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


@dataclass(frozen=True)
class Box:
    class_id: int
    confidence: float | None
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Match:
    ground_truth: Box | None
    prediction: Box | None
    iou: float


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_json_value(rows))


def _resolve_data_yaml(value: str | None) -> Path:
    candidates = [Path(value)] if value else []
    environment_value = os.environ.get("ROADSENSE_RDD_DATA", "").strip()
    if environment_value:
        candidates.append(Path(environment_value))
    candidates.extend(DEFAULT_DATA_CANDIDATES)
    for candidate in candidates:
        if candidate.is_dir():
            candidate = candidate / "data.yaml"
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "RDD dataset YAML was not found. Supply --data or set ROADSENSE_RDD_DATA. "
        f"Searched: {searched}"
    )


def _resolve_split_path(data_yaml: Path, data: dict[str, Any], split: str) -> Path:
    value = data.get(split)
    if not value:
        raise ValueError(f"Dataset YAML has no {split!r} split.")
    path = Path(str(value))
    if not path.is_absolute():
        path = data_yaml.parent / path
    return path.resolve()


def load_dataset(data_yaml: Path) -> tuple[dict[str, Any], Path, Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names", {})
    names = {int(key): str(value) for key, value in names.items()}
    expected = {index: name for index, name in enumerate(CLASS_NAMES)}
    if names != expected:
        raise ValueError(f"Unexpected RDD class mapping in {data_yaml}: {names}; expected {expected}")
    return data, _resolve_split_path(data_yaml, data, "val"), data_yaml


def _label_path(image_path: Path, validation_images: Path) -> Path:
    relative = image_path.relative_to(validation_images)
    return validation_images.parent.parent / "labels" / validation_images.name / relative.with_suffix(".txt")


def _read_ground_truth(image_path: Path, validation_images: Path) -> list[Box]:
    label_path = _label_path(image_path, validation_images)
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing validation label for {image_path}: {label_path}")
    try:
        width, height = _image_size(image_path)
        boxes = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid label at {label_path}:{line_number}")
            class_id, x_center, y_center, box_width, box_height = map(float, fields)
            class_id = int(class_id)
            if class_id not in range(len(CLASS_NAMES)):
                raise ValueError(f"Unknown class {class_id} at {label_path}:{line_number}")
            x1 = (x_center - box_width / 2) * width
            y1 = (y_center - box_height / 2) * height
            x2 = (x_center + box_width / 2) * width
            y2 = (y_center + box_height / 2) * height
            boxes.append(Box(class_id, None, (x1, y1, x2, y2)))
        return boxes
    except UnicodeDecodeError as error:
        raise ValueError(f"Unreadable label file: {label_path}") from error


def _image_size(image_path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read validation image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def _iou(left: Box, right: Box) -> float:
    lx1, ly1, lx2, ly2 = left.xyxy
    rx1, ry1, rx2, ry2 = right.xyxy
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _match(ground_truth: list[Box], predictions: list[Box], iou_threshold: float) -> list[Match]:
    candidates = sorted(
        (
            (_iou(truth, prediction), truth_index, prediction_index)
            for truth_index, truth in enumerate(ground_truth)
            for prediction_index, prediction in enumerate(predictions)
            if _iou(truth, prediction) >= iou_threshold
        ),
        reverse=True,
    )
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[Match] = []
    for overlap, truth_index, prediction_index in candidates:
        if truth_index in matched_truth or prediction_index in matched_predictions:
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(prediction_index)
        matches.append(Match(ground_truth[truth_index], predictions[prediction_index], overlap))
    matches.extend(Match(ground_truth[index], None, 0.0) for index in range(len(ground_truth)) if index not in matched_truth)
    matches.extend(Match(None, predictions[index], 0.0) for index in range(len(predictions)) if index not in matched_predictions)
    return matches


def _predict_all(model: YOLO, image_paths: list[Path], imgsz: int, device: str | None, batch: int) -> dict[str, list[Box]]:
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=0.001,
        iou=0.7,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
        save=False,
        stream=True,
    )
    predictions: dict[str, list[Box]] = {}
    for result in results:
        path = Path(str(result.path)).resolve()
        boxes: list[Box] = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            boxes = [
                Box(int(class_id), float(confidence), tuple(float(value) for value in coordinates))
                for coordinates, confidence, class_id in zip(xyxy, confidences, class_ids)
            ]
        predictions[str(path)] = boxes
    return predictions


def _filtered(predictions: list[Box], confidence: float) -> list[Box]:
    return [prediction for prediction in predictions if (prediction.confidence or 0.0) >= confidence]


def _counts(
    images: list[Path], validation_images: Path, predictions: dict[str, list[Box]], confidence: float, iou: float
) -> tuple[dict[str, Any], list[Match], dict[str, list[Box]]]:
    totals = Counter()
    all_matches: list[Match] = []
    filtered_by_image: dict[str, list[Box]] = {}
    per_class = {name: Counter() for name in CLASS_NAMES}
    for image_path in images:
        truths = _read_ground_truth(image_path, validation_images)
        image_predictions = _filtered(predictions.get(str(image_path.resolve()), []), confidence)
        filtered_by_image[str(image_path.resolve())] = image_predictions
        matches = _match(truths, image_predictions, iou)
        all_matches.extend(matches)
        totals["ground_truth"] += len(truths)
        totals["predictions"] += len(image_predictions)
        for match in matches:
            if match.ground_truth is not None and match.prediction is not None and match.ground_truth.class_id == match.prediction.class_id:
                totals["tp"] += 1
                per_class[CLASS_NAMES[match.ground_truth.class_id]]["tp"] += 1
            elif match.ground_truth is None:
                totals["fp"] += 1
                per_class[CLASS_NAMES[match.prediction.class_id]]["fp"] += 1
            elif match.prediction is None:
                totals["fn"] += 1
                per_class[CLASS_NAMES[match.ground_truth.class_id]]["fn"] += 1
            else:
                totals["fn"] += 1
                totals["fp"] += 1
                per_class[CLASS_NAMES[match.ground_truth.class_id]]["fn"] += 1
                per_class[CLASS_NAMES[match.prediction.class_id]]["fp"] += 1
        for truth in truths:
            per_class[CLASS_NAMES[truth.class_id]]["ground_truth"] += 1
        for prediction in image_predictions:
            per_class[CLASS_NAMES[prediction.class_id]]["predictions"] += 1
    totals["precision"] = _ratio(totals["tp"], totals["tp"] + totals["fp"])
    totals["recall"] = _ratio(totals["tp"], totals["tp"] + totals["fn"])
    totals["f1"] = _f1(totals["precision"], totals["recall"])
    for values in per_class.values():
        values["precision"] = _ratio(values["tp"], values["tp"] + values["fp"])
        values["recall"] = _ratio(values["tp"], values["tp"] + values["fn"])
        values["f1"] = _f1(values["precision"], values["recall"])
    return {"overall": dict(totals), "per_class": {name: dict(values) for name, values in per_class.items()}}, all_matches, filtered_by_image


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


def _official_metrics(model: YOLO, data_yaml: Path, output_dir: Path, imgsz: int, device: str | None, batch: int) -> dict[str, Any]:
    validation = model.val(
        data=str(data_yaml), split="val", conf=0.001, iou=0.7, imgsz=imgsz,
        batch=batch,
        device=device, plots=True, save_json=False, project=str(output_dir), name="official_val", exist_ok=True,
        verbose=False,
    )
    box = validation.box
    per_class_ap50 = [float(value) for value in np.asarray(box.ap50).reshape(-1)]
    per_class_ap = [float(value) for value in np.asarray(box.ap).reshape(-1)]
    per_class_precision = [float(value) for value in np.asarray(box.p).reshape(-1)]
    per_class_recall = [float(value) for value in np.asarray(box.r).reshape(-1)]
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "per_class": {
            name: {
                "precision": per_class_precision[index] if index < len(per_class_precision) else None,
                "recall": per_class_recall[index] if index < len(per_class_recall) else None,
                "AP50": per_class_ap50[index] if index < len(per_class_ap50) else None,
                "AP50-95": per_class_ap[index] if index < len(per_class_ap) else None,
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "source": "Ultralytics YOLO.val on the untouched val split",
    }


def _analysis_records(images: list[Path], validation_images: Path, predictions: dict[str, list[Box]], confidence: float, iou: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[int]]]:
    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    matrix = [[0 for _ in range(len(CLASS_NAMES) + 1)] for _ in range(len(CLASS_NAMES) + 1)]
    for image_path in images:
        truths = _read_ground_truth(image_path, validation_images)
        all_predictions = predictions.get(str(image_path.resolve()), [])
        selected_predictions = _filtered(all_predictions, confidence)
        matches = _match(truths, selected_predictions, iou)
        for match in matches:
            if match.ground_truth and match.prediction:
                truth_id, prediction_id = match.ground_truth.class_id, match.prediction.class_id
                matrix[truth_id][prediction_id] += 1
                if truth_id != prediction_id:
                    false_negatives.append(_miss_record(image_path, match.ground_truth, match.prediction, match.iou, "WRONG_CLASS"))
                    false_positives.append(_false_positive_record(image_path, match.prediction, match.iou, "WRONG_CLASS"))
            elif match.ground_truth:
                best = max(all_predictions, key=lambda prediction: _iou(match.ground_truth, prediction), default=None)
                best_iou = _iou(match.ground_truth, best) if best else 0.0
                reason = "NO_DETECTION"
                if best and best.class_id == match.ground_truth.class_id and (best.confidence or 0.0) < confidence:
                    reason = "LOW_CONFIDENCE"
                elif best and best.class_id == match.ground_truth.class_id and best_iou > 0:
                    reason = "LOCALIZATION_FAILURE"
                elif best and best.class_id != match.ground_truth.class_id and best_iou >= iou:
                    reason = "WRONG_CLASS"
                false_negatives.append(_miss_record(image_path, match.ground_truth, best, best_iou, reason))
                matrix[match.ground_truth.class_id][-1] += 1
            else:
                matrix[-1][match.prediction.class_id] += 1
                false_positives.append(_false_positive_record(image_path, match.prediction, 0.0, "BACKGROUND"))
    return false_negatives, false_positives, matrix


def _box_dict(box: Box | None) -> dict[str, Any] | None:
    if box is None:
        return None
    return {"class_id": box.class_id, "class_name": CLASS_NAMES[box.class_id], "confidence": box.confidence, "xyxy": list(box.xyxy)}


def _miss_record(image_path: Path, truth: Box, prediction: Box | None, overlap: float, reason: str) -> dict[str, Any]:
    return {"image": image_path.name, "image_path": str(image_path), "ground_truth": _box_dict(truth), "best_prediction": _box_dict(prediction), "iou": round(overlap, 6), "reason": reason}


def _false_positive_record(image_path: Path, prediction: Box, overlap: float, reason: str) -> dict[str, Any]:
    return {"image": image_path.name, "image_path": str(image_path), "prediction": _box_dict(prediction), "iou": round(overlap, 6), "reason": reason}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data_yaml = _resolve_data_yaml(args.data)
    data, validation_images, _ = load_dataset(data_yaml)
    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"RDD model checkpoint not found: {model_path}")
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in validation_images.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not images:
        raise ValueError(f"No validation images found in {validation_images}")

    model = YOLO(str(model_path))
    predictions = _predict_all(model, images, args.imgsz, args.device, args.batch)
    official = _official_metrics(model, data_yaml, output_dir, args.imgsz, args.device, args.batch)
    baseline, _, _ = _counts(images, validation_images, predictions, args.baseline_confidence, args.baseline_iou)

    threshold_rows = []
    for confidence in args.confidences:
        counts, _, _ = _counts(images, validation_images, predictions, confidence, args.baseline_iou)
        threshold_rows.append({"confidence": confidence, **counts["overall"]})
    best_f1 = max(threshold_rows, key=lambda row: (row["f1"], -row["confidence"]))
    highest_recall = sorted(threshold_rows, key=lambda row: (-row["recall"], row["confidence"]))[:3]

    iou_rows = []
    for iou in args.ious:
        counts, _, _ = _counts(images, validation_images, predictions, args.baseline_confidence, iou)
        iou_rows.append({"iou_threshold": iou, **counts["overall"]})

    false_negatives, false_positives, matrix = _analysis_records(images, validation_images, predictions, args.baseline_confidence, args.baseline_iou)
    per_class = baseline["per_class"]
    for name, metrics in official["per_class"].items():
        per_class[name].update(metrics)

    report = {
        "evaluation": {
            "model": str(model_path), "dataset_yaml": str(data_yaml), "validation_images": str(validation_images),
            "validation_image_count": len(images), "class_mapping": {str(index): name for index, name in enumerate(CLASS_NAMES)},
            "baseline_confidence": args.baseline_confidence, "baseline_iou": args.baseline_iou,
            "metrics_note": "Precision, recall, F1, mAP50, and mAP50-95 are primary object-detection metrics; generic classification accuracy is not reported.",
            "weights_or_labels_modified": False,
        },
        "official_ultralytics": official,
        "baseline_matching": {"overall": baseline["overall"], "per_class": per_class},
        "threshold_analysis": threshold_rows,
        "threshold_recommendations": {"highest_f1": best_f1, "higher_recall_options": highest_recall, "production_threshold_changed": False},
        "iou_analysis": iou_rows,
        "confusion_matrix": {"labels": [*CLASS_NAMES, "background"], "rows_true_columns_predicted": matrix},
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "false_positive_summary": dict(Counter(record["reason"] for record in false_positives)),
    }
    _write_json(output_dir / "evaluation_report.json", report)
    _write_json(output_dir / "confusion_matrix.json", report["confusion_matrix"])
    _write_csv(output_dir / "confidence_thresholds.csv", threshold_rows)
    _write_json(output_dir / "confidence_thresholds.json", threshold_rows)
    _write_csv(output_dir / "iou_analysis.csv", iou_rows)
    _write_json(output_dir / "iou_analysis.json", iou_rows)
    _write_json(output_dir / "false_negatives.json", false_negatives)
    _write_json(output_dir / "false_positives.json", false_positives)
    _write_csv(output_dir / "false_negatives.csv", false_negatives)
    _write_csv(output_dir / "false_positives.csv", false_positives)
    _write_confusion_png(output_dir / "confusion_matrix.png", matrix)
    print(json.dumps({"output": str(output_dir), "validation_images": len(images), "baseline": baseline["overall"], "highest_f1": best_f1}, indent=2))
    return report


def _write_confusion_png(path: Path, matrix: list[list[int]]) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(np.asarray(matrix), cmap="Blues")
    figure.colorbar(image, ax=axis)
    labels = [*CLASS_NAMES, "background"]
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="Ground truth", title="RDD validation object-detection confusion matrix")
    for row in range(len(matrix)):
        for column in range(len(matrix[row])):
            axis.text(column, row, matrix[row][column], ha="center", va="center", color="black")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the untouched RDD2022 validation split without changing production artifacts.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=str, help="Dataset directory or data.yaml; overrides ROADSENSE_RDD_DATA.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "training" / "audits" / "rdd_yolo11n_evaluation")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1, help="Evaluation batch size; 1 is safest on small GPUs.")
    parser.add_argument("--device", default=None, help="Ultralytics device value, for example 0 or cpu.")
    parser.add_argument("--baseline-confidence", type=float, default=0.25)
    parser.add_argument("--baseline-iou", type=float, default=0.50)
    parser.add_argument("--confidences", type=float, nargs="+", default=list(DEFAULT_CONFIDENCE_THRESHOLDS))
    parser.add_argument("--ious", type=float, nargs="+", default=list(DEFAULT_IOU_THRESHOLDS))
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
