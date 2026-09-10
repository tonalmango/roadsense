"""Road Sense - YOLOv8 Road Damage Detection"""

from ultralytics import YOLO
from pathlib import Path
import cv2

from severity import calculate_severity
from priority import calculate_priority


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "model" / "best.pt"

# Load trained YOLOv8 model
model = YOLO(str(MODEL_PATH))


# --------------------------------------------------
# DETECT ROAD DAMAGE IN ONE IMAGE
# --------------------------------------------------

def detect_image(image_path, output_folder="results"):

    image_path = Path(image_path)
    output_folder = Path(output_folder)

    # Create results folder if it doesn't exist
    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nProcessing: {image_path.name}")

    # Run YOLO detection
    results = model.predict(
        source=str(image_path),
        conf=0.15,
        save=False,
        verbose=False
    )

    result = results[0]

    detections = []

    # Read image
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image.")
        return []

    # --------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            # Class ID
            class_id = int(box.cls[0])

            # Confidence
            confidence = float(box.conf[0])

            # Damage class
            damage_type = model.names[class_id]

            # Bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # --------------------------------------------------
            # CALCULATE SEVERITY
            # --------------------------------------------------

            severity = calculate_severity(
                damage_type,
                confidence
            )

            # --------------------------------------------------
            # CALCULATE PRIORITY
            # --------------------------------------------------

            priority = calculate_priority(
                severity
            )

            # --------------------------------------------------
            # STORE DETECTION
            # --------------------------------------------------

            detection = {
                "damage_type": damage_type,
                "confidence": confidence,
                "severity": severity,
                "priority": priority,
                "bbox": [x1, y1, x2, y2]
            }

            detections.append(detection)

            # --------------------------------------------------
            # DRAW BOUNDING BOX
            # --------------------------------------------------

            label = (
                f"{damage_type} "
                f"{confidence:.2f} "
                f"{severity}"
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # --------------------------------------------------
    # SAVE ANNOTATED IMAGE
    # --------------------------------------------------

    output_path = output_folder / image_path.name

    cv2.imwrite(
        str(output_path),
        image
    )

    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

    if detections:

        for detection in detections:

            print(
                f"Damage: {detection['damage_type']} | "
                f"Confidence: "
                f"{detection['confidence']:.2%} | "
                f"Severity: {detection['severity']} | "
                f"Priority: {detection['priority']}"
            )

    else:

        print("No road damage detected.")

    print(
        f"Result saved to: {output_path}"
    )

    return detections


# --------------------------------------------------
# PROCESS ALL IMAGES IN FRAMES FOLDER
# --------------------------------------------------

def process_all_images():

    image_folder = Path("frames")
    output_folder = Path("results")

    # Supported image formats
    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png"
    ]

    image_files = []

    for extension in image_extensions:

        image_files.extend(
            image_folder.glob(extension)
        )

    # Check if images exist
    if not image_files:

        print("No images found in frames folder.")
        return

    print("\nRoad Sense YOLOv8 Detection")
    print("---------------------------")

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Images found: {len(image_files)}"
    )

    # Process every image
    for image_path in image_files:

        detect_image(
            image_path,
            output_folder
        )

    print("\n--------------------------------")
    print("All images processed successfully!")
    print("Check the results folder.")
    print("--------------------------------")


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    process_all_images()