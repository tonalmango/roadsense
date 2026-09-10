import streamlit as st
from pathlib import Path
import cv2
from ultralytics import YOLO

from severity import calculate_severity
from priority import calculate_priority
from gps_mapping import get_gps


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Road Sense",
    page_icon="🛣️",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "model" / "best.pt"

model = YOLO(str(MODEL_PATH))


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🛣️ Road Sense")

st.subheader(
    "AI-Based Road Damage Detection and Repair Priority System"
)

st.write(
    "Upload a road image to detect potholes, cracks, "
    "and manholes using YOLOv8."
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    # Create temporary uploads folder
    upload_folder = Path("uploads")
    upload_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    image_path = upload_folder / uploaded_file.name

    # Save uploaded image
    with open(image_path, "wb") as file:

        file.write(
            uploaded_file.getbuffer()
        )

    # Read image
    image = cv2.imread(
        str(image_path)
    )

    # --------------------------------------------------
    # RUN YOLO
    # --------------------------------------------------

    results = model.predict(
        source=str(image_path),
        conf=0.15,
        save=False,
        verbose=False
    )

    result = results[0]

    detections = []


    # --------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            damage_type = model.names[
                class_id
            ]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            # Severity
            severity = calculate_severity(
                damage_type,
                confidence
            )


            # Priority
            priority = calculate_priority(
                severity
            )


            # Store result
            detections.append({
                "damage_type": damage_type,
                "confidence": confidence,
                "severity": severity,
                "priority": priority
            })


            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )


            # Label
            label = (
                f"{damage_type} "
                f"{confidence:.0%} "
                f"{severity}"
            )


            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    st.divider()

    col1, col2 = st.columns(2)


    # Original image
    with col1:

        st.subheader("Original Image")

        st.image(
            uploaded_file,
            use_container_width=True
        )


    # Detection image
    with col2:

        st.subheader("Detection Result")

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            image_rgb,
            use_container_width=True
        )


    # --------------------------------------------------
    # DAMAGE INFORMATION
    # --------------------------------------------------

    st.divider()

    st.subheader("🚧 Damage Analysis")


    if detections:

        for i, detection in enumerate(
            detections,
            start=1
        ):

            st.markdown(
                f"### Damage {i}"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Damage Type",
                    detection["damage_type"]
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{detection['confidence']:.1%}"
                )

            with col3:

                st.metric(
                    "Severity",
                    detection["severity"]
                )

            with col4:

                st.metric(
                    "Priority",
                    detection["priority"]
                )


    else:

        st.success(
            "No road damage detected."
        )


    # --------------------------------------------------
    # GPS LOCATION
    # --------------------------------------------------

    st.divider()

    st.subheader("📍 GPS Location")


    gps = get_gps(
        uploaded_file.name
    )


    if gps["latitude"] is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Latitude",
                gps["latitude"]
            )

        with col2:

            st.metric(
                "Longitude",
                gps["longitude"]
            )


        st.info(
            "GPS coordinates shown are demo coordinates "
            "for project demonstration."
        )

    else:

        st.warning(
            "GPS location not available for this image."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Road Sense | AI-Based Road Damage Detection System"
)