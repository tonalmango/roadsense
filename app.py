"""RoadSense Streamlit dashboard for the existing video-analysis pipeline."""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pydeck as pdk
import streamlit as st

from gps_mapping import load_gps_records
from pipeline import run_pipeline


ROOT = Path(__file__).parent
UPLOADS_FOLDER = ROOT / "uploads"
RUNS_FOLDER = ROOT / "results" / "runs"
SEVERITY_COLORS = {
    "Minor": [80, 170, 90],
    "Moderate": [245, 166, 35],
    "Severe": [220, 65, 65],
}
ROAD_DAMAGE_CLASS = "RoadDamages"


st.set_page_config(page_title="RoadSense", page_icon="RS", layout="wide")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_uploaded_file(uploaded_file, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as file:
        file.write(uploaded_file.getbuffer())


def _run_analysis(video_file, gps_file, sampling_interval, road_context, traffic_factor):
    """Persist user inputs and invoke the existing pipeline once."""
    run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run_folder = RUNS_FOLDER / run_id
    frames_folder = run_folder / "frames"
    results_folder = run_folder / "results"
    video_path = UPLOADS_FOLDER / run_id / Path(video_file.name).name
    _save_uploaded_file(video_file, video_path)

    gps_path = None
    if gps_file is not None:
        gps_path = UPLOADS_FOLDER / run_id / Path(gps_file.name).name
        _save_uploaded_file(gps_file, gps_path)

    result = run_pipeline(
        video_path=video_path,
        frames_folder=frames_folder,
        results_folder=results_folder,
        gps_data_path=gps_path,
        every_seconds=sampling_interval,
        road_context=road_context if road_context != "Not supplied" else None,
        traffic_factor=traffic_factor,
    )
    if not result["success"]:
        return result

    final_path = Path(result["final_detections_path"])
    return {
        **result,
        "run_id": run_id,
        "results_folder": str(results_folder),
        "gps_path": str(gps_path) if gps_path else None,
        "detections": _load_json(final_path),
    }


def _filtered_detections(detections, severity_filter, damage_filter, priority_filter):
    return [
        detection
        for detection in detections
        if (severity_filter == "All" or detection.get("severity") == severity_filter)
        and (
            damage_filter == "All"
            or str(detection.get("damage_type", "")).lower() == damage_filter.lower()
        )
        and (priority_filter == "All" or detection.get("priority_level") == priority_filter)
    ]


def _render_summary(statistics, detections):
    critical_count = sum(
        detection.get("damage_type") == ROAD_DAMAGE_CLASS
        and detection.get("priority_level") == "Critical"
        for detection in detections
    )
    road_damage_count = sum(
        detection.get("damage_type") == ROAD_DAMAGE_CLASS for detection in detections
    )
    columns = st.columns(4)
    columns[0].metric("Frames Processed", statistics.get("sampled_frames", 0))
    columns[1].metric("Usable Frames", statistics.get("usable_frames", 0))
    columns[2].metric("RoadDamages Detected", road_damage_count)
    columns[3].metric("Critical Issues", critical_count)


def _render_map(detections, gps_path):
    map_rows = []
    for detection in detections:
        latitude = detection.get("latitude")
        longitude = detection.get("longitude")
        if latitude is None or longitude is None:
            continue
        map_rows.append(
            {
                **detection,
                "color": SEVERITY_COLORS.get(detection.get("severity"), [120, 120, 120]),
            }
        )

    if not map_rows:
        st.info("No GPS-tagged detections match the current filters.")
        return

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=map_rows,
            get_position="[longitude, latitude]",
            get_fill_color="color",
            get_radius=10,
            radius_min_pixels=7,
            radius_max_pixels=18,
            pickable=True,
        )
    ]

    route_records = load_gps_records(gps_path)
    if len(route_records) >= 2:
        layers.insert(
            0,
            pdk.Layer(
                "PathLayer",
                data=[
                    {
                        "path": [
                            [record["longitude"], record["latitude"]]
                            for record in route_records
                        ]
                    }
                ],
                get_path="path",
                get_color=[75, 120, 200],
                get_width=4,
                width_min_pixels=2,
            ),
        )

    center = map_rows[0]
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=center["latitude"], longitude=center["longitude"], zoom=14, pitch=0
        ),
        tooltip={
            "html": (
                "<b>{damage_type}</b><br/>Confidence: {confidence}<br/>"
                "Severity: {severity} ({severity_score})<br/>"
                "Priority: {priority_level} ({priority_score})<br/>"
                "Timestamp: {timestamp_seconds}s"
            )
        },
    )
    st.pydeck_chart(deck, width="stretch")


def _render_table(detections):
    rows = [
        {
            "ID": detection.get("id"),
            "Class ID": detection.get("class_id"),
            "Damage Type": detection.get("damage_type"),
            "Confidence": round(float(detection.get("confidence", 0)), 3),
            "Severity": detection.get("severity"),
            "Severity Score": detection.get("severity_score"),
            "Priority": detection.get("priority_level"),
            "Priority Score": detection.get("priority_score"),
            "Timestamp (s)": detection.get("timestamp_seconds"),
            "Latitude": detection.get("latitude"),
            "Longitude": detection.get("longitude"),
        }
        for detection in detections
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_details(detections, results_folder):
    if not detections:
        st.info("No detection is available for the current filters.")
        return

    selected = st.selectbox(
        "Select a detection",
        detections,
        format_func=lambda item: f"#{item['id']} — {item['damage_type']} at {item['timestamp_seconds']}s",
    )
    image_path = Path(results_folder) / Path(selected["frame"]).name
    if image_path.is_file():
        st.image(str(image_path), caption=f"Annotated frame: {selected['frame']}")
    else:
        st.warning("Annotated frame image is not available for this detection.")

    left, right = st.columns(2)
    with left:
        st.json(
            {
                "damage_type": selected.get("damage_type"),
                "class_id": selected.get("class_id"),
                "confidence": selected.get("confidence"),
                "bbox": selected.get("bbox"),
                "severity_score": selected.get("severity_score"),
                "severity": selected.get("severity"),
            }
        )
        st.caption(
            "Severity is a reproducible prototype heuristic for RoadDamages only, "
            "based on bounding-box extent and a small confidence-reliability component."
        )
    with right:
        st.json(
            {
                "latitude": selected.get("latitude"),
                "longitude": selected.get("longitude"),
                "gps_timestamp": selected.get("gps_timestamp"),
                "gps_match_method": selected.get("gps_match_method"),
                "gps_time_difference_seconds": selected.get("gps_time_difference_seconds"),
            }
        )
        st.write("**Priority reasoning**")
        st.write(selected.get("priority_reasoning", selected.get("priority_reason", "No priority reasoning available.")))


def _render_repair_order(detections):
    if not detections:
        st.info("No repair recommendations are available.")
        return
    for level in ("Critical", "High", "Medium", "Low"):
        level_detections = sorted(
            (
                item
                for item in detections
                if item.get("damage_type") == ROAD_DAMAGE_CLASS
                and item.get("priority_level") == level
            ),
            key=lambda item: item.get("priority_score") or 0,
            reverse=True,
        )
        if level_detections:
            st.markdown(f"**{level} locations**")
            for detection in level_detections:
                st.write(
                    f"#{detection['id']} — {detection['damage_type']} at "
                    f"{detection['timestamp_seconds']}s, score {detection['priority_score']}"
                )


def _render_methodology():
    with st.expander("Methodology and prototype limitations"):
        st.markdown(
            "- **Frame sampling:** one frame is sampled at the selected interval and "
            "checked with Laplacian-variance blur and mean brightness measurements.\n"
            "- **YOLO detection:** the active RAD YOLO11n model identifies HMV, LMV, "
            "Pedestrian, RoadDamages, SpeedBump, and UnsurfacedRoad.\n"
            "- **Severity:** an explainable prototype heuristic applies only to the broad "
            "RoadDamages class. It does not classify potholes, cracks, manholes, or erosion.\n"
            "- **GPS:** timestamps are interpolated between supplied GPS readings where possible, "
            "or matched to the nearest reading with the time difference shown.\n"
            "- **Repair priority:** a prototype decision-support score applies to RoadDamages and combines severity, damage "
            "risk, and optional road-context/traffic inputs. Road context and traffic are not "
            "measured by RoadSense."
        )


st.markdown("<h1 style='margin-bottom:0'>RoadSense</h1>", unsafe_allow_html=True)
st.markdown("### Spotting Trouble Before It Spreads")
st.caption("AI-powered proactive road damage inspection and repair prioritization.")

with st.sidebar:
    st.header("Analyze a dashcam video")
    video_file = st.file_uploader("Dashcam MP4", type=["mp4"])
    gps_file = st.file_uploader("Optional GPS CSV or JSON", type=["csv", "json"])
    sampling_interval = st.number_input(
        "Sampling interval (seconds)", min_value=0.1, value=1.0, step=0.1
    )
    road_context = st.selectbox(
        "Optional road context (prototype)",
        ["Not supplied", "motorway", "arterial", "collector", "local"],
    )
    traffic_enabled = st.checkbox("Provide traffic factor (prototype)")
    traffic_factor = (
        st.slider("Traffic factor (0–100)", 0, 100, 50)
        if traffic_enabled
        else None
    )
    analyze_clicked = st.button("Analyze Road", type="primary", use_container_width=True)

if analyze_clicked:
    if video_file is None:
        st.sidebar.error("Upload an MP4 dashcam video before analysis.")
    else:
        with st.spinner("Running the RoadSense pipeline..."):
            run_data = _run_analysis(
                video_file, gps_file, sampling_interval, road_context, traffic_factor
            )
        if run_data["success"]:
            st.session_state["roadsense_run"] = run_data
            st.success("Road analysis complete.")
        else:
            st.error(f"Road analysis failed: {run_data['error']}")

run_data = st.session_state.get("roadsense_run")
if run_data is None:
    st.info("Upload a dashcam MP4 and select Analyze Road to generate a dashboard.")
    _render_methodology()
    st.stop()

detections = run_data["detections"]
statistics = run_data["statistics"]
damage_types = sorted({str(item.get("damage_type", "unknown")) for item in detections})

st.header("1. Processing Summary")
_render_summary(statistics, detections)

st.header("2. Filters")
filter_columns = st.columns(3)
severity_filter = filter_columns[0].selectbox(
    "Severity", ["All", "Minor", "Moderate", "Severe", "N/A"]
)
damage_filter = filter_columns[1].selectbox("Damage Type", ["All", *damage_types])
priority_filter = filter_columns[2].selectbox(
    "Priority", ["All", "Critical", "High", "Medium", "Low"]
)
filtered = _filtered_detections(
    detections, severity_filter, damage_filter, priority_filter
)
st.caption(f"Showing {len(filtered)} of {len(detections)} detections after filtering.")

st.header("3. Interactive Road Damage Map")
if run_data.get("gps_path") and "demo" in Path(run_data["gps_path"]).name.lower():
    st.warning("This run uses a DEMO GPS file; its coordinates are not claimed as dashcam capture.")
elif run_data.get("gps_path") is None:
    st.info("No GPS file was supplied. GPS-tagged markers and route are unavailable.")
_render_map(filtered, run_data.get("gps_path"))

st.header("4. Detection Table")
_render_table(filtered)

st.header("5. Detection Details")
_render_details(filtered, run_data["results_folder"])

st.header("6. Decision Support: Recommended repair order")
_render_repair_order(filtered)

st.header("7. Methodology")
_render_methodology()
