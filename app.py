"""RoadSense municipal road-condition intelligence console."""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pydeck as pdk
import streamlit as st

from gps_mapping import load_gps_records
from feature_status import build_feature_status
from pipeline import run_image_pipeline, run_pipeline


ROOT = Path(__file__).parent
UPLOADS_FOLDER = ROOT / "uploads"
RUNS_FOLDER = ROOT / "results" / "runs"
ROAD_DAMAGE_CLASS = "RoadDamages"
SEVERITY_COLORS = {
    "Minor": [86, 164, 109],
    "Moderate": [222, 164, 55],
    "Severe": [205, 82, 71],
}


st.set_page_config(page_title="RoadSense | Road Condition Intelligence", page_icon="RS", layout="wide")


def _inject_console_style():
    """Apply a restrained operations-console visual layer to Streamlit."""
    st.markdown(
        """
        <style>
        :root { --rs-accent:#58b7c4; --rs-panel:#14191e; --rs-line:#29343d;
                --rs-text:#f1f4f5; --rs-muted:#9ba7af; }
        .stApp { background:#0b0e11; color:var(--rs-text); }
        [data-testid="stHeader"] { background:rgba(11,14,17,.92); }
        section[data-testid="stSidebar"] { background:#10151a; border-right:1px solid var(--rs-line); }
        section[data-testid="stSidebar"] > div { padding-top:1.2rem; }
        h1, h2, h3 { color:var(--rs-text); letter-spacing:-.02em; }
        div[data-testid="stMetric"] { background:var(--rs-panel); border:1px solid var(--rs-line);
          border-radius:4px; padding:.8rem .9rem; }
        div[data-testid="stMetricLabel"] { color:var(--rs-muted); font-size:.68rem;
          font-weight:700; letter-spacing:.09em; text-transform:uppercase; }
        div[data-testid="stMetricValue"] { color:var(--rs-text); font-size:1.65rem; }
        .rs-kicker { color:var(--rs-accent); font-size:.71rem; font-weight:700;
          letter-spacing:.14em; text-transform:uppercase; margin-bottom:.35rem; }
        .rs-title { color:var(--rs-text); font-size:2.25rem; font-weight:760;
          line-height:1; letter-spacing:.03em; margin:0; }
        .rs-subtitle { color:var(--rs-muted); font-size:.93rem; margin-top:.42rem; }
        .rs-status { display:inline-block; color:#a9e1d7; border:1px solid #28645d;
          background:#11231f; border-radius:2px; padding:.24rem .48rem; font-size:.65rem;
          font-weight:700; letter-spacing:.09em; text-transform:uppercase; }
        .rs-panel { background:var(--rs-panel); border:1px solid var(--rs-line); border-radius:5px;
          padding:1rem 1.05rem; margin:.35rem 0 1rem 0; }
        .rs-section-label { color:var(--rs-accent); font-size:.68rem; font-weight:700;
          letter-spacing:.13em; text-transform:uppercase; margin-bottom:.26rem; }
        .rs-section-title { color:var(--rs-text); font-size:1.25rem; font-weight:680; margin:0; }
        .rs-section-copy { color:var(--rs-muted); font-size:.82rem; margin:.3rem 0 0; }
        .rs-flow { display:flex; flex-wrap:wrap; align-items:center; gap:.45rem; color:#b8c2c8;
          font-size:.68rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
        .rs-flow-step { border:1px solid var(--rs-line); background:#10151a; padding:.42rem .55rem; border-radius:3px; }
        .rs-flow-arrow { color:var(--rs-accent); }
        .rs-data-label { color:var(--rs-muted); font-size:.65rem; font-weight:700;
          letter-spacing:.1em; text-transform:uppercase; margin-bottom:.16rem; }
        .rs-data-value { color:var(--rs-text); font-size:1rem; font-weight:650; margin-bottom:.75rem; }
        .rs-chip { display:inline-block; padding:.24rem .45rem; margin:.1rem .18rem .1rem 0;
          border-radius:2px; font-size:.68rem; font-weight:700; letter-spacing:.05em; }
        .rs-chip-minor { color:#a9d9b4; background:#173022; border:1px solid #326342; }
        .rs-chip-moderate { color:#f4d18b; background:#362a13; border:1px solid #785b25; }
        .rs-chip-severe { color:#f1aaa2; background:#371917; border:1px solid #7f3530; }
        .rs-chip-na { color:#b4bec4; background:#20272d; border:1px solid #3b474f; }
        .rs-empty { border:1px dashed #3a464e; background:#10151a; padding:1.4rem; border-radius:4px;
          color:var(--rs-muted); text-align:center; }
        .rs-priority-row { border-left:3px solid var(--rs-accent); background:#11171c;
          border-top:1px solid var(--rs-line); border-right:1px solid var(--rs-line);
          border-bottom:1px solid var(--rs-line); padding:.7rem .8rem; margin:.45rem 0; }
        .stButton > button { width:100%; border-radius:3px; border:1px solid #5bb9c5;
          background:#1d5862; color:#f5fbfc; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
        .stButton > button:hover { border-color:#82d2dc; background:#236b75; color:white; }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:#10151a; border-color:#34424b; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_uploaded_file(uploaded_file, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as file:
        file.write(uploaded_file.getbuffer())


def _run_analysis(media_file, gps_file, sampling_interval, road_context, traffic_factor, minimum_confidence, temporal_confirmation_frames, preprocess_frames):
    """Persist uploaded inputs and invoke the existing end-to-end pipeline."""
    run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run_folder = RUNS_FOLDER / run_id
    frames_folder = run_folder / "frames"
    results_folder = run_folder / "results"
    media_path = UPLOADS_FOLDER / run_id / Path(media_file.name).name
    _save_uploaded_file(media_file, media_path)

    gps_path = None
    if gps_file is not None:
        gps_path = UPLOADS_FOLDER / run_id / Path(gps_file.name).name
        _save_uploaded_file(gps_file, gps_path)

    is_image = media_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    if is_image:
        result = run_image_pipeline(
            image_path=media_path, results_folder=results_folder, gps_data_path=gps_path,
            road_context=road_context if road_context != "Not supplied" else None,
            traffic_factor=traffic_factor, minimum_confidence=minimum_confidence,
            preprocess=preprocess_frames,
        )
    else:
        result = run_pipeline(
            video_path=media_path, frames_folder=frames_folder, results_folder=results_folder,
            gps_data_path=gps_path, every_seconds=sampling_interval,
            road_context=road_context if road_context != "Not supplied" else None,
            traffic_factor=traffic_factor, minimum_confidence=minimum_confidence,
            temporal_confirmation_frames=temporal_confirmation_frames,
            preprocess_frames=preprocess_frames,
        )
    if not result["success"]:
        return result

    return {
        **result,
        "run_id": run_id,
        "results_folder": str(results_folder),
        "gps_path": str(gps_path) if gps_path else None,
        "media_type": "image" if is_image else "video",
        "detections": _load_json(result["final_detections_path"]),
    }


def _filtered_detections(detections, severity_filter, damage_filter, priority_filter, capture_date_filter="All"):
    return [
        detection
        for detection in detections
        if (severity_filter == "All" or detection.get("severity") == severity_filter)
        and (
            damage_filter == "All"
            or str(detection.get("damage_type", "")).lower() == damage_filter.lower()
        )
        and (priority_filter == "All" or detection.get("priority_level") == priority_filter)
        and (capture_date_filter == "All" or detection.get("capture_date") == capture_date_filter)
    ]


def _format_timestamp(timestamp):
    if timestamp is None:
        return "N/A"
    minutes, seconds = divmod(float(timestamp), 60)
    return f"{int(minutes):02d}:{seconds:04.1f}"


def _format_percent(value):
    return "N/A" if value is None else f"{float(value):.0%}"


def _format_score(value):
    return "N/A" if value is None else f"{float(value):.1f} / 100"


def _chip(value, kind="severity"):
    normalized = str(value or "N/A").lower()
    if kind == "severity":
        css_class = {"minor": "minor", "moderate": "moderate", "severe": "severe"}.get(normalized, "na")
    else:
        css_class = {"low": "minor", "medium": "moderate", "high": "severe", "critical": "severe"}.get(normalized, "na")
    return f'<span class="rs-chip rs-chip-{css_class}">{value or "N/A"}</span>'


def _section_heading(label, title, copy=""):
    st.markdown(
        f'<div class="rs-section-label">{label}</div><p class="rs-section-title">{title}</p>'
        f'<p class="rs-section-copy">{copy}</p>',
        unsafe_allow_html=True,
    )


def _render_pipeline_strip(run_data):
    statistics = run_data["statistics"]
    st.markdown('<div class="rs-panel">', unsafe_allow_html=True)
    _section_heading(
        "SCAN STATUS",
        "Pipeline complete",
        ("Image processed through the local RoadSense workflow." if run_data.get("media_type") == "image"
         else f"{statistics.get('sampled_frames', 0)} sampled frames processed through the local RoadSense workflow."),
    )
    st.markdown(
        ("""<div class="rs-flow">
        <span class="rs-flow-step">Frame extraction</span><span class="rs-flow-arrow">&rarr;</span>
        <span class="rs-flow-step">Quality check</span><span class="rs-flow-arrow">&rarr;</span>
        <span class="rs-flow-step">YOLO11 detection</span><span class="rs-flow-arrow">&rarr;</span>
        <span class="rs-flow-step">Severity</span><span class="rs-flow-arrow">&rarr;</span>
        <span class="rs-flow-step">GPS sync</span><span class="rs-flow-arrow">&rarr;</span>
        <span class="rs-flow-step">Priority</span></div>"""),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_overview(statistics, detections):
    road_damages = [item for item in detections if item.get("priority_score") is not None]
    road_conditions = [item for item in detections if item.get("detection_group") == "road_defect_condition"]
    context_objects = [item for item in detections if item.get("detection_group") == "road_context"]
    high_priority = [
        item for item in road_damages if item.get("priority_level") in {"Critical", "High"}
    ]
    gps_matched = sum(item.get("gps_match_method") != "unavailable" for item in detections)
    _section_heading("RESULTS OVERVIEW", "Road scan summary", "Values are generated from the current pipeline run only.")
    columns = st.columns(6)
    is_image = statistics.get("media_type") == "image"
    columns[0].metric("Image analyzed" if is_image else "Frames processed", 1 if is_image else statistics.get("sampled_frames", 0))
    columns[1].metric("Usable input" if is_image else "Usable frames", statistics.get("usable_inputs", 0) if is_image else statistics.get("usable_frames", 0))
    columns[2].metric("Total detections", len(detections))
    columns[3].metric("Road defects / conditions", len(road_conditions))
    columns[4].metric("Context objects", len(context_objects))
    columns[5].metric("GPS matched", gps_matched)
    if high_priority:
        st.caption(f"{len(high_priority)} repair-relevant road-damage detection(s) currently rank High or Critical in prototype maintenance prioritization.")


def _render_map(detections, gps_path, embedded_records=None):
    map_rows = []
    for detection in detections:
        if detection.get("latitude") is None or detection.get("longitude") is None:
            continue
        map_rows.append(
            {
                **detection,
                "color": SEVERITY_COLORS.get(detection.get("severity"), [120, 132, 140]),
                "confidence_display": _format_percent(detection.get("confidence")),
                "severity_score_display": _format_score(detection.get("severity_score")),
                "priority_score_display": _format_score(detection.get("priority_score")),
            }
        )

    if not map_rows:
        st.markdown(
            '<div class="rs-empty"><b>GPS DATA UNAVAILABLE</b><br/>Vision analysis completed, but geographic placement is unavailable for the current filtered results.</div>',
            unsafe_allow_html=True,
        )
        return

    sources = sorted({str(item.get("gps_source", "Unavailable")) for item in map_rows})
    st.caption(f"GPS source: {', '.join(sources)}")

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
    route_records = embedded_records or load_gps_records(gps_path)
    if len(route_records) >= 2:
        layers.insert(
            0,
            pdk.Layer(
                "PathLayer",
                data=[{"path": [[record["longitude"], record["latitude"]] for record in route_records]}],
                get_path="path",
                get_color=[88, 183, 196],
                get_width=4,
                width_min_pixels=2,
            ),
        )

    center = map_rows[0]
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=center["latitude"], longitude=center["longitude"], zoom=14, pitch=0
            ),
            tooltip={
                "html": (
                    "<b>{damage_type}</b><br/>Confidence: {confidence_display}<br/>"
                    "Severity: {severity} ({severity_score_display})<br/>"
                    "Priority: {priority_level} ({priority_score_display})<br/>"
                    "Timestamp: {timestamp_seconds}s"
                )
            },
        ),
        width="stretch",
    )


def _render_evidence(detections, results_folder):
    _section_heading("DETECTION EVIDENCE", "Annotated frame review", "Select a pipeline detection to inspect the corresponding annotated evidence frame.")
    if not detections:
        st.markdown('<div class="rs-empty"><b>NO DETECTIONS FOUND</b><br/>The model did not produce detections for the selected frames and filters.</div>', unsafe_allow_html=True)
        return

    detections_by_id = {item["id"]: item for item in detections}
    selected_id = st.selectbox(
        "Detection evidence record",
        options=list(detections_by_id),
        format_func=lambda item_id: (
            f"#{item_id} | {detections_by_id[item_id]['damage_type']} | "
            f"{_format_timestamp(detections_by_id[item_id].get('timestamp_seconds'))}"
        ),
        label_visibility="collapsed",
    )
    selected = detections_by_id[selected_id]
    image_path = Path(results_folder) / Path(selected["frame"]).name
    left, right = st.columns([1.5, 1])
    with left:
        if image_path.is_file():
            st.image(str(image_path), caption=f"Annotated evidence | {selected['frame']}", width="stretch")
        else:
            st.markdown('<div class="rs-empty"><b>EVIDENCE IMAGE UNAVAILABLE</b><br/>The JSON record remains available, but its annotated frame file was not found.</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="rs-panel">', unsafe_allow_html=True)
        st.markdown('<div class="rs-data-label">Detection class</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rs-data-value">{selected.get("damage_type", "N/A")}</div>', unsafe_allow_html=True)
        if selected.get("damage_type") == ROAD_DAMAGE_CLASS:
            st.markdown('<div class="rs-data-label">Damage subtype</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="rs-data-value">{selected.get("damage_subtype", "Unknown / Not classified")}</div>', unsafe_allow_html=True)
        st.markdown('<div class="rs-data-label">Detection confidence</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rs-data-value">{_format_percent(selected.get("confidence"))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="rs-data-label">Timestamp</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rs-data-value">{_format_timestamp(selected.get("timestamp_seconds"))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="rs-data-label">GPS source</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rs-data-value">{selected.get("gps_source", "Unavailable")}</div>', unsafe_allow_html=True)
        st.markdown('<div class="rs-data-label">Severity</div>', unsafe_allow_html=True)
        st.markdown(_chip(selected.get("severity")), unsafe_allow_html=True)
        st.markdown('<div class="rs-data-label" style="margin-top:.65rem">Severity score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rs-data-value">{_format_score(selected.get("severity_score"))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="rs-data-label" style="margin-top:.65rem">Maintenance priority</div>', unsafe_allow_html=True)
        st.markdown(_chip(selected.get("priority_level"), "priority"), unsafe_allow_html=True)
        st.markdown('<div class="rs-data-label" style="margin-top:.65rem">Priority score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rs-data-value">{_format_score(selected.get("priority_score"))}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption(
            f"BBox: {selected.get('bbox')} | Image: {selected.get('image_width')} x {selected.get('image_height')}"
        )
        if selected.get("priority_reasoning"):
            st.caption(selected["priority_reasoning"])


def _render_detection_feed(detections):
    _section_heading("DETECTION FEED", "Chronological inspection log", "Filterable detections produced by the selected analysis run.")
    rows = [
        {
            "TIME": _format_timestamp(item.get("timestamp_seconds")),
            "CLASS": item.get("damage_type"),
            "CONFIDENCE": _format_percent(item.get("confidence")),
            "SEVERITY": item.get("severity"),
            "SEVERITY SCORE": _format_score(item.get("severity_score")),
            "PRIORITY": item.get("priority_level"),
            "PRIORITY SCORE": _format_score(item.get("priority_score")),
        }
        for item in sorted(detections, key=lambda item: item.get("timestamp_seconds") or 0)
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.markdown('<div class="rs-empty"><b>NO DETECTIONS MATCH FILTERS</b><br/>Adjust severity, class, or priority filters to review additional records.</div>', unsafe_allow_html=True)


def _render_priority(detections):
    _section_heading("MAINTENANCE PRIORITY", "Prototype maintenance prioritization", "Only repair-relevant road-damage records are ranked. This is decision support, not an autonomous repair decision.")
    ranked = sorted(
        (item for item in detections if item.get("priority_score") is not None),
        key=lambda item: item["priority_score"],
        reverse=True,
    )
    if not ranked:
        st.markdown('<div class="rs-empty"><b>NO ROAD DAMAGE PRIORITIES</b><br/>No repair-relevant road-damage detections match the current filters.</div>', unsafe_allow_html=True)
        return
    for item in ranked:
        st.markdown(
            f'<div class="rs-priority-row"><b>#{item["id"]} &middot; {item["damage_type"]}</b> '
            f'&mdash; {item.get("priority_level")} ({item.get("priority_score")}) '
            f'| Severity: {item.get("severity")} | { _format_timestamp(item.get("timestamp_seconds")) }</div>',
            unsafe_allow_html=True,
        )
        st.caption(item.get("priority_reasoning", item.get("priority_reason", "")))


def _render_quality_and_traceability(run_data, detections):
    """Expose actual preprocessing/GPS evidence for a judge-facing audit trail."""
    statistics = run_data["statistics"]
    results_folder = Path(run_data["results_folder"])
    is_image = statistics.get("media_type") == "image"
    metadata_path = (
        results_folder / "image_metadata.json"
        if is_image
        else results_folder.parent / "frames" / "frame_metadata.json"
    )
    metadata = _load_json(metadata_path) if metadata_path.is_file() else {}
    with st.expander("DATA QUALITY & TRACEABILITY", expanded=False):
        st.caption("Audit values below are measured from this run's saved metadata and detections; they are not performance claims.")
        if is_image:
            columns = st.columns(3)
            columns[0].metric("Blur score", metadata.get("blur_score", "N/A"))
            columns[1].metric("Brightness", metadata.get("brightness_score", "N/A"))
            columns[2].metric("Quality flag", "Usable" if metadata.get("usable") else "Flagged")
        else:
            frames = metadata.get("frames", [])
            usable = sum(bool(frame.get("usable")) for frame in frames if isinstance(frame, dict))
            flagged = len(frames) - usable
            columns = st.columns(4)
            columns[0].metric("Sampled frames", len(frames))
            columns[1].metric("Usable frames", usable)
            columns[2].metric("Flagged frames", flagged)
            columns[3].metric("Sampling interval", metadata.get("video", {}).get("sampling", {}).get("interval", "N/A"))
            st.caption("Frame usability uses Laplacian-variance blur and grayscale-brightness heuristics; flagged frames remain recorded for audit.")

        gps_methods = pd.Series([item.get("gps_match_method", "unavailable") for item in detections]).value_counts()
        if not gps_methods.empty:
            st.write("GPS match methods")
            st.dataframe(gps_methods.rename_axis("method").reset_index(name="detections"), width="stretch", hide_index=True)

        final_path = Path(run_data["final_detections_path"])
        statistics_path = Path(run_data["statistics_path"])
        download_columns = st.columns(2)
        if final_path.is_file():
            download_columns[0].download_button(
                "Download final detections JSON", final_path.read_bytes(),
                file_name="final_detections.json", mime="application/json",
            )
        if statistics_path.is_file():
            download_columns[1].download_button(
                "Download processing statistics", statistics_path.read_bytes(),
                file_name="processing_statistics.json", mime="application/json",
            )


def _render_feature_status(run_data, detections):
    """Expose only capabilities actually present in this local prototype."""
    has_gps = any(item.get("gps_match_method") != "unavailable" for item in detections)
    has_capture_date = any(item.get("capture_date") for item in detections)
    with st.expander("FEATURE STATUS", expanded=False):
        st.caption("Status reflects this local build and current run; it is not a model-performance claim.")
        st.dataframe(
            pd.DataFrame(build_feature_status(run_data.get("media_type"), has_gps, has_capture_date)),
            width="stretch", hide_index=True,
        )


def _render_methodology():
    with st.expander("HOW ROADSENSE WORKS | TECHNICAL NOTES", expanded=False):
        st.markdown(
            "**Video** -> **Frame extraction + quality checks** -> **YOLO11 detection** -> "
            "**Severity analysis** -> **GPS synchronization** -> **Priority scoring** -> "
            "**Interactive map + maintenance ranking**"
        )
        st.caption("Stack: Python, OpenCV, Ultralytics YOLO11, PyTorch/CUDA, RAD + RDD2022 road-damage models, Streamlit, PyDeck, Pandas, JSON/CSV.")
        st.markdown("**SEVERITY MODEL**")
        st.write(
            "Generic RoadDamages and any verified specific damage label use a reproducible rule-based severity heuristic based on detection characteristics and confidence. "
            "It is a prototype component and can later be replaced by a learned severity model."
        )
        st.markdown("**LIMITATIONS**")
        st.write(
            "The RAD taxonomy uses a broad RoadDamages class. The optional RDD2022 refinement model can identify pothole, longitudinal crack, transverse crack, and alligator crack only after a spatially matching secondary detection. "
            "This is a batch prototype, GPS quality depends on supplied data, and model recall can improve with more data and validation-driven optimization."
        )


_inject_console_style()
st.markdown(
    """<div style="display:flex;justify-content:space-between;align-items:end;gap:1rem;margin:.5rem 0 1.25rem 0;">
    <div><div class="rs-kicker">RoadSense</div><div class="rs-title">ROAD CONDITION INTELLIGENCE</div>
    <div class="rs-subtitle">Video &rarr; Evidence &rarr; Location &rarr; Action</div></div>
    <div class="rs-status">&#9679; Local inference</div></div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="rs-kicker">Road scan</div><h3 style="margin-top:0">Analysis input</h3>', unsafe_allow_html=True)
    st.caption("Upload a dashcam recording to begin local analysis.")
    media_file = st.file_uploader("Road media", type=["mp4", "mov", "avi", "mkv", "jpg", "jpeg", "png", "webp"])
    gps_file = st.file_uploader("Optional GPS CSV, GPX, or JSON", type=["csv", "gpx", "json"])
    sampling_interval = st.number_input("Sampling interval (seconds)", min_value=0.1, value=1.0, step=0.1)
    minimum_confidence = st.slider("Minimum confidence", min_value=0.15, max_value=0.80, value=0.25, step=0.05,
                                   help="Acceptance floor. Individual RAD and damage-model classes also retain their configured thresholds.")
    temporal_confirmation_frames = st.selectbox("Video temporal confirmation", [1, 2, 3], index=0,
                                                help="Use 2+ to suppress isolated road-condition detections. 1 keeps all accepted frames.")
    preprocess_frames = st.checkbox("Explicit 512 px letterbox normalization", value=False,
                                    help="Optional in-memory preprocessing that preserves aspect ratio and maps boxes back to source coordinates.")
    road_context = st.selectbox("Road context (prototype input)", ["Not supplied", "motorway", "arterial", "collector", "local"])
    traffic_enabled = st.checkbox("Provide traffic factor (prototype)")
    traffic_factor = st.slider("Traffic factor (0-100)", 0, 100, 50) if traffic_enabled else None
    analyze_clicked = st.button("Analyze Road", type="primary")
    st.markdown("---")
    st.caption("Local batch prototype. No cloud inference or GPS accuracy claim is implied.")

if analyze_clicked:
    if media_file is None:
        st.sidebar.error("Upload a supported road video or image before starting analysis.")
    else:
        with st.status("ANALYSIS IN PROGRESS", expanded=True) as status:
            st.write("Running the local RoadSense pipeline on the uploaded road media.")
            run_data = _run_analysis(media_file, gps_file, sampling_interval, road_context, traffic_factor, minimum_confidence, temporal_confirmation_frames, preprocess_frames)
            if run_data["success"]:
                statistics = run_data["statistics"]
                units = "image" if run_data.get("media_type") == "image" else f"{statistics.get('sampled_frames', 0)} sampled frame(s)"
                st.write(f"Pipeline completed: {units}, {len(run_data['detections'])} detection(s).")
                status.update(label="ANALYSIS COMPLETE", state="complete", expanded=False)
                st.session_state["roadsense_run"] = run_data
            else:
                status.update(label="ANALYSIS FAILED", state="error", expanded=True)
                st.error(run_data["error"])

run_data = st.session_state.get("roadsense_run")
if run_data is None:
    st.markdown('<div class="rs-empty"><b>NO ROAD SCAN LOADED</b><br/>Upload a dashcam video or road image in the Road Scan panel to begin.</div>', unsafe_allow_html=True)
    _render_methodology()
    st.stop()

detections = run_data["detections"]
statistics = run_data["statistics"]
_render_pipeline_strip(run_data)
_render_overview(statistics, detections)
if statistics.get("media_type") == "image" and statistics.get("external_gps_ignored"):
    st.info("External timestamped GPS was supplied, but no video-relative timestamp exists for this image. Only embedded image EXIF GPS is used.")

st.markdown('<div class="rs-panel">', unsafe_allow_html=True)
_section_heading("FILTERS", "Inspection controls", "Filters update the map, evidence view, feed, and maintenance ranking.")
damage_types = sorted({str(item.get("damage_type", "unknown")) for item in detections})
capture_dates = sorted({str(item["capture_date"]) for item in detections if item.get("capture_date")})
filter_columns = st.columns(4 if capture_dates else 3)
severity_filter = filter_columns[0].selectbox("Severity", ["All", "Minor", "Moderate", "Severe", "N/A"])
damage_filter = filter_columns[1].selectbox("Detection class", ["All", *damage_types])
priority_filter = filter_columns[2].selectbox("Maintenance priority", ["All", "Critical", "High", "Medium", "Low", "N/A"])
capture_date_filter = filter_columns[3].selectbox("Capture date", ["All", *capture_dates]) if capture_dates else "All"
if not capture_dates:
    st.caption("Capture-date filtering is unavailable because this input has no reliable embedded capture date.")
filtered = _filtered_detections(detections, severity_filter, damage_filter, priority_filter, capture_date_filter)
st.caption(f"{len(filtered)} of {len(detections)} detection(s) shown.")
st.markdown("</div>", unsafe_allow_html=True)

map_column, evidence_column = st.columns([1.35, 1])
with map_column:
    _section_heading("ROAD CONDITION MAP", "Location-aware inspection", "Detected road conditions synchronized with available location data.")
    if run_data.get("gps_path") and "demo" in Path(run_data["gps_path"]).name.lower():
        st.warning("DEMO GPS DATA: coordinates are demonstration data, not claimed dashcam capture.")
    elif run_data.get("statistics", {}).get("gps_source") == "Unavailable":
        gps_message = run_data.get("statistics", {}).get("embedded_video_gps", {}).get("message")
        st.info(
            "VISION ANALYSIS COMPLETE: GPS coordinates could not be extracted from this video, "
            "so geographic placement is unavailable."
            + (f" {gps_message}" if gps_message else "")
        )
    _render_map(filtered, run_data.get("gps_path"), run_data.get("statistics", {}).get("gps_records"))
with evidence_column:
    _render_evidence(filtered, run_data["results_folder"])

_render_detection_feed(filtered)
_render_priority(filtered)
_render_quality_and_traceability(run_data, detections)
_render_feature_status(run_data, detections)
_render_methodology()
