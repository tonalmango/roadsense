# RoadSense

RoadSense is a local, batch-oriented road-condition inspection prototype for the
PK01PS001 challenge: **“RoadSense: Spotting Trouble Before It Spreads.”**

It turns dashcam media into a reviewable inspection record:

```text
Video or image → quality metadata → YOLO detection → prototype severity
→ timestamp/GPS matching → prototype maintenance priority → map and JSON evidence
```

## Run the dashboard

```powershell
python -m streamlit run app.py
```

Upload an MP4/MOV/AVI/MKV video or JPG/JPEG/PNG/WebP image. GPS CSV/JSON is
optional. The supplied `gps/demo_gps.csv` is explicitly synthetic demonstration
data and must not be presented as real dashcam capture.

## Pipeline command

```powershell
python pipeline.py videos\demo_road.mp4 --gps-data gps\demo_gps.csv --every-seconds 1 --min-confidence 0.25
```

Outputs are written per run under `results/runs/<run_id>/` in the dashboard, or
to the requested folders for the command-line pipeline.

## Road intelligence modules

The JSON detection records can be passed to these focused capabilities:

| Capability | Module |
| --- | --- |
| Temporal tracking and duplicate suppression | `temporal_tracking.py` |
| Pothole/road-damage evidence cards | `evidence_cards.py` |
| Road-condition health score | `health_score.py` |
| Smart repair queue | `repair_queue.py` |
| Road segment analytics | `segment_analytics.py` |
| Inspection comparison | `inspection_comparison.py` |
| Explainable severity and risk | `explainable_risk.py` |
| Confidence calibration | `confidence_calibration.py` |
| Mission replay | `mission_replay.py` |
| PDF inspection report | `inspection_report.py` |
| GPS/data-quality layer | `data_quality.py` |
| Active-learning feedback selection | `active_learning.py` |

These modules are deliberately model-agnostic and return JSON-serializable
objects, so they can be called from the Streamlit dashboard, batch scripts, or
future API endpoints. `create_pdf_report` requires the `reportlab` dependency.

## Active model profiles

The default profile is configured in `model_config.py`. It uses the RAD model
when available, and automatically uses the compatible local `model/best.pt`
profile when the RAD checkpoint is not present. The available local fallback
contains `Pothole`, `Crack`, and `Manhole` classes; it is not silently treated
as the six-class RAD model.

```text
training\rad_yolo11m\weights\best.pt
```

To explicitly select a profile, set `ROADSENSE_MODEL_PROFILE=rad`,
`ROADSENSE_MODEL_PROFILE=pothole`, or `ROADSENSE_MODEL_PROFILE=rdd2022`.

The RAD class taxonomy is:

```text
HMV, LMV, Pedestrian, RoadDamages, SpeedBump, UnsurfacedRoad
```

`RoadDamages` is a broad class. It is **not** a separate pothole, crack,
manhole, or erosion classifier. Severity and maintenance priority are
reproducible prototype heuristics applied only to `RoadDamages`; confidence is
detection certainty, not model accuracy.

## GPS data format

External video GPS uses CSV columns:

```csv
timestamp_seconds,latitude,longitude
0.0,20.2961,85.8245
```

The system interpolates between valid records when possible and marks GPS as
unavailable rather than guessing a location.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m py_compile app.py pipeline.py detection.py frame_extraction.py gps_mapping.py severity.py priority.py
```

## Scope and limitations

- This is a local batch prototype, not a production real-time service.
- Severity is rule-based; it does not measure road depth or structural condition.
- Priority is prototype decision support, not an autonomous municipal decision.
- GPS quality depends on the supplied source data.
- Large datasets, model checkpoints, generated frames, and run artifacts are intentionally ignored by Git.
