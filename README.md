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

## Active model profiles

The default profile is configured in `model_config.py` and currently expects:

```text
training\rad_yolo11m\weights\best.pt
```

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
