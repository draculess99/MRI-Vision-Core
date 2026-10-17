# Consuming the AMD demo scenarios (contract for the AMD MRI-Core Streamlit app)

Research/demo benchmark only; not clinical accuracy or medical performance.

## What the file is
`benchmarks/amd/amd_demo_scenarios.json` (schema `1.0`, set `amd_mri_core_demo_v1`) lists 22 benchmark scenarios built from RSNA Knee Abnormality Detection studies that carry all 12 competition labels.
It holds metadata only: study identifiers, the reference labels supplied by the competition, structure counts, tags and short descriptions. It contains no DICOM data, images, report text, model weights, credentials or local paths.
Pull the file from the repository; do not copy DICOM files into Git.

## Fields (per scenario)
`scenario_id` (`AMD-001` .. `AMD-022`, simplest first), `title` (selectbox label, unique), `study_id` (RSNA `StudyInstanceUID`), `scenario_type`, `tags`, `presentation_tag` (short badge),
`why_selected`, `expected_demo_behavior`, `positive_targets`, `positive_target_count`, `negative_target_count`, `label_vector` (12 ints in the order of the top-level `targets` list),
`series_count`, `dicom_count`, `planes`, `plane_series_counts`, `fluid_sensitive_series`, `fat_suppressed_series`, `data_source`. Top level: `schema_version`, `scenario_set`, `scenario_count`, `disclaimer`, `selection_method`, `targets`, `scenarios`.
Model-based tags (`borderline_model_case`, `model_label_disagreement`, `high_confidence_agreement`) and a `model_signals` object appear only if the file was generated with `--model-signals`; the shipped file does not contain them.

## Minimal Python
```python
import json
from pathlib import Path

data = json.loads(Path("benchmarks/amd/amd_demo_scenarios.json").read_text())
assert data["schema_version"] == "1.0" and data["scenario_count"] == len(data["scenarios"]) == 22
scenarios = data["scenarios"]
```

## Streamlit
```python
selected = st.selectbox(
    "Select benchmark scenario",
    scenarios,
    format_func=lambda x: f'{x["scenario_id"]} — {x["title"]}',
)
st.caption(selected["why_selected"])
st.caption(data["disclaimer"])
study_id = selected["study_id"]
```

## Resolving the study
The app must resolve `study_id` through its own configured MRI data root and inference provider (for example the `RSNAKneeMetadata` / inference engine it already uses), never from a path stored in the JSON.
The data root is a local, git-ignored RSNA cache (`train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm` plus `train.csv` / `train_series.csv`). If the study is not available on that machine, show the scenario as "data not available" instead of failing.
The `positive_targets` / `label_vector` fields are the competition-supplied reference labels for display and comparison; they are not a diagnosis, and the app's model probabilities must be presented as research outputs.

## Stability
Pin `schema_version` and `scenario_set`. New optional fields may be added in a minor revision; required fields will not be removed or renamed within `1.x`. To regenerate or verify against the source cache, see `README.md` in this folder.
