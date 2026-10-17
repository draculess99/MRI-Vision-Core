# AMD curated benchmark results

*Research/demo benchmark only; not clinical accuracy or medical performance.*

Run: `2026-10-17T19:45:42.899861+00:00`  |  scenario set: `amd_mri_core_demo_v1`  |  schema: `amd-benchmark-runner/v1`

## Coverage

- Total curated scenarios: **22**
- Locally available: **22**
- Successfully inferred: **22**
- Unavailable: **0**
- Failed: **0**
- Inference coverage: **100.0%**

Metrics are computed only over locally available scenarios. Unavailable scenarios are reported separately and are never assigned fabricated predictions.

## Model ranking / classification metrics

Threshold: `0.5`  |  denominator: **22** successfully-inferred scenario(s).

- Macro ROC-AUC (12 valid targets): **0.599**
- Macro PR-AUC (12 valid targets): **0.485**

| Target | n | pos | neg | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| ACL | 22 | 9 | 13 | 0.419 | 0.375 |
| MCL | 22 | 4 | 18 | 0.458 | 0.210 |
| Medial Meniscus | 22 | 10 | 12 | 0.558 | 0.577 |
| Lateral Meniscus | 22 | 5 | 17 | 0.671 | 0.351 |
| Medial OA | 22 | 4 | 18 | 0.736 | 0.634 |
| Lateral OA | 22 | 4 | 18 | 0.667 | 0.332 |
| PF OA | 22 | 7 | 15 | 0.324 | 0.298 |
| Effusion | 22 | 10 | 12 | 0.592 | 0.620 |
| Synovitis | 22 | 9 | 13 | 0.752 | 0.688 |
| Baker's | 22 | 6 | 16 | 0.667 | 0.660 |
| Contusion | 22 | 8 | 14 | 0.652 | 0.605 |
| Fracture | 22 | 7 | 15 | 0.695 | 0.468 |

## Runtime (successfully-inferred scenarios only)

- Mean latency: **0.987s**
- Median latency: **0.961s**
- Min / max latency: **0.906s** / **1.204s**
- Total wall time: **21.723s** over 22 run(s)

## Limitations

- Only scenarios whose underlying study is present under the configured data_root are inferred; metrics never include unavailable or failed scenarios.
- ROC-AUC/PR-AUC per target require at least one positive and one negative successfully-inferred example; otherwise they are reported as not computable.
- The checkpoint is an externally trained, unevaluated research model bridged into this adapter (see hackathon/amd/kaggle_mil_bridge.py); probabilities are raw, uncalibrated sigmoid outputs.
- Metrics are computed only over locally available scenarios. Unavailable scenarios are reported separately and are never assigned fabricated predictions.

*Research/demo benchmark only; not clinical accuracy or medical performance.*
