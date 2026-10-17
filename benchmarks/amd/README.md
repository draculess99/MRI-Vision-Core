# AMD MRI-Core demo benchmark: 22 scenarios

Research/demo benchmark only; not clinical accuracy or medical performance.

Source: `amd_demo_scenarios.json` (schema 1.0, set `amd_mri_core_demo_v1`), consumed by the AMD MRI-Core Streamlit app on another machine through `study_id`. No DICOM, image, report text, model weight or local path is stored here.
Contract for the consumer: `CONSUMER.md`. Aggregate statistics: `amd_demo_scenarios_summary.json`.

## Why 22 and how it was generated
The local cache holds 58 RSNA studies with all 12 structured competition labels. 22 is enough to give every target a dedicated focused case (12 anchors), the highest multilabel burdens (2), a baseline (1),
structural outliers (2) and farthest-point label/structure diversity (5), while staying short enough for a selectbox and a README table. Selection is deterministic (no randomness; ties broken by study ID):
1. target anchors: for each target, from the rarest to the most common, the unselected study positive for it with the fewest other positives;
2. the two highest multilabel burdens; 3. the lowest-complexity remaining study; 4. two structural outliers (series / DICOM / plane structure);
5. the remaining slots by farthest-point selection on a combined distance (0.6 x label Hamming + 0.4 x structural distance).
Tags are rules on label and structure properties (not clinical judgements). Scenarios are ordered simplest first. Reference labels are those supplied by the competition; nothing is added clinically.

Regenerate (needs the local RSNA cache; `--check` fails if the committed files differ):

```
python scripts/build_amd_demo_scenarios.py
python scripts/build_amd_demo_scenarios.py --check
```

## Target coverage (22 scenarios)
| Target | scenarios positive | scenarios negative |
|---|---|---|
| ACL | 9 | 13 |
| MCL | 4 | 18 |
| Medial Meniscus | 10 | 12 |
| Lateral Meniscus | 5 | 17 |
| Medial OA | 4 | 18 |
| Lateral OA | 4 | 18 |
| PF OA | 7 | 15 |
| Effusion | 10 | 12 |
| Synovitis | 9 | 13 |
| Baker's | 6 | 16 |
| Contusion | 8 | 14 |
| Fracture | 7 | 15 |

Distribution of positive-label counts: 1 positive: 7, 2 positive: 4, 3 positive: 2, 4 positive: 1, 6 positive: 2, 7 positive: 4, 9 positive: 2 (min / median / max = [1, 2.5, 9]).
Distinct label vectors: 22 of 22; series count range [4, 9]; DICOM count range [98, 572]; plane-structure patterns: 10.

## Scenario-type counts
| scenario_type | count |
|---|---|
| high_multilabel | 2 |
| label_diversity | 5 |
| low_label_count | 1 |
| structural_diversity | 2 |
| target_anchor | 12 |

## Scenarios
| ID | Title | Type | Positive targets | Series | DICOM files |
|---|---|---|---|---|---|
| AMD-001 | Effusion-focused / isolated finding | target_anchor | 1 | 5 | 113 |
| AMD-002 | PF OA-focused / isolated finding | target_anchor | 1 | 6 | 114 |
| AMD-003 | Baker's-focused / isolated finding | target_anchor | 1 | 4 | 135 |
| AMD-004 | Lateral meniscus-focused / isolated finding | target_anchor | 1 | 5 | 160 |
| AMD-005 | Fracture-focused / isolated finding | target_anchor | 1 | 6 | 176 |
| AMD-006 | Medial meniscus-focused / isolated finding | target_anchor | 1 | 7 | 189 |
| AMD-007 | MCL-focused / isolated finding | target_anchor | 1 | 5 | 189 |
| AMD-008 | Synovitis-focused / low-complexity | target_anchor | 2 | 6 | 114 |
| AMD-009 | Contusion-focused / low-complexity | target_anchor | 2 | 4 | 114 |
| AMD-010 | Minimal findings (2 positive targets) | low_label_count | 2 | 5 | 122 |
| AMD-011 | ACL-focused / low-complexity | target_anchor | 2 | 9 | 233 |
| AMD-012 | Medial OA-focused / low-complexity | target_anchor | 3 | 6 | 191 |
| AMD-013 | Structural outlier / 9 series, 572 images | structural_diversity | 3 | 9 | 572 |
| AMD-014 | Lateral OA-focused / multi-finding | target_anchor | 4 | 7 | 180 |
| AMD-015 | Distinct label pattern / ACL, Medial meniscus, PF OA +3 | label_diversity | 6 | 6 | 165 |
| AMD-016 | Structural outlier / 5 series, 298 images | structural_diversity | 6 | 5 | 298 |
| AMD-017 | Distinct label pattern / Medial meniscus, Lateral meniscus, Medial OA +4 | label_diversity | 7 | 5 | 98 |
| AMD-018 | Distinct label pattern / ACL, Medial meniscus, Lateral meniscus +4 | label_diversity | 7 | 5 | 105 |
| AMD-019 | Distinct label pattern / ACL, MCL, PF OA +4 | label_diversity | 7 | 5 | 106 |
| AMD-020 | Distinct label pattern / ACL, MCL, Medial meniscus +4 | label_diversity | 7 | 4 | 110 |
| AMD-021 | High multilabel burden (9 of 12 targets) | high_multilabel | 9 | 6 | 213 |
| AMD-022 | High multilabel burden (9 of 12 targets) [7 series, 239 images] | high_multilabel | 9 | 7 | 239 |
