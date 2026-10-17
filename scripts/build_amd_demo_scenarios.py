"""Deterministically select the 22 benchmark/demo scenarios consumed by the AMD MRI-Core Streamlit app.

Candidates: the locally available RSNA studies with all 12 structured labels (competition-supplied) whose DICOM series directories exist locally.
Selection is a documented greedy diversity strategy (no randomness; every tie is broken by study ID):
  1. target anchors   - for each of the 12 targets, in ascending prevalence, the most focused (fewest other positives) unselected study positive for it
  2. high multilabel  - the two unselected studies with the most positive targets
  3. low label count  - the unselected study with the fewest positive targets
  4. structural       - two unselected studies maximising the minimum structural distance to the selection (series, DICOM and plane structure)
  5. label diversity  - remaining slots by farthest-point on a combined label (Hamming) + structural distance
Tags are deterministic rules on label/structure properties. Model-based tags are added only with --model-signals (out-of-fold fold checkpoints, computed after
selection, never used to select). Nothing here is clinical ground truth beyond the supplied competition labels; this is a research/demo benchmark.

  python scripts/build_amd_demo_scenarios.py                       # writes benchmarks/amd/*
  python scripts/build_amd_demo_scenarios.py --check               # regenerate in memory and fail if the committed files differ
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mri_core.rsna_knee_dataset import TARGET_COLUMNS, load_rsna_metadata  # noqa: E402

SCHEMA_VERSION = "1.0"
SCENARIO_SET = "amd_mri_core_demo_v1"
COUNT = 22
PLANES = ("Axial", "Coronal", "Sagittal")
DISCLAIMER = "Research/demo benchmark only; not clinical accuracy or medical performance."
SHORT = {"Medial Meniscus": "Medial meniscus", "Lateral Meniscus": "Lateral meniscus", "Medial OA": "Medial OA", "Lateral OA": "Lateral OA", "PF OA": "PF OA"}
W_LABEL, W_STRUCT = 0.6, 0.4


def discover_candidates(data_dir: Path, dicom_root: Path) -> list:
    """One record per labelled study whose series directories all exist locally. Reuses the repository's RSNA metadata loader."""
    md = load_rsna_metadata(data_dir, dicom_root)
    records = []
    for _, row in md.complete_label_train.sort_values("StudyInstanceUID").iterrows():
        uid = str(row["StudyInstanceUID"])
        series = md.series_for(uid)
        dicoms, complete = 0, True
        for _, s in series.iterrows():
            directory = md.dicom_series_dir(uid, str(s["SeriesInstanceUID"]))
            if not directory.is_dir():
                complete = False
                break
            dicoms += sum(1 for e in os.scandir(directory) if e.is_file())
        if not complete or series.empty:
            continue
        plane_counts = {p: int((series["Anatomical_Plane"].astype(str) == p).sum()) for p in PLANES}
        records.append({"study_id": uid, "labels": [int(row[t]) for t in TARGET_COLUMNS], "series_count": int(len(series)), "dicom_count": int(dicoms),
                        "plane_counts": plane_counts, "fluid_sensitive_series": int((series["Fluid_Sensitive"] == 1).sum()),
                        "fat_suppressed_series": int((series["Fat_Suppression"] == 1).sum())})
    return records


def structure_matrix(records: list) -> np.ndarray:
    raw = np.array([[r["series_count"], r["dicom_count"], *(r["plane_counts"][p] for p in PLANES), r["fluid_sensitive_series"]] for r in records], float)
    low, high = raw.min(axis=0), raw.max(axis=0)
    return (raw - low) / np.where(high > low, high - low, 1.0)


def select(records: list, count: int = COUNT) -> list:
    """Returns [(index, step)] in selection order."""
    if len(records) < count:
        raise ValueError(f"only {len(records)} labelled local studies; need {count}")
    order = sorted(range(len(records)), key=lambda i: records[i]["study_id"])  # stable tie-break
    labels = np.array([r["labels"] for r in records])
    struct = structure_matrix(records)
    pos = labels.sum(axis=1)
    chosen, steps = [], []

    def hamming(i, j):
        return float((labels[i] != labels[j]).sum()) / len(TARGET_COLUMNS)

    def sdist(i, j):
        return float(np.abs(struct[i] - struct[j]).mean())

    def min_dist(i, kind):
        if not chosen:
            return 1.0
        f = {"label": lambda j: hamming(i, j), "struct": lambda j: sdist(i, j), "combined": lambda j: W_LABEL * hamming(i, j) + W_STRUCT * sdist(i, j)}[kind]
        return min(f(j) for j in chosen)

    def add(i, step):
        chosen.append(i)
        steps.append(step)

    def pick(candidates, key):
        fresh = [i for i in candidates if not any((labels[i] == labels[j]).all() for j in chosen)]
        best = None
        for i in (fresh or candidates):  # a duplicate label vector is allowed only when no alternative remains  # candidates are in study-ID order, so `>` keeps the earliest on ties
            k = key(i)
            if best is None or k > best[0]:
                best = (k, i)
        return best[1]

    prevalence = labels.sum(axis=0)
    for t in sorted(range(len(TARGET_COLUMNS)), key=lambda t: (prevalence[t], t)):  # 1. rarest target first
        pool = [i for i in order if i not in chosen and labels[i, t] == 1]
        if pool:
            add(pick(pool, lambda i: (-pos[i], min_dist(i, "combined"))), f"target_anchor:{TARGET_COLUMNS[t]}")
    for _ in range(2):  # 2. highest multilabel burden
        add(pick([i for i in order if i not in chosen], lambda i: (pos[i], min_dist(i, "combined"))), "high_multilabel")
    add(pick([i for i in order if i not in chosen], lambda i: (-pos[i], min_dist(i, "combined"))), "low_label_count")  # 3.
    for _ in range(2):  # 4. structural outliers
        add(pick([i for i in order if i not in chosen], lambda i: min_dist(i, "struct")), "structural_diversity")
    while len(chosen) < count:  # 5. farthest point on label + structure
        add(pick([i for i in order if i not in chosen], lambda i: min_dist(i, "combined")), "label_diversity")
    return list(zip(chosen, steps))[:count]


def quantile(values, q):
    return float(np.quantile(np.array(values, float), q))


def build(records: list, selected: list) -> list:
    pos_all = [sum(r["labels"]) for r in records]
    q = {"low": quantile(pos_all, 0.25), "high": quantile(pos_all, 0.85), "series_lo": quantile([r["series_count"] for r in records], 0.10),
         "series_hi": quantile([r["series_count"] for r in records], 0.90), "dicom_lo": quantile([r["dicom_count"] for r in records], 0.10),
         "dicom_hi": quantile([r["dicom_count"] for r in records], 0.90)}
    pattern_count = {}
    for r in records:
        key = tuple(r["plane_counts"][p] for p in PLANES)
        pattern_count[key] = pattern_count.get(key, 0) + 1
    out = []
    for idx, step in selected:
        r = records[idx]
        positives = [t for t, v in zip(TARGET_COLUMNS, r["labels"]) if v]
        n_pos = len(positives)
        pattern = tuple(r["plane_counts"][p] for p in PLANES)
        structural = (r["series_count"] <= q["series_lo"] or r["series_count"] >= q["series_hi"] or r["dicom_count"] <= q["dicom_lo"]
                      or r["dicom_count"] >= q["dicom_hi"] or pattern_count[pattern] <= 2)
        tags = []
        if step.startswith("target_anchor"):
            tags.append("target_anchor")
        if n_pos <= q["low"]:
            tags.append("low_label_count")
        if n_pos >= q["high"]:
            tags.append("high_multilabel")
        if n_pos >= 7:
            tags.append("complex_case")
        if step == "label_diversity":
            tags.append("label_diversity")
        if structural:
            tags.append("structural_diversity")
        if 4 <= n_pos <= len(TARGET_COLUMNS) - 4 and structural:
            tags.append("review_candidate")
        scenario_type = step.split(":")[0]
        tags = sorted(set(tags) | {scenario_type})
        anchor = step.split(":")[1] if ":" in step else None
        if scenario_type == "target_anchor":
            kind = "isolated finding" if n_pos == 1 else ("low-complexity" if n_pos <= 3 else "multi-finding")
            title, badge = f"{SHORT.get(anchor, anchor)}-focused / {kind}", ("Simple" if n_pos <= 3 else "Focused")
        elif scenario_type == "high_multilabel":
            title, badge = f"High multilabel burden ({n_pos} of 12 targets)", "Complex"
        elif scenario_type == "low_label_count":
            title, badge = (("No positive targets / clear baseline", "Baseline") if n_pos == 0 else (f"Minimal findings ({n_pos} positive target{'s' if n_pos != 1 else ''})", "Simple"))
        elif scenario_type == "structural_diversity":
            title, badge = f"Structural outlier / {r['series_count']} series, {r['dicom_count']} images", "Outlier"
        else:
            lead = ", ".join(SHORT.get(t, t) for t in positives[:3]) + (f" +{n_pos - 3}" if n_pos > 3 else "") if positives else "no positives"
            title, badge = f"Distinct label pattern / {lead}", ("Complex" if n_pos >= 7 else "Diverse")
        out.append({
            "study_id": r["study_id"], "title": title, "scenario_type": scenario_type, "selection_step": step, "tags": tags, "presentation_tag": badge,
            "positive_targets": positives, "positive_target_count": n_pos, "negative_target_count": len(TARGET_COLUMNS) - n_pos, "label_vector": r["labels"],
            "series_count": r["series_count"], "dicom_count": r["dicom_count"], "planes": [p for p in PLANES if r["plane_counts"][p]],
            "plane_series_counts": r["plane_counts"], "fluid_sensitive_series": r["fluid_sensitive_series"], "fat_suppressed_series": r["fat_suppressed_series"],
            "why_selected": why(step, anchor, n_pos, r, tags),
            "expected_demo_behavior": (f"Resolve the study by study_id, show the deterministic quality status and the 12 model probabilities side by side; the competition reference labels are "
                                       f"positive for {', '.join(positives) if positives else 'no target'}. Probabilities are research outputs, not a diagnosis; a reviewer decision is required."),
            "data_source": "RSNA Knee Abnormality Detection"})
    out.sort(key=lambda s: (s["positive_target_count"], s["dicom_count"], s["study_id"]))  # presentation order: simple first, complex last
    seen = {}
    for s in out:  # titles must be unique inside the selectbox
        seen[s["title"]] = seen.get(s["title"], 0) + 1
        if seen[s["title"]] > 1:
            s["title"] += f" [{s['series_count']} series, {s['dicom_count']} images]"
    for k, s in enumerate(out, 1):
        s["scenario_id"] = f"AMD-{k:03d}"
    return [{"scenario_id": s.pop("scenario_id"), **s} for s in out]


def why(step, anchor, n_pos, r, tags) -> str:
    base = {"target_anchor": f"Coverage anchor for {anchor}: the most focused unselected study positive for it ({n_pos} positive target{'s' if n_pos != 1 else ''}).",
            "high_multilabel": f"Among the highest multilabel burdens in the local labelled set ({n_pos} of 12 targets positive).",
            "low_label_count": f"The lowest-complexity remaining study ({n_pos} positive targets), a clear baseline for the demo.",
            "structural_diversity": f"Structurally distant from the rest of the set ({r['series_count']} series, {r['dicom_count']} DICOM files, plane pattern "
                                    f"{'/'.join(str(r['plane_counts'][p]) for p in PLANES)}).",
            "label_diversity": "Added by farthest-point selection: its label pattern and study structure are the most different from the scenarios already chosen."}[step.split(":")[0]]
    extra = " Useful for the review/uncertainty demonstration." if "review_candidate" in tags else ""
    return base + extra


def model_signals(metadata, scenarios, ensemble_dir: Path):
    """Optional: out-of-fold probabilities (the fold whose validation set contains the study) added as tags. Computed after selection; never used to select."""
    import torch
    from mri_core.rsna_inference_engine import run_single_model_inference
    paths = sorted(ensemble_dir.glob("checkpoint_rsna_stratified_fold*.pt"))
    holdout = {}
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        for uid in (state.get("valid_study_ids") or []):
            holdout[str(uid)] = path
    mapped = sum(s["study_id"] in holdout for s in scenarios)
    if not mapped:
        print("WARNING: no checkpoint records its validation studies, so no out-of-fold fold can be assigned; no model tags were added")
    for s in scenarios:
        path = holdout.get(s["study_id"])
        if path is None:
            continue
        result = run_single_model_inference(metadata, path, s["study_id"])
        if result.probabilities is None:
            continue
        p = [float(result.probabilities[t]) for t in TARGET_COLUMNS]
        label = s["label_vector"]
        borderline = sum(abs(x - 0.5) < 0.05 for x in p)
        wrong = sum((x >= 0.5) != bool(y) for x, y in zip(p, label))
        s["model_signals"] = {"source": "out-of-fold fold checkpoint (research model, near-chance AUROC; not a diagnosis)", "borderline_targets": int(borderline),
                              "targets_disagreeing_with_label": int(wrong)}
        if borderline >= 6:
            s["tags"] = sorted(set(s["tags"]) | {"borderline_model_case"})
        if wrong >= 7:
            s["tags"] = sorted(set(s["tags"]) | {"model_label_disagreement"})
        if wrong <= 3:
            s["tags"] = sorted(set(s["tags"]) | {"high_confidence_agreement"})


def summarize(scenarios, candidates) -> dict:
    n = len(scenarios)
    pos_counts = [s["positive_target_count"] for s in scenarios]
    per_target = {t: {"positive_scenarios": sum(t in s["positive_targets"] for s in scenarios), "negative_scenarios": sum(t not in s["positive_targets"] for s in scenarios)}
                  for t in TARGET_COLUMNS}
    types, tag_counts = {}, {}
    for s in scenarios:
        types[s["scenario_type"]] = types.get(s["scenario_type"], 0) + 1
        for t in s["tags"]:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    vectors = {tuple(s["label_vector"]) for s in scenarios}
    ham = [sum(a != b for a, b in zip(x, y)) for i, x in enumerate(sorted(vectors)) for y in sorted(vectors)[i + 1:]]
    return {"scenario_count": n, "candidate_pool": len(candidates), "targets_with_a_positive_scenario": sum(v["positive_scenarios"] > 0 for v in per_target.values()),
            "per_target": per_target, "positive_count_distribution": {str(k): pos_counts.count(k) for k in sorted(set(pos_counts))},
            "positive_count_min_median_max": [min(pos_counts), float(np.median(pos_counts)), max(pos_counts)], "scenario_type_counts": dict(sorted(types.items())),
            "tag_counts": dict(sorted(tag_counts.items())), "series_count_range": [min(s["series_count"] for s in scenarios), max(s["series_count"] for s in scenarios)],
            "dicom_count_range": [min(s["dicom_count"] for s in scenarios), max(s["dicom_count"] for s in scenarios)],
            "distinct_label_vectors": len(vectors), "pairwise_hamming_min_mean_max": [min(ham), round(float(np.mean(ham)), 3), max(ham)] if ham else None,
            "plane_patterns": len({tuple(s["plane_series_counts"][p] for p in PLANES) for s in scenarios}),
            "model_tags_present": any("model_signals" in s for s in scenarios)}


def readme(doc, summary) -> str:
    cov = "\n".join(f"| {t} | {v['positive_scenarios']} | {v['negative_scenarios']} |" for t, v in summary["per_target"].items())
    types = "\n".join(f"| {k} | {v} |" for k, v in summary["scenario_type_counts"].items())
    dist = ", ".join(f"{k} positive: {v}" for k, v in summary["positive_count_distribution"].items())
    listing = "\n".join(f"| {s['scenario_id']} | {s['title']} | {s['scenario_type']} | {s['positive_target_count']} | {s['series_count']} | {s['dicom_count']} |" for s in doc["scenarios"])
    return f"""# AMD MRI-Core demo benchmark: 22 scenarios

{DISCLAIMER}

Source: `amd_demo_scenarios.json` (schema {SCHEMA_VERSION}, set `{SCENARIO_SET}`), consumed by the AMD MRI-Core Streamlit app on another machine through `study_id`. No DICOM, image, report text, model weight or local path is stored here.
Contract for the consumer: `CONSUMER.md`. Aggregate statistics: `amd_demo_scenarios_summary.json`.

## Why 22 and how it was generated
The local cache holds {summary['candidate_pool']} RSNA studies with all 12 structured competition labels. 22 is enough to give every target a dedicated focused case (12 anchors), the highest multilabel burdens (2), a baseline (1),
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
{cov}

Distribution of positive-label counts: {dist} (min / median / max = {summary['positive_count_min_median_max']}).
Distinct label vectors: {summary['distinct_label_vectors']} of {summary['scenario_count']}; series count range {summary['series_count_range']}; DICOM count range {summary['dicom_count_range']}; plane-structure patterns: {summary['plane_patterns']}.

## Scenario-type counts
| scenario_type | count |
|---|---|
{types}

## Scenarios
| ID | Title | Type | Positive targets | Series | DICOM files |
|---|---|---|---|---|---|
{listing}
"""


def render(candidates, with_model: bool, metadata=None, ensemble_dir=None):
    selected = select(candidates)
    scenarios = build(candidates, selected)
    if with_model:
        model_signals(metadata, scenarios, ensemble_dir)
    doc = {"schema_version": SCHEMA_VERSION, "scenario_set": SCENARIO_SET, "scenario_count": len(scenarios), "disclaimer": DISCLAIMER,
           "selection_method": "deterministic greedy diversity selection: target anchors (12), high multilabel (2), low label count (1), structural outliers (2), "
                              "farthest-point label+structure diversity (5); ties broken by study ID; see benchmarks/amd/README.md",
           "targets": list(TARGET_COLUMNS), "data_source": "RSNA Knee Abnormality Detection", "scenarios": scenarios}
    summary = summarize(scenarios, candidates)
    return (json.dumps(doc, indent=2, ensure_ascii=True) + "\n", json.dumps({"scenario_set": SCENARIO_SET, "schema_version": SCHEMA_VERSION, **summary}, indent=2, ensure_ascii=True) + "\n",
            readme(doc, summary).replace("\r\n", "\n"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/rsna-knee"))
    ap.add_argument("--dicom-root", type=Path, default=None, help="folder holding train_series/ (default: <data-dir>/raw)")
    ap.add_argument("--out-dir", type=Path, default=Path("benchmarks/amd"))
    ap.add_argument("--model-signals", action="store_true", help="add tags from out-of-fold fold-checkpoint inference (needs torch and the local checkpoints)")
    ap.add_argument("--ensemble-dir", type=Path, default=Path("outputs/rsna-knee-ensemble"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    dicom_root = args.dicom_root or args.data_dir / "raw"
    candidates = discover_candidates(args.data_dir, dicom_root)
    metadata = load_rsna_metadata(args.data_dir, dicom_root) if args.model_signals else None
    doc_text, summary_text, readme_text = render(candidates, args.model_signals, metadata, args.ensemble_dir)
    files = {"amd_demo_scenarios.json": doc_text, "amd_demo_scenarios_summary.json": summary_text, "README.md": readme_text}
    if args.check:
        bad = [n for n, text in files.items() if not (args.out_dir / n).is_file() or (args.out_dir / n).read_bytes() != text.encode("utf-8")]
        print("up to date" if not bad else f"differs: {bad}")
        return 1 if bad else 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out_dir / name).write_bytes(text.encode("utf-8"))  # bytes: identical output on every platform (LF line endings)
    print(f"wrote {len(files)} files to {args.out_dir} ({len(candidates)} candidate studies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
