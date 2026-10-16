"""Phase 17 label-quality audit (CPU, local, text-free). Audits the registered 1,200-study training selection for all 12 targets.

Never reads or uses: the Phase 16 hold-out pool labels, the 58 gold studies, test labels, model predictions or any model performance.
Report text is only used locally for its length (never printed or stored). Original manifests are read-only; outputs go to a new folder.

  python scripts/audit_weak_label_quality.py --selection <selected_studies.json> [--header-features <csv>] [--out outputs/weak-labels/label_quality]
"""

import argparse
import hashlib
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mri_core import weak_label_quality as wq  # noqa: E402
from mri_core.rsna_knee_dataset import TARGET_COLUMNS  # noqa: E402

TIER1 = ("ACL", "Medial Meniscus", "Lateral Meniscus")
STATES = ("positive", "negative", "uncertain", "not_mentioned")


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def cv_metadata_auc(features, ids, y, mask, folds=5, seed=0, rounds=200):
    """Out-of-fold AUC of a metadata-only logistic regression predicting the (decided) label, with a bootstrap CI. Labels of the audit set only."""
    from sklearn.model_selection import StratifiedKFold
    from mri_core.weak_label_confound import fit_predict_baselines
    use = np.flatnonzero(mask == 1)
    if len(use) < 40 or len(set(y[use])) < 2 or y[use].sum() < folds:
        return None
    ids_u = [ids[i] for i in use]
    values = np.full((len(ids_u), 12), np.nan)
    masks = np.zeros((len(ids_u), 12))
    for j in (TARGET_COLUMNS.index(t) for t in TIER1):  # reuse the 3-slot baseline helper: fill all three slots with this target
        values[:, j], masks[:, j] = y[use], 1
    oof = np.full(len(use), np.nan)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(use)), y[use]):
        pred = fit_predict_baselines(features, [ids_u[i] for i in tr], values[tr], masks[tr], {"te": [ids_u[i] for i in te]})
        oof[te] = pred["logreg"]["te"][:, 0]
    auc = wq.auc_of_score(y[use], oof)
    rng = np.random.default_rng(seed)
    draws = [wq.auc_of_score(y[use][i], oof[i]) for i in (rng.integers(0, len(use), len(use)) for _ in range(rounds))]
    draws = [d for d in draws if d is not None]
    return {"auc": auc, "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if len(draws) > 20 else None, "n": int(len(use))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/rsna-knee"))
    ap.add_argument("--manifest-dir", type=Path, default=Path("outputs/weak-labels"))
    ap.add_argument("--header-features", type=Path, default=None)
    ap.add_argument("--public-labels", type=Path, default=None, help="third-party label CSV; descriptive agreement only, never used to define the candidate subset")
    ap.add_argument("--out", type=Path, default=Path("outputs/weak-labels/label_quality"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    profiles = {p: wq.load_manifest_rows(args.manifest_dir / f"weak_manifest_tier1_v{p}.jsonl") for p in wq.PROFILES}
    base = profiles["0.1.0"]
    selection = json.loads(args.selection.read_text())
    ids = [u for u in base if u in set(selection)]  # manifest order
    train = pd.read_csv(args.data_dir / "train.csv", usecols=["StudyInstanceUID", "Report"] + list(TARGET_COLUMNS))
    gold_ids = set(train.loc[train[list(TARGET_COLUMNS)].notna().all(axis=1), "StudyInstanceUID"].astype(str))
    test_ids = set(pd.read_csv(args.data_dir / "test.csv")["StudyInstanceUID"].astype(str))
    holdout_pool = set(base) - set(selection)
    wq.assert_audit_scope(ids, selection, {"gold": gold_ids, "test": test_ids, "phase16 hold-out pool": holdout_pool})
    series = pd.read_csv(args.data_dir / "train_series.csv")
    n_series = series.groupby("StudyInstanceUID").size()
    report_len = train.set_index("StudyInstanceUID")["Report"].astype(str).str.len()
    result = wq.candidate_labels(profiles, ids, ignore_tier_mask=True)
    tier1 = wq.candidate_labels(profiles, ids, tier_targets=TIER1)
    lang = np.array([base[u]["language"] for u in ids])
    ns = np.array([n_series.get(u, np.nan) for u in ids], float)
    rl = np.log1p(np.array([report_len.get(u, np.nan) for u in ids], float))
    features = None
    if args.header_features and args.header_features.is_file():
        features = pd.read_csv(args.header_features, index_col=0)
        features = features.loc[[u for u in ids if u in features.index]]
    audit = {"scope": {"studies": len(ids), "selection_sha256": sha_ids(selection), "manifest_sha256": sha_file(args.manifest_dir / "weak_manifest_tier1_v0.1.0.jsonl"),
                       "profiles": {p: sha_file(args.manifest_dir / f"weak_manifest_tier1_v{p}.jsonl") for p in wq.PROFILES},
                       "languages": dict(Counter(lang)), "excluded_from_audit": {"gold": len(gold_ids), "holdout_pool": len(holdout_pool), "test": len(test_ids)},
                       "never_negative_from_missing": wq.never_negative_from_missing(base, result, ids)},
             "targets": {}}
    shared_any = np.array([[wq.shared_evidence_counts(base[u])[t] for t in TARGET_COLUMNS] for u in ids])
    for j, target in enumerate(TARGET_COLUMNS):
        entries = [base[u]["targets"][target] for u in ids]
        states = Counter(e["state"] for e in entries)
        decided = result["mask"][:, j] == 1
        y = result["values"][:, j]
        conf = Counter(result["confidence"][decided, j])
        stab = Counter(result["stability"][decided, j])
        reasons = {s: dict(Counter(e["reason"] for e in entries if e["state"] == s)) for s in STATES if states.get(s)}
        sections = dict(Counter(e["section"] for e, d in zip(entries, decided) if d))
        unsupported = int(sum(1 for e in entries if e["state"] in wq.DECIDED and not e.get("evidence_sha256")))
        row = {"tier1_trainable": target in TIER1, "states": {s: int(states.get(s, 0)) for s in STATES}, "decided": int(decided.sum()),
               "decided_fraction": float(decided.mean()), "positive_rate_among_decided": float(y[decided].mean()) if decided.any() else None,
               "undecided_breakdown": {"uncertain": int(states.get("uncertain", 0)), "not_mentioned": int(states.get("not_mentioned", 0))},
               "decided_reasons": reasons, "sections_of_decided": sections, "confidence_tiers": {k: int(v) for k, v in conf.items()},
               "stability_across_profiles": {k: int(v) for k, v in stab.items()}, "unsupported_decided_without_evidence": unsupported,
               "decided_with_evidence_shared_by_>=4_targets": int((shared_any[decided, j] >= wq.SHARED_EVIDENCE_LIMIT).sum())}
        by_lang = {}
        for lg in ("en", "es"):
            m = lang == lg
            d = decided & m
            by_lang[lg] = {"studies": int(m.sum()), "decided_rate": float(decided[m].mean()) if m.any() else None,
                           "positive_rate": float(y[d].mean()) if d.any() else None, "positives": int(y[d].sum()), "decided": int(d.sum())}
        by_lang["p_equal_positive_rate"] = wq.two_by_two_p(int(by_lang["en"]["positives"]), by_lang["en"]["decided"], int(by_lang["es"]["positives"]), by_lang["es"]["decided"])
        by_lang["p_equal_decided_rate"] = wq.two_by_two_p(int(decided[lang == "en"].sum()), int((lang == "en").sum()), int(decided[lang == "es"].sum()), int((lang == "es").sum()))
        row["by_language"] = by_lang
        # acquisition / reporting associations inside the audit set (labels of the audit set only)
        ok = decided & ~np.isnan(ns)
        row["association"] = {"auc_series_count_for_label": wq.auc_of_score(y[ok].astype(int), ns[ok]) if ok.any() else None,
                              "auc_series_count_for_decided_vs_not": wq.auc_of_score(decided.astype(int), np.nan_to_num(ns)),
                              "auc_log_report_length_for_decided_vs_not": wq.auc_of_score(decided.astype(int), np.nan_to_num(rl)),
                              "auc_log_report_length_for_label": wq.auc_of_score(y[decided].astype(int), rl[decided]) if decided.any() else None}
        cand = result["candidate"][:, j]
        row["candidate_all12_rules"] = {"labels": int(cand.sum()), "positives": int(y[cand].sum()), "negatives": int((~(y[cand] == 1)).sum()), "retained_of_decided": float(cand.sum() / max(decided.sum(), 1))}
        if features is not None and target in TIER1:
            fid = list(features.index)
            keep = np.array([u in set(fid) for u in ids])
            row["cv_metadata_auc_all_decided"] = cv_metadata_auc(features, ids, np.nan_to_num(y).astype(int), (decided & keep).astype(int))
            row["cv_metadata_auc_candidate_only"] = cv_metadata_auc(features, ids, np.nan_to_num(y).astype(int), (cand & keep).astype(int))
        audit["targets"][target] = row
    # label-free selection comparison: the 1,200 versus every eligible (non-gold) training study with a report
    eligible = train.loc[~train["StudyInstanceUID"].astype(str).isin(gold_ids), "StudyInstanceUID"].astype(str)
    elig_n = np.array([n_series.get(u, np.nan) for u in eligible], float)
    audit["selection_bias_label_free"] = {"eligible_studies": int(len(eligible)), "audit_mean_series": float(np.nanmean(ns)), "eligible_mean_series": float(np.nanmean(elig_n)),
                                          "audit_median_series": float(np.nanmedian(ns)), "eligible_median_series": float(np.nanmedian(elig_n)),
                                          "audit_language_mix": {k: v / len(ids) for k, v in Counter(lang).items()},
                                          "note": "The audit set is the English/Spanish subset with at least one decided label, stratified with a per-class quota (60 per class per active target); eligible includes unsupported languages."}
    # extractor-profile sensitivity by language: do later profiles move the Spanish rates?
    sens = {}
    for t in TIER1:
        sens[t] = {}
        for p, rows in profiles.items():
            for lg in ("en", "es"):
                st = [rows[u]["targets"][t]["state"] for u, l in zip(ids, lang) if l == lg and u in rows]
                dec = [x for x in st if x in wq.DECIDED]
                sens[t][f"{p}_{lg}"] = {"decided": len(dec), "positive_rate": (sum(x == "positive" for x in dec) / len(dec)) if dec else None}
    audit["profile_sensitivity_by_language"] = sens
    if args.public_labels and args.public_labels.is_file():
        pub = pd.read_csv(args.public_labels).set_index("StudyInstanceUID")
        agree = {}
        for t in TIER1:
            j = TARGET_COLUMNS.index(t)
            agree[t] = {}
            for lg in ("en", "es"):
                n = a_ = both_pos = ours_pos_theirs_neg = ours_neg_theirs_pos = 0
                for i, u in enumerate(ids):
                    if lang[i] != lg or u not in pub.index or result["mask"][i, j] != 1:
                        continue
                    v = pub.loc[u, t]
                    theirs = 1 if v >= 0.7 else (0 if v <= 0.1 else None)
                    if theirs is None:
                        continue
                    n += 1
                    a_ += int(theirs == result["values"][i, j])
                    ours_pos_theirs_neg += int(result["values"][i, j] == 1 and theirs == 0)
                    ours_neg_theirs_pos += int(result["values"][i, j] == 0 and theirs == 1)
                agree[t][lg] = {"both_decided": n, "agree": a_, "ours_positive_theirs_negative": ours_pos_theirs_neg, "ours_negative_theirs_positive": ours_neg_theirs_pos}
        audit["independent_annotator_agreement_descriptive_only"] = {"file_sha256": sha_file(args.public_labels), "rule": "theirs positive if >= 0.7, negative if <= 0.1, otherwise not compared",
                                                                       "note": "third-party LLM labels (built from the same reports and tuned on the gold studies by their author); not used to define or filter any label", "per_target": agree}
    # candidate subset (Tier-1 trainable) and its effect on acquisition association
    cand_summary = {}
    for t in TIER1:
        j = TARGET_COLUMNS.index(t)
        mask0 = tier1["mask"][:, j] == 1
        cand = tier1["candidate"][:, j]
        cand_summary[t] = {"original_decided": int(mask0.sum()), "original_positives": int(tier1["values"][mask0, j].sum()),
                           "candidate": int(cand.sum()), "candidate_positives": int(tier1["values"][cand, j].sum()),
                           "candidate_negatives": int(cand.sum() - tier1["values"][cand, j].sum()),
                           "retained_fraction": float(cand.sum() / max(mask0.sum(), 1))}
    audit["candidate_subset"] = {"rules": ["decided (positive/negative) and Tier-1 trainable", "confidence 'high': positive with no modifier, or negative from negated_lesion / normal_structure / normal_cartilage",
                                          "evidence recorded and not shared by >= 4 targets of the study", "same decided state in manifest profiles 0.1.0, 0.2.0 and 0.2.2 (studies dropped by a later profile count as unstable)"],
                                 "per_target": cand_summary, "studies_with_any_candidate": int(tier1["candidate"].any(axis=1).sum()),
                                 "fingerprint_sha256": wq.candidate_fingerprint(ids, tier1), "never_negative_from_missing": wq.never_negative_from_missing(base, tier1, ids)}
    audit["power"] = {f"auc{a}_halfwidth{h}_prevalence{r}": list(wq.positives_needed(a, r, h)) for a in (0.65, 0.75) for h in (0.05, 0.03) for r in (0.13, 0.38)}  # (positives, total) for a single-AUC 95% CI
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2, default=float))
    np.savez_compressed(args.out / "candidate_labels_tier1.npz", ids=np.array(ids), values=tier1["values"], mask=tier1["mask"], candidate=tier1["candidate"],
                        confidence=tier1["confidence"].astype(str), stability=tier1["stability"].astype(str), targets=np.array(TARGET_COLUMNS))
    print(json.dumps({"studies": len(ids), "candidate": audit["candidate_subset"], "scope": audit["scope"]}, indent=1, default=float)[:2500])


if __name__ == "__main__":
    main()
