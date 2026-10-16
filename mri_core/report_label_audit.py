"""Reproducible, blinded audit sample of report excerpts for human adjudication of weak labels.

The sample is stratified by how a label changed between two extractor profiles (changed labels are oversampled) and balanced
across languages inside each stratum. Two files are produced: a BLIND sheet (excerpt only, empty adjudication columns) and a KEY
(automated states, strata, optional third-party reference). Nothing here treats an automated label as ground truth, and no
precision is computed: adjudication is a separate, human step. Report text stays in git-ignored output directories.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .report_label_rules import RuleBasedExtractor, SPECS, fold, split_sections, split_sentences
from .report_labels import LabelState

TIER1 = ("ACL", "Medial Meniscus", "Lateral Meniscus")
EXCERPT_MAX_CHARS = 420
MAX_SENTENCES = 3

# (stratum name, from-state, to-state or None for "unchanged <state>", quota)
STRATA = (
    ("not_mentioned->positive", "not_mentioned", "positive", 14),
    ("not_mentioned->negative", "not_mentioned", "negative", 12),
    ("uncertain->positive", "uncertain", "positive", 14),
    ("uncertain->negative", "uncertain", "negative", 4),
    ("not_mentioned->uncertain", "not_mentioned", "uncertain", 6),
    ("positive->uncertain", "positive", "uncertain", 10),
    ("negative->uncertain", "negative", "uncertain", 6),
    ("negative->positive", "negative", "positive", 5),
    ("unchanged positive", "positive", None, 8),
    ("unchanged negative", "negative", None, 8),
    ("unchanged uncertain", "uncertain", None, 6),
    ("anatomy_mentioned_but_undecided", "not_mentioned", None, 7),
)


def _digest(*parts) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()


def excerpt_for(report: str, target: str) -> str:
    """Up to three sentences of the report that name the target structure (impression first), de-identified text only."""
    spec = SPECS[target]
    found = []
    for section, block in split_sections(report, inline_headings=True):
        for sentence in split_sentences(block):
            folded = fold(sentence)
            if spec.anchor.search(folded) or (spec.generic is not None and spec.generic.search(folded)):
                found.append((section != "impression", sentence.strip().replace("\n", " ")))
    seen, out = set(), []
    for _, sentence in sorted(found, key=lambda item: item[0]):
        if sentence not in seen:
            seen.add(sentence)
            out.append(sentence)
    return " | ".join(out[:MAX_SENTENCES])[:EXCERPT_MAX_CHARS]


def build_audit_sample(train: pd.DataFrame, gold_ids: set, *, old_profile: str = "0.1.0", new_profile: str = "0.2.1", seed: int = 0,
                       strata=STRATA, targets: Sequence[str] = TIER1) -> list[dict]:
    """Deterministic stratified sample of (study, target) pairs from training-only reports (gold studies are never eligible)."""
    old, new = RuleBasedExtractor(old_profile), RuleBasedExtractor(new_profile)
    pools: dict[str, list[dict]] = {name: [] for name, *_ in strata}
    for row in train.itertuples():
        uid = str(row.StudyInstanceUID)
        if uid in gold_ids:
            continue
        a, b = old.extract(row.Report, uid), new.extract(row.Report, uid)
        if not b.supported:
            continue
        for target in targets:
            sa, sb = a.state(target).value, b.state(target).value
            for name, from_state, to_state, _ in strata:
                if name == "anatomy_mentioned_but_undecided":
                    hit = sa == sb == "not_mentioned" and bool(excerpt_for(row.Report, target))
                elif to_state is None:
                    hit = sa == sb == from_state
                else:
                    hit = (sa, sb) == (from_state, to_state)
                if hit:
                    pools[name].append({"study_uid": uid, "language": b.language, "target": target, "stratum": name,
                                        "state_old": sa, "state_new": sb,
                                        "reason_new": b.labels[[l.target for l in b.labels].index(target)].reason,
                                        "report": row.Report})
    sample = []
    for name, _, _, quota in strata:
        pool = sorted(pools[name], key=lambda r: _digest(seed, r["study_uid"], r["target"], name))
        by_language = {"en": [r for r in pool if r["language"] == "en"], "es": [r for r in pool if r["language"] == "es"]}
        chosen = by_language["es"][: quota // 2] + by_language["en"][: quota - min(len(by_language["es"]), quota // 2)]
        if len(chosen) < quota:  # fill from whichever language still has candidates
            rest = [r for r in pool if r not in chosen]
            chosen += rest[: quota - len(chosen)]
        sample.extend(chosen[:quota])
    for i, record in enumerate(sorted(sample, key=lambda r: _digest(seed, "order", r["study_uid"], r["target"]))):
        record["audit_id"] = f"A{i + 1:03d}"
    return sorted(sample, key=lambda r: r["audit_id"])


def write_audit_files(sample: list[dict], out_dir: Path, *, reference: Optional[dict] = None) -> tuple[Path, Path]:
    """BLIND sheet (what the adjudicator sees) and KEY (automated states + strata). `reference` maps (uid, target) to a
    third-party label state and goes only into the key."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet, key = out_dir / "label_audit_sheet_blind.csv", out_dir / "label_audit_key.csv"
    with open(sheet, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audit_id", "language", "target", "excerpt", "adjudicated_state (positive|negative|uncertain|not_mentioned)",
                         "adjudicator_confidence (1-3)", "notes"])
        for r in sample:
            writer.writerow([r["audit_id"], r["language"], r["target"], excerpt_for(r["report"], r["target"]), "", "", ""])
    with open(key, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audit_id", "study_uid", "language", "target", "stratum", "state_old", "state_new", "reason_new",
                         "reference_third_party_state"])
        for r in sample:
            ref = (reference or {}).get((r["study_uid"], r["target"]), "")
            writer.writerow([r["audit_id"], r["study_uid"], r["language"], r["target"], r["stratum"], r["state_old"], r["state_new"],
                             r["reason_new"] or "", ref])
    return sheet, key
