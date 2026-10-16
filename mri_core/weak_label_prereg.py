"""Pre-registered decision rules for the Phase 16 paired real-vs-shuffled comparison (see configs/phase16_registry.json).

Pure functions of per-run results, so the verdict is computed by code that was fixed before any run existed.
Nothing here is a clinical claim: AUCs measure agreement with automated weak labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "phase16_registry.json"
C1_MARGIN, C2_MARGIN = 0.10, 0.05
SANITY_BAND = (0.45, 0.55)
CLIP_MARGIN = 0.03
CONTROL_ALARM = 0.60


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_name(seed: int, clip: float, label: str) -> str:
    return f"s{seed}_clip{clip:g}_{label}"


def evaluate(registry: Mapping, results: Mapping[str, Mapping]) -> dict:
    """results[run_name] = {val_final_macro_auc, val_final_loss, val_floor_loss, holdout_macro_auc}. Returns the registered verdict."""
    seeds, clips = list(registry["seeds"]), list(registry["grad_clip_arms"])
    out = {"arms": {}, "clip_comparison": None, "control_alarm": False}
    for clip in clips:
        pairs = []
        for seed in seeds:
            real, shuf = results.get(run_name(seed, clip, "real")), results.get(run_name(seed, clip, "shuf"))
            if real and shuf:
                pairs.append((seed, real, shuf))
        n = len(pairs)
        per_seed = [{"seed": s,
                     "C1_val_diff": r["val_final_macro_auc"] - h["val_final_macro_auc"],
                     "C2_holdout_diff": r["holdout_macro_auc"] - h["holdout_macro_auc"],
                     "C3_below_floor": r["val_final_loss"] < r["val_floor_loss"],
                     "shuffled_val_auc": h["val_final_macro_auc"], "shuffled_holdout_auc": h["holdout_macro_auc"]} for s, r, h in pairs]
        c1 = sum(p["C1_val_diff"] >= C1_MARGIN for p in per_seed)
        c2 = sum(p["C2_holdout_diff"] >= C2_MARGIN for p in per_seed)
        c3 = sum(bool(p["C3_below_floor"]) for p in per_seed)
        sv = float(np.mean([p["shuffled_val_auc"] for p in per_seed])) if per_seed else None
        sh = float(np.mean([p["shuffled_holdout_auc"] for p in per_seed])) if per_seed else None
        sane = bool(per_seed) and SANITY_BAND[0] <= sv <= SANITY_BAND[1] and SANITY_BAND[0] <= sh <= SANITY_BAND[1]
        alarm = any(p["shuffled_val_auc"] >= CONTROL_ALARM and p["shuffled_holdout_auc"] >= CONTROL_ALARM for p in per_seed)
        out["control_alarm"] = out["control_alarm"] or alarm
        if n < len(seeds):
            verdict = "incomplete"
        elif not sane:
            verdict = "invalid_control"
        elif min(c1, c2, c3) == n:
            verdict = "supported"
        elif min(c1, c2, c3) >= n - 1:
            verdict = "promising"
        else:
            verdict = "not_supported"
        real_auc = [results[run_name(s, clip, "real")] for s, _, _ in pairs]
        out["arms"][f"clip{clip:g}"] = {
            "complete_seeds": n, "C1_count": c1, "C2_count": c2, "C3_count": c3, "control_sane": sane,
            "mean_shuffled_val_auc": sv, "mean_shuffled_holdout_auc": sh,
            "mean_real_val_auc": float(np.mean([r["val_final_macro_auc"] for r in real_auc])) if pairs else None,
            "mean_real_holdout_auc": float(np.mean([r["holdout_macro_auc"] for r in real_auc])) if pairs else None,
            "per_seed": per_seed, "verdict": verdict}
    if len(clips) == 2:
        names = [(run_name(s, clips[0], "real"), run_name(s, clips[1], "real")) for s in seeds]
        if all(a in results and b in results for a, b in names):
            diffs = [results[b]["val_final_macro_auc"] - results[a]["val_final_macro_auc"] for a, b in names]
            same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
            out["clip_comparison"] = {"paired_diffs_clip%g_minus_clip%g" % (clips[1], clips[0]): diffs, "mean": float(np.mean(diffs)),
                                      "different": bool(same_sign and abs(np.mean(diffs)) >= CLIP_MARGIN)}
        else:
            out["clip_comparison"] = {"incomplete": True}
    return out
