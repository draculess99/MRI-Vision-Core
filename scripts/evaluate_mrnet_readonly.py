"""Read-only external evaluation of preserved checkpoints on MRNet (no training, no checkpoint or dataset modification).

  python scripts/evaluate_mrnet_readonly.py --mrnet-root D:/MRI_DATASETS/MRNet --checkpoint B0=<last.pt> --checkpoint C1=<last.pt> --out outputs/mrnet_eval/run1
  python scripts/evaluate_mrnet_readonly.py ... --preflight 24     # shapes, finiteness, throughput only: NO labels are read for metrics, NO AUC is computed

Pre-registered protocol: docs/phase18_mrnet_evaluation_protocol.md. Results are transfer performance on MRNet only.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mri_core import mrnet_eval as me  # noqa: E402


def tree_signature(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            h.update(f"{p.relative_to(root).as_posix()}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mrnet-root", type=Path, required=True)
    ap.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--untrained-seeds", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--preflight", type=int, default=0, help="score only the first N exams and report shapes/throughput, never metrics")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)
    ckpts = {}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        ckpts[name] = Path(path)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    rows, provenance = me.load_reference(args.mrnet_root)
    if args.preflight:
        rows = rows[:args.preflight]
    before = {"checkpoints": {n: me.sha256_file(p) for n, p in ckpts.items()}, "dataset_signature": tree_signature(args.mrnet_root)}
    summary = {"experiment": "mrnet_readonly_evaluation", "preflight": bool(args.preflight), "device": str(device), "exams": len(rows),
               "reference_provenance": provenance, "hashes_before": before, "claim_scope": "transfer performance on MRNet only; not RSNA accuracy, not clinical accuracy"}
    scores, throughput, meta = {}, {}, None
    config0 = None
    for name, path in ckpts.items():
        model, config = me.load_checkpoint_model(path)
        config0 = config0 or config
        ds = me.MRNetDataset(rows, int(config["image_size"]), int(config["slices_per_series"]))
        out = me.predict(model, ds, device, args.batch_size)
        assert np.isfinite(out["probs"]).all(), name
        scores[name], meta = me.endpoint_scores(out["probs"]), out["meta"]
        throughput[name] = {k: round(out[k], 3) for k in ("forward_seconds", "total_seconds", "exams_per_second_forward", "images_per_second_forward")}
        summary.setdefault("tensor", {"images_per_exam": [3, ds.slices, 1, ds.size, ds.size]})
    summary["throughput"] = throughput
    if args.preflight:  # no labels are used for anything here
        summary["probability_ranges"] = {n: [float(s.min()), float(s.max())] for n, s in scores.items()}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "preflight.json").write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps({k: summary[k] for k in ("exams", "tensor", "throughput", "probability_ranges")}, indent=1))
        return 0
    seeds = me.UNTRAINED_SEEDS[:args.untrained_seeds]
    untrained = {}
    ds = me.MRNetDataset(rows, int(config0["image_size"]), int(config0["slices_per_series"]))
    t = time.perf_counter()
    for seed in seeds:
        untrained[seed] = me.endpoint_scores(me.predict(me.untrained_model(config0, seed), ds, device, args.batch_size)["probs"])
    summary["untrained_seconds"] = round(time.perf_counter() - t, 1)
    labels = {e: np.array([int(r[e]) for r in rows]) for e in me.ENDPOINTS}
    split = np.array([r["split"] for r in rows])
    names = list(ckpts)
    differences = [(a, b) for a in names for b in ("META_boosted", "META_logreg", "UNTRAINED")] + [(names[i], names[j]) for i in range(len(names)) for j in range(i)]
    meta_all = {e: me.metadata_oof_scores(meta, labels[e]) for e in me.ENDPOINTS}
    summary["primary_all_exams"] = me.evaluate(labels, scores, untrained, meta_all, differences, rounds=args.rounds)
    valid = split == "valid"
    meta_valid = {}
    for e in me.ENDPOINTS:  # metadata fitted on the MRNet train split, scored on the valid split
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(meta[~valid])
        meta_valid[e] = np.stack([LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(meta[~valid]), labels[e][~valid]).predict_proba(sc.transform(meta[valid]))[:, 1],
                                  HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, random_state=0).fit(meta[~valid], labels[e][~valid]).predict_proba(meta[valid])[:, 1]], axis=1)
    summary["secondary_valid_split"] = me.evaluate({e: labels[e][valid] for e in me.ENDPOINTS}, {n: s[valid] for n, s in scores.items()},
                                                   {k: u[valid] for k, u in untrained.items()}, meta_valid, differences, rounds=args.rounds)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "predictions.csv", "w", encoding="utf-8") as handle:
        handle.write("exam_id,split,acl,meniscus," + ",".join(f"{n}_acl,{n}_meniscus" for n in names) + "\n")
        for i, r in enumerate(rows):
            handle.write(f"{r['exam_id']},{r['split']},{r['acl']},{r['meniscus']}," + ",".join(f"{scores[n][i, 0]!r},{scores[n][i, 1]!r}" for n in names) + "\n")
    after = {"checkpoints": {n: me.sha256_file(p) for n, p in ckpts.items()}, "dataset_signature": tree_signature(args.mrnet_root)}
    summary["hashes_after"] = after
    summary["read_only_verified"] = after == before
    summary["predictions_sha256"] = me.sha256_file(args.out / "predictions.csv")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    summary["summary_sha256"] = me.sha256_file(args.out / "summary.json")
    print(json.dumps({"read_only_verified": summary["read_only_verified"], "exams": len(rows), "summary_sha256": summary["summary_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
