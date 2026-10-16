"""Phase 16X post-analysis (local, CPU, exploratory): acquisition-feature correlations of the trained model's scores and per-target paired
bootstrap differences, from the kernel's per-study prediction files and the manifest labels. Run from the repo root."""
import json
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
from mri_core.weak_label_data import load_weak_manifest
from mri_core.rsna_knee_dataset import TARGET_COLUMNS

B = "outputs/kaggle_smoke/out_p16x_v2/phase16x/"
m = load_weak_manifest("outputs/weak-labels/weak_manifest_tier1_v0.1.0.jsonl")
idx = {u: i for i, u in enumerate(m.ids)}
feat = pd.read_csv(B + "header_features.csv", index_col=0)
T = ["ACL", "Medial Meniscus", "Lateral Meniscus"]
short = {"ACL": "ACL", "Medial Meniscus": "Medial", "Lateral Meniscus": "Lateral"}
for part in ("val", "holdout"):
    p = pd.read_csv(B + f"predictions_{part}.csv")
    ids = list(p["study"])
    rows = [idx[u] for u in ids]
    print(f"\n===== {part}: {len(ids)} studies")
    print("Spearman of trained_real with acquisition features (|rho| >= 0.2 shown), per target:")
    cols = [c for c in feat.columns if c != "manufacturer"]
    for t in T:
        s = p[f"trained_real_{short[t].split()[0]}"].values if f"trained_real_{short[t].split()[0]}" in p else None
    tcols = {t: [c for c in p.columns if c.startswith("trained_real")][i] for i, t in enumerate(T)}
    for t in T:
        s = p[tcols[t]].values
        out = []
        for c in cols:
            x = feat.loc[ids, c].astype(float).values
            ok = ~np.isnan(x)
            if ok.sum() > 10 and np.ptp(x[ok]) > 0:
                r = spearmanr(s[ok], x[ok]).correlation
                if abs(r) >= 0.2:
                    out.append((c, round(float(r), 2)))
        print(f"  {t}:", sorted(out, key=lambda z: -abs(z[1]))[:6])
    # per-target paired bootstrap: trained minus each metadata model
    rng = np.random.default_rng(0)
    lab = m.values[rows]; msk = m.masks[rows]
    for k, t in enumerate(T):
        j = TARGET_COLUMNS.index(t)
        on = msk[:, j] == 1
        y = np.nan_to_num(lab[on, j]).astype(int)
        sc = {n: p[[c for c in p.columns if c.startswith(n)][k]].values[on] for n in ("trained_real", "metadata_logreg", "metadata_boosted")}
        pt = {n: roc_auc_score(y, v) for n, v in sc.items()}
        diffs = {n: [] for n in ("metadata_logreg", "metadata_boosted")}
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            if 0 < y[i].sum() < len(i):
                a = roc_auc_score(y[i], sc["trained_real"][i])
                for n in diffs:
                    diffs[n].append(a - roc_auc_score(y[i], sc[n][i]))
        ci = {n: np.percentile(v, [2.5, 97.5]).round(3).tolist() for n, v in diffs.items()}
        print(f"  {t:16s} AUC trained {pt['trained_real']:.3f} logreg {pt['metadata_logreg']:.3f} boosted {pt['metadata_boosted']:.3f} | trained-logreg {pt['trained_real']-pt['metadata_logreg']:+.3f} {ci['metadata_logreg']} | trained-boosted {pt['trained_real']-pt['metadata_boosted']:+.3f} {ci['metadata_boosted']}")
sp = json.load(open(B + "final_summary.json"))["analysis"]
for part in ("val", "holdout"):
    print(part, "Spearman(trained_real, metadata_logreg):", [None if x is None else round(x, 3) for x in sp[part]["spearman_with_metadata_logreg"]["trained_real"]])
