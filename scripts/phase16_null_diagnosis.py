"""Phase 16 null diagnosis (local, CPU, no images): shuffle audit, AUC invariance, metadata-only baselines. Run from the repo root.
Uses only manifest labels and train_series.csv / report language; no gold studies."""
import warnings
warnings.filterwarnings("ignore")
import json,sys,numpy as np,pandas as pd
sys.path.insert(0,".")
from mri_core.weak_label_data import load_weak_manifest, hash_split
from mri_core.weak_label_diagnostics import shuffle_labels
from mri_core.rsna_knee_dataset import TARGET_COLUMNS
from mri_core.report_label_language import detect_language
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
T=["ACL","Medial Meniscus","Lateral Meniscus"]; J=[TARGET_COLUMNS.index(t) for t in T]
m=load_weak_manifest("outputs/weak-labels/weak_manifest_tier1_v0.1.0.jsonl")
sel=set(json.load(open("outputs/kaggle_smoke/out_p15/phase15/warm/selected_studies.json")))
order=[u for u in m.ids if u in sel]; idx={u:i for i,u in enumerate(m.ids)}
tr,va=hash_split(order,0.2,101); tr_ids=[order[i] for i in tr]; va_ids=[order[i] for i in va]
ho_ids=[u for u in m.ids if u not in sel]
V=lambda ids: m.values[[idx[u] for u in ids]]; M=lambda ids: m.masks[[idx[u] for u in ids]]
# ---- 1. shuffle audit (seed 1101 as run in Kaggle: train rows of the partition in manifest order)
tv,tm=V(tr_ids),M(tr_ids)
sv,sm,perm=shuffle_labels(tv,tm,1101)
print("== shuffle audit: train",len(tr_ids),"val",len(va_ids),"holdout",len(ho_ids))
print("rows moved:",int((perm!=np.arange(len(perm))).sum()),"/",len(perm),"| multiset of (values,masks) preserved:",
      np.array_equal(np.sort(np.nan_to_num(sv[:,J],nan=-1),axis=0),np.sort(np.nan_to_num(tv[:,J],nan=-1),axis=0)))
for t,j in zip(T,J):
    both=(tm[:,j]==1)&(sm[:,j]==1)
    agree=(np.nan_to_num(tv[both,j])==np.nan_to_num(sv[both,j])).mean()
    p=np.nanmean(tv[tm[:,j]==1,j]); exp=p*p+(1-p)**2
    r=np.corrcoef(np.nan_to_num(tv[both,j]),np.nan_to_num(sv[both,j]))[0,1]
    print(f"  {t}: positive rate {p:.3f}; label agreement original-vs-shuffled {agree:.3f} (chance {exp:.3f}); corr {r:+.3f}")
print("  validation/holdout labels come straight from manifest.values (never passed through shuffle_labels); overlaps train-val, train-holdout, val-holdout:",len(set(tr_ids)&set(va_ids)),len(set(tr_ids)&set(ho_ids)),len(set(va_ids)&set(ho_ids)))
# ---- 3. AUC properties with near-constant predictions
rng=np.random.default_rng(0); y=(rng.random(500)<0.3).astype(int); s=rng.normal(size=500)+0.3*y
base=roc_auc_score(y,s)
small=(0.5+1e-3*s).astype(np.float32); print("\n== AUC with near-constant float32 predictions: original",round(base,4),"| scaled to std~1e-3 around 0.5:",round(roc_auc_score(y,small),4),"| ties in scaled:",len(small)-len(np.unique(small)))
# ---- 4. metadata-only models
tr_s=pd.read_csv("data/rsna-knee/train_series.csv"); tr_c=pd.read_csv("data/rsna-knee/train.csv",usecols=["StudyInstanceUID","Report"])
g=tr_s.groupby("StudyInstanceUID")
F=pd.DataFrame({"n_series":g.size()})
for p in ("Axial","Coronal","Sagittal"): F["n_"+p]=g.apply(lambda d:(d.Anatomical_Plane==p).sum())
F["n_fluid"]=g.Fluid_Sensitive.sum(); F["n_fatsup"]=g.Fat_Suppression.sum()
for p in ("Axial","Coronal","Sagittal"): F["fluid_"+p]=g.apply(lambda d:((d.Anatomical_Plane==p)&(d.Fluid_Sensitive==1)).sum())
acq=list(F.columns)
rep=tr_c.set_index("StudyInstanceUID").Report.astype(str)
F["lang"]=[detect_language(rep.get(u,"")) for u in F.index]; F["replen"]=[np.log1p(len(rep.get(u,""))) for u in F.index]
L=pd.get_dummies(F["lang"]).astype(float)
def design(ids,cols,lang):
    X=F.loc[ids,cols].astype(float).values
    return np.hstack([X,L.loc[ids].values]) if lang else X
print("\n== metadata-only logistic regression, trained on the 942 seed-101 training studies (no images, no network)")
print("   languages in pool:",dict(F.loc[order+ho_ids,"lang"].value_counts()))
rows=[]
for name,cols,lang in (("acquisition-only",acq,False),("acq+language",acq,True),("language+report-length",["replen"],True)):
    out={}
    for split,ids in (("val",va_ids),("holdout",ho_ids)):
        aucs=[]
        for t,j in zip(T,J):
            ytr=np.nan_to_num(tv[:,j]); mtr=tm[:,j]==1
            clf=LogisticRegression(max_iter=2000,C=0.5).fit(design(tr_ids,cols,lang)[mtr],ytr[mtr])
            yv=np.nan_to_num(V(ids)[:,j]); mv=M(ids)[:,j]==1
            aucs.append(roc_auc_score(yv[mv],clf.predict_proba(design(ids,cols,lang)[mv])[:,1]))
        out[split]=(np.mean(aucs),aucs)
    print(f"   {name:24s} val macro {out['val'][0]:.3f} {np.round(out['val'][1],3)} | holdout macro {out['holdout'][0]:.3f} {np.round(out['holdout'][1],3)}")
# single-feature
print("\n== best single acquisition feature (|AUC-0.5|), hold-out / val, per target ACL,Med,Lat")
for c in acq:
    r=[]
    for ids in (ho_ids,va_ids):
        a=[]
        for t,j in zip(T,J):
            yv=np.nan_to_num(V(ids)[:,j]); mv=M(ids)[:,j]==1
            x=F.loc[ids,c].astype(float).values[mv]
            a.append(roc_auc_score(yv[mv],x) if np.ptp(x)>0 else np.nan)
        r.append(np.round(a,3))
    print(f"   {c:14s} holdout {r[0]} val {r[1]}")
