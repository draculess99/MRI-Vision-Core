"""Build the Kaggle code bundle (git-HEAD contents only) plus BUNDLE_FILES.json with per-file SHA-256s, into outputs/kaggle_smoke/dataset2."""
import subprocess, zipfile, hashlib, json
from pathlib import Path
head=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
tracked=subprocess.check_output(["git","ls-files","mri_core","configs","scripts/kaggle_rsna_runner.py","scripts/kaggle_rsna_knee.py","scripts/kaggle_cnn.py","scripts/kaggle_weak_label_experiment.py","scripts/kaggle_weak_label_submit.py","scripts/build_weak_label_manifest.py"],text=True).split()
files=sorted(f for f in tracked if f.endswith((".py",".json",".md")) and "__pycache__" not in f)
d=Path("outputs/kaggle_smoke/dataset2"); d.mkdir(parents=True,exist_ok=True)
out=d/"rsna_weak_label_code.zip"
with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
    for f in files:
        zi=zipfile.ZipInfo(f,(2026,10,15,0,0,0)); zi.compress_type=zipfile.ZIP_DEFLATED; zi.external_attr=0o644<<16
        z.writestr(zi,subprocess.check_output(["git","show",f"HEAD:{f}"]))
zf=zipfile.ZipFile(out)
side={"git_commit":head,"zip_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"files":{n:hashlib.sha256(zf.read(n)).hexdigest() for n in zf.namelist()},
      "manifest_reference":json.load(open("outputs/kaggle_smoke/manifest_reference.json")),"manifest_reference_v022":json.load(open("outputs/kaggle_smoke/manifest_reference_v022.json"))}
(d/"BUNDLE_FILES.json").write_text(json.dumps(side,indent=1))
(d/"dataset-metadata.json").write_text(json.dumps({"title":"rsna-weak-label-code","id":"draculess/rsna-weak-label-code","licenses":[{"name":"other"}],"isPrivate":True},indent=2))
print(head,len(files),out.stat().st_size,side["zip_sha256"])
