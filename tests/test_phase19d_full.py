"""Phase 19D: scaler growth after clean updates (never reached in the 24-update rehearsal), full-driver resume/pause wiring and the local partition preflight."""
import hashlib
import json
import pathlib

import pytest

from mri_core import weak_label_train
from mri_core.weak_label_train import train_weak_label

from .test_convnext_roi import run_cfg, toy_data
from .test_phase19c_amp import poisoned_factory, rows

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "scripts" / "kaggle_phase19_driver.py").read_text(encoding="utf-8")


def test_scaler_grows_after_clean_updates_halves_on_overflow_and_survives_a_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(weak_label_train, "AMP_DEVICE_TYPES", ("cuda", "cpu"))
    ds = toy_data(8)
    cfg = {**run_cfg("D"), "epochs": 2, "batch_size": 2, "amp": True, "amp_init_scale": 8.0, "amp_growth_interval": 3}  # 4 updates per epoch
    full, part = tmp_path / "full", tmp_path / "part"
    train_weak_label(ds, ds, cfg, full, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids, model_factory=poisoned_factory(cfg, {3}))
    train_weak_label(ds, ds, cfg, part, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids, model_factory=poisoned_factory(cfg, {3}), stop_after_epochs=1)
    train_weak_label(ds, ds, cfg, part, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids, model_factory=poisoned_factory(cfg, set()))
    expected = [8.0, 8.0, 16.0, 8.0, 8.0, 8.0, 16.0, 16.0]  # +x2 after 3 clean updates, x0.5 at the poisoned 4th, tracker reset, grows again
    for d in (full, part):
        r = rows(d)
        assert [x["scale_after"] for x in r] == expected and [x["skipped"] for x in r] == [False, False, False, True, False, False, False, False]
    assert [x["scale_after"] for x in rows(full)] == [x["scale_after"] for x in rows(part)]


def test_full_driver_resumes_from_one_attached_checkpoint_folder_pauses_on_the_time_budget_and_does_not_score_a_paused_run():
    assert "_found = [f for f in _found if f.name == f\"s{SEED}_D\"]" in DRIVER and "attach exactly one" in DRIVER
    assert 'shutil.copytree(_found[0], work, ignore=shutil.ignore_patterns("last_best.pt"))' in DRIVER
    assert '"--max-seconds", MAX_SECONDS' in DRIVER and 'MAX_SECONDS = int(os.environ.get("P19_MAX_SECONDS", str(6 * 3600)))' in DRIVER
    paused = DRIVER.index('if state["epoch"] < EPOCHS')
    assert paused < DRIVER.index('"score_D"') and paused < DRIVER.index("fit_predict_baselines")  # no scoring or baselines before all 12 epochs exist
    assert 'EPOCHS = int(os.environ.get("P19_EPOCHS", "12" if os.environ.get("P19_MODE", "full") == "full" else "3"))' in DRIVER
    for needle in ('"grad_accum_steps": ACCUM', '"amp_init_scale": 1024.0', '"learning_rate": 1e-4', '"weight_decay": 0.05', '"grad_clip": 1.0', '"early_stopping_patience": 0',
                   'BATCH = int(os.environ.get("P19_BATCH", "2"))', 'os.environ.get("P19_ACCUM", "2")', 'VAL_FRACTION = 0.2 if MODE == "full" else 0.4'):
        assert needle in DRIVER, needle


def test_full_kernel_folder_is_private_t4_internet_for_the_official_weights_only_and_matches_the_driver():
    d = ROOT / "outputs" / "kaggle_smoke" / "p19d_arm_d_full_DO_NOT_PUSH_without_approval"
    if not d.is_dir():
        pytest.skip("local kernel folder not built")
    m = json.loads((d / "kernel-metadata.json").read_text())
    assert m["id"] == "draculess/rsna-phase19d-arm-d-full" and m["is_private"] and m["enable_gpu"] and m["enable_internet"] is True
    assert m["dataset_sources"] == ["draculess/rsna-weak-label-code", "draculess/rsna-weak-label-phase17a-inputs"] and m["competition_sources"] == ["rsna-knee-abnormality-detection"] and m["kernel_sources"] == []
    assert (d / "kaggle_phase19d_arm_d_full.py").read_text(encoding="utf-8") == DRIVER  # byte-identical; P19_MODE defaults to full


def _local_manifest():
    import subprocess
    import sys
    data = ROOT / "data" / "rsna-knee"
    sel = ROOT / "outputs" / "kaggle_smoke" / "p17a_inputs" / "selected_studies.json"
    if not (data / "train.csv").is_file() or not sel.is_file():
        pytest.skip("local competition metadata not present")
    return data, sel


def test_local_partition_preflight_reproduces_the_registered_seed_201_split_and_excludes_gold_and_test(tmp_path):
    import subprocess
    import sys
    from mri_core.rsna_knee_dataset import load_rsna_metadata
    from mri_core.weak_label_data import assert_no_leakage, hash_split, load_weak_manifest
    data, sel = _local_manifest()
    reg = json.loads((ROOT / "configs" / "phase17_registry.json").read_text())
    out = tmp_path / "m.jsonl"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_weak_label_manifest.py"), "--train-csv", str(data / "train.csv"), "--test-csv", str(data / "test.csv"),
                    "--sample-submission", str(data / "sample_submission.csv"), "--active-targets", "ACL,Medial Meniscus,Lateral Meniscus", "--output", str(out)], check=True, capture_output=True)
    manifest = load_weak_manifest(out)
    assert manifest.sha256 == reg["manifest_sha256"]
    selection = json.loads(sel.read_text())
    assert hashlib.sha256("\n".join(sorted(selection)).encode()).hexdigest() == reg["training_pool"]["sha256"] and len(selection) == 1200
    order = [u for u in manifest.ids if u in set(selection)]
    tr, va = hash_split(order, 0.2, 201)
    val_ids, train_ids = [order[i] for i in va], [order[i] for i in tr]
    assert (len(train_ids), len(val_ids)) == (962, 238)
    assert hashlib.sha256("\n".join(sorted(val_ids)).encode()).hexdigest() == reg["partitions"]["201"]["validation_sha256"]
    metadata = load_rsna_metadata(data, ROOT / "data" / "rsna-knee")
    gold = set(metadata.complete_label_train["StudyInstanceUID"].astype(str))
    test = set(metadata.test["StudyInstanceUID"].astype(str)) | set(metadata.sample_submission["StudyInstanceUID"].astype(str))
    assert_no_leakage(train_ids, val_ids, gold_ids=gold, test_ids=test)
    assert not (set(train_ids) | set(val_ids)) & (gold | test)


def test_phase17a_comparison_artifacts_are_present_for_the_paired_analysis():
    base = ROOT / "outputs" / "kaggle_smoke" / "out_p17a_seed201_v2" / "phase17a"
    if not base.is_dir():
        pytest.skip("Phase 17A outputs not present")
    for arm in ("B0", "C1", "S"):
        assert (base / "runs" / f"s201_{arm}" / "validation_predictions.csv").is_file()
    assert (base / "predictions_seed201.csv").is_file() and (base / "final_summary.json").is_file()
