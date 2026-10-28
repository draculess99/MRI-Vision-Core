"""Phase 19C v1 failure: the 48 stability studies come from the registered TRAINING partition, so re-splitting them with the registered 0.2 fraction leaves no validation study."""
import hashlib
import json
import pathlib

import pytest

from mri_core.weak_label_data import hash_split, stability_split_preflight

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 201
KAGGLE_48 = ROOT / "outputs" / "kaggle_smoke" / "out_p19c_stability_v1" / "phase19" / "stability_studies.json"


def bucket(uid):
    return int(hashlib.sha256(f"{SEED}:{uid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def registered_training_first_48():
    """Reproduces how the driver picks studies: the first 48 of the registered training partition (hash >= 0.2)."""
    ids = [f"1.2.826.0.1.{i:06d}" for i in range(400)]
    train_idx, val_idx = hash_split(ids, 0.2, SEED)
    return [ids[i] for i in train_idx][:48], [ids[i] for i in val_idx]


def test_original_failure_is_reproduced_by_resplitting_registered_training_studies_with_0_2():
    sub, _ = registered_training_first_48()
    assert len(sub) == 48 and all(bucket(u) >= 0.2 for u in sub)
    with pytest.raises(ValueError, match="empty train or validation"):
        hash_split(sub, 0.2, SEED)  # the exact error of the first Kaggle launch
    with pytest.raises(ValueError, match="stability split is empty"):
        stability_split_preflight(sub, [], 0.2, SEED)  # now caught by the preflight before any GPU work


def test_val_fraction_0_4_gives_a_non_empty_internal_split_that_never_touches_the_registered_validation_set():
    sub, registered_val = registered_training_first_48()
    report = stability_split_preflight(sub, registered_val, 0.4, SEED)
    assert report["internal_train"] > 0 and report["internal_validation"] > 0 and report["internal_train"] + report["internal_validation"] == 48
    train, val = hash_split(sub, 0.4, SEED)
    assert all(0.2 <= bucket(sub[i]) < 0.4 for i in val) and not {sub[i] for i in val} & set(registered_val)  # the band [0.2, 0.4) of TRAINING studies
    with pytest.raises(ValueError, match="registered validation partition"):
        stability_split_preflight(sub + registered_val[:1], registered_val, 0.4, SEED)


def test_preflight_refuses_partitions_that_are_too_small_to_be_a_health_check():
    sub, _ = registered_training_first_48()
    with pytest.raises(ValueError, match="too small"):
        stability_split_preflight(sub, [], 0.4, SEED, min_val=999)


@pytest.mark.skipif(not KAGGLE_48.is_file(), reason="Kaggle v1 output not present")
def test_the_actual_48_stability_studies_give_30_internal_training_and_18_internal_validation_studies():
    ids = json.loads(KAGGLE_48.read_text())
    assert len(ids) == 48
    with pytest.raises(ValueError):
        hash_split(ids, 0.2, SEED)
    report = stability_split_preflight(ids, [], 0.4, SEED)
    assert (report["internal_train"], report["internal_validation"]) == (30, 18)
    registry = json.loads((ROOT / "configs" / "phase17_registry.json").read_text())
    assert registry["partitions"][str(SEED)]["validation_sha256"].startswith("b97c2728")


def test_driver_keeps_the_registered_full_partition_and_runs_the_preflight_before_the_gpu_step():
    source = (ROOT / "scripts" / "kaggle_phase19_driver.py").read_text(encoding="utf-8")
    assert 'VAL_FRACTION = 0.2 if MODE == "full" else 0.4' in source and "P19_VAL_FRACTION" not in source
    assert source.index("stability_split_preflight(sub, val_ids, VAL_FRACTION, SEED)") < source.index('"stability_part1"')
    assert '"val_fraction": VAL_FRACTION' in source and '"early_stopping_patience": 0' in source  # no model selection on the internal split
    assert "never reported as benchmark performance" in source and "last_best.pt" in source
