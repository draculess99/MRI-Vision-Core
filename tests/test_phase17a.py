"""Phase 17A: label-mask override (B0/C1), registry/partition integrity, fingerprints, checkpoint preservation, resume and the comparison harness."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mri_core import weak_label_compare as cmp_
from mri_core import weak_label_mask_override as mo
from mri_core.rsna_knee_dataset import TARGET_COLUMNS
from mri_core.weak_label_data import hash_split, load_weak_manifest
from scripts.kaggle_weak_label_experiment import main as cli

from .test_weak_label_experiment import CFG
from .test_weak_label_real_smoke import build_manifest_for
from .test_weak_label_resolution import world  # noqa: F401  (fixture)

pytestmark = pytest.mark.rsna
TIER1 = ("ACL", "Medial Meniscus", "Lateral Meniscus")
J = {t: TARGET_COLUMNS.index(t) for t in TARGET_COLUMNS}


def toy(n=12):
    ids = [f"u{i}" for i in range(n)]
    values = np.full((n, 12), np.nan)
    masks = np.zeros((n, 12), np.float32)
    rng = np.random.default_rng(0)
    for t in TIER1:
        j = J[t]
        decided = rng.random(n) < 0.8
        masks[decided, j] = 1
        values[decided, j] = (rng.random(decided.sum()) < 0.4).astype(float)
    return ids, values, masks


def candidate_from(masks, keep_every=2):
    cand = np.zeros(masks.shape, bool)
    for t in TIER1:
        rows = np.flatnonzero(masks[:, J[t]] == 1)[::keep_every]
        cand[rows, J[t]] = True
    return cand


class TestMaskOverride:
    def test_b0_is_the_original_mask_and_c1_only_removes_entries(self):
        ids, values, masks = toy()
        override = mo.build_override(candidate_from(masks), values, ids, manifest_sha256="m", selection_sha256="s", targets=TIER1)
        b0 = mo.apply_override("B0", ids, values, masks, manifest_sha256="m", override=None)
        c1 = mo.apply_override("C1", ids, values, masks, manifest_sha256="m", override=override)
        assert np.array_equal(b0, masks) and b0 is not masks
        assert (c1 <= masks).all() and c1.sum() < masks.sum() and c1.sum() == sum(len(d["studies"]) for d in override["targets"].values())

    def test_label_values_and_missing_entries_are_never_changed_or_turned_negative(self):
        ids, values, masks = toy()
        before_values, before_masks = values.copy(), masks.copy()
        override = mo.build_override(candidate_from(masks), values, ids, manifest_sha256="m", selection_sha256="s", targets=TIER1)
        c1 = mo.apply_override("C1", ids, values, masks, manifest_sha256="m", override=override)
        assert np.array_equal(values, before_values, equal_nan=True) and np.array_equal(masks, before_masks)
        undecided = masks == 0
        assert (c1[undecided] == 0).all()  # missing/uncertain/masked entries stay masked, never trainable
        assert np.isnan(values[undecided]).all()

    def test_c1_requires_a_matching_override(self):
        ids, values, masks = toy()
        override = mo.build_override(candidate_from(masks), values, ids, manifest_sha256="m", selection_sha256="s", targets=TIER1)
        with pytest.raises(mo.MaskOverrideError):
            mo.apply_override("C1", ids, values, masks, manifest_sha256="m", override=None)
        with pytest.raises(mo.MaskOverrideError):
            mo.apply_override("C1", ids, values, masks, manifest_sha256="OTHER", override=override)
        with pytest.raises(mo.MaskOverrideError):
            mo.apply_override("C1", ids, values, masks, manifest_sha256="m", override=override, expected_fingerprint="0" * 64)
        with pytest.raises(mo.MaskOverrideError):
            mo.apply_override("C2", ids, values, masks, manifest_sha256="m", override=override)

    def test_override_that_disagrees_with_the_manifest_is_rejected(self):
        ids, values, masks = toy()
        override = mo.build_override(candidate_from(masks), values, ids, manifest_sha256="m", selection_sha256="s", targets=TIER1)
        flipped = values.copy()
        j = J["ACL"]
        row = override["targets"]["ACL"]["studies"][0]
        flipped[ids.index(row), j] = 1 - flipped[ids.index(row), j]
        with pytest.raises(mo.MaskOverrideError, match="value"):
            mo.apply_override("C1", ids, flipped, masks, manifest_sha256="m", override=override)
        masked = masks.copy()
        masked[ids.index(row), j] = 0
        with pytest.raises(mo.MaskOverrideError, match="masked"):
            mo.apply_override("C1", ids, values, masked, manifest_sha256="m", override=override)

    def test_studies_not_in_the_run_are_ignored_and_exact_ids_are_required(self):
        ids, values, masks = toy()
        override = mo.build_override(candidate_from(masks), values, ids, manifest_sha256="m", selection_sha256="s", targets=TIER1)
        subset = ids[:6]
        c1 = mo.apply_override("C1", subset, values[:6], masks[:6], manifest_sha256="m", override=override)
        assert c1.shape == (6, 12)
        shifted = [f"x{u}" for u in subset]  # misaligned IDs: nothing matches, so nothing is trainable (never silently reused)
        assert mo.apply_override("C1", shifted, values[:6], masks[:6], manifest_sha256="m", override=override).sum() == 0

    def test_fingerprint_is_order_independent_stable_and_tamper_evident(self, tmp_path):
        ids, values, masks = toy()
        override = mo.build_override(candidate_from(masks), values, ids, manifest_sha256="m", selection_sha256="s", targets=TIER1)
        again = mo.build_override(candidate_from(masks), values, ids, manifest_sha256="m", selection_sha256="s", targets=TIER1)
        assert override["fingerprint"] == again["fingerprint"]
        reordered = {"targets": {t: {"studies": d["studies"][::-1], "values": d["values"][::-1]} for t, d in override["targets"].items()}}
        assert mo.fingerprint({t: d["studies"] for t, d in reordered["targets"].items()}, {t: d["values"] for t, d in reordered["targets"].items()}) == override["fingerprint"]
        path = tmp_path / "o.json"
        path.write_text(json.dumps(override))
        assert mo.load_override(path)["fingerprint"] == override["fingerprint"]
        override["targets"]["ACL"]["values"][0] = 1 - override["targets"]["ACL"]["values"][0]
        path.write_text(json.dumps(override))
        with pytest.raises(mo.MaskOverrideError, match="fingerprint"):
            mo.load_override(path)


class TestFrozenDefinition:
    def test_candidate_and_registry_configs_are_consistent(self):
        reg = json.load(open("configs/phase17_registry.json"))
        cand = json.load(open("configs/phase17_candidate_v1.json"))
        p16 = json.load(open("configs/phase16_registry.json"))
        assert cand["candidate_fingerprint_sha256"] == reg["candidate_fingerprint_sha256"] == "e3cc0bd9f0806fc0fad4e3bd589cabce00a40937bdc9d92efed0e50e5912b4b9"
        assert cand["override_fingerprint_sha256"] == reg["override_fingerprint_sha256"]
        assert reg["training_pool"] == p16["training_pool"] and reg["manifest_sha256"] == p16["manifest_sha256"] == cand["source_manifest_sha256"]
        assert reg["primary_target"] == "Medial Meniscus" and set(reg["secondary_targets"]) == {"ACL", "Lateral Meniscus"}
        assert cand["counts_per_target"]["Medial Meniscus"] == {"studies": 874, "positives": 329}
        assert not set(reg["seeds"]) & ({0, 1} | set(p16["seeds"]))  # never-used seeds
        assert set(reg["partitions"]) == {str(s) for s in reg["seeds"]} and all(len(p["validation_sha256"]) == 64 for p in reg["partitions"].values())
        assert reg["evaluation_policy"]["claim_allowed_without_adjudicated_reference"] is False
        assert "already inspected" in reg["evaluation_policy"]["phase16_holdout"]

    def test_partition_recipe_is_deterministic_and_disjoint(self):
        ids = [f"u{i}" for i in range(400)]
        for seed in (201, 202):
            tr, va = hash_split(ids, 0.2, seed)
            assert not set(tr) & set(va) and sorted(tr + va) == list(range(400))
            assert (tr, va) == hash_split(list(ids), 0.2, seed)
        assert hash_split(ids, 0.2, 201)[1] != hash_split(ids, 0.2, 202)[1]

    def test_the_override_in_the_inputs_package_is_pinned_when_present(self):
        path = "outputs/kaggle_smoke/p17a_inputs/candidate_mask_v1.json"
        try:
            data = mo.load_override(path)
        except FileNotFoundError:
            pytest.skip("inputs package not built in this checkout")
        cand = json.load(open("configs/phase17_candidate_v1.json"))
        assert data["fingerprint"] == cand["override_fingerprint_sha256"]
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == cand["override_file_sha256"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class TestTrainingWithTheOverride:
    def setup(self, world):  # noqa: F811
        manifest = build_manifest_for(SimpleNamespace(meta=world.meta, tmp=world.tmp))
        rows = load_weak_manifest(manifest)
        override = mo.build_override(candidate_from(rows.masks), rows.values, list(rows.ids), manifest_sha256=rows.sha256, selection_sha256="s", targets=TIER1)
        path = world.tmp / "override.json"
        path.write_text(json.dumps(override))
        cfg = world.tmp / "cfg.json"
        cfg.write_text(json.dumps({**CFG, "epochs": 2, "val_fraction": 0.34, "num_workers": 0, "arm": "C", "image_size": 16, "slices_per_series": 3,
                                   "cache_dir": str(world.tmp / "cache")}))
        common = ["--data-dir", str(world.meta), "--dicom-root", str(world.raw), "--manifest", str(manifest), "--config", str(cfg)]
        return manifest, rows, override, path, common

    def test_c1_training_uses_the_candidate_mask_and_leaves_manifest_and_b0_checkpoint_untouched(self, world):  # noqa: F811
        manifest, rows, override, path, common = self.setup(world)
        manifest_hash = sha(manifest)
        assert cli(["train", *common, "--work-dir", str(world.tmp / "b0")]) == 0
        b0_hash = sha(world.tmp / "b0" / "last.pt")
        assert cli(["train", *common, "--work-dir", str(world.tmp / "c1"), "--label-mask", "C1", "--label-mask-file", str(path),
                    "--expected-mask-fingerprint", override["fingerprint"]]) == 0
        s_b0 = json.loads((world.tmp / "b0" / "split_summary.json").read_text())
        s_c1 = json.loads((world.tmp / "c1" / "split_summary.json").read_text())
        n = lambda s: sum(sum(c.values()) for split in ("train_targets", "validation_targets") for c in s[split].values())
        assert n(s_c1) < n(s_b0)
        assert sha(manifest) == manifest_hash and sha(world.tmp / "b0" / "last.pt") == b0_hash  # nothing existing is rewritten
        fp = lambda w: torch.load(world.tmp / w / "last.pt", map_location="cpu", weights_only=True)["dataset_fingerprint"]
        assert fp("b0") != fp("c1")  # a C1 checkpoint can never be resumed as a B0 run (or vice versa)
        state = torch.load(world.tmp / "c1" / "last.pt", map_location="cpu", weights_only=True)
        assert state["config"]["label_mask"]["fingerprint"] == override["fingerprint"]

    def test_c1_run_refuses_a_wrong_fingerprint_and_a_missing_file(self, world):  # noqa: F811
        manifest, rows, override, path, common = self.setup(world)
        with pytest.raises(mo.MaskOverrideError):
            cli(["train", *common, "--work-dir", str(world.tmp / "x"), "--label-mask", "C1", "--label-mask-file", str(path), "--expected-mask-fingerprint", "0" * 64])
        with pytest.raises(mo.MaskOverrideError):
            cli(["train", *common, "--work-dir", str(world.tmp / "y"), "--label-mask", "C1"])

    def test_interrupted_c1_run_resumes_to_the_same_result_and_a_b0_resume_is_refused(self, world):  # noqa: F811
        manifest, rows, override, path, common = self.setup(world)
        flags = ["--label-mask", "C1", "--label-mask-file", str(path)]
        assert cli(["train", *common, "--work-dir", str(world.tmp / "full"), *flags]) == 0
        assert cli(["train", *common, "--work-dir", str(world.tmp / "part"), *flags, "--stop-after-epochs", "1"]) == 0
        mid = torch.load(world.tmp / "part" / "last.pt", map_location="cpu", weights_only=True)
        assert mid["epoch"] == 1
        with pytest.raises(Exception):  # same work dir, different mask: the dataset fingerprint differs
            cli(["train", *common, "--work-dir", str(world.tmp / "part")])
        assert cli(["train", *common, "--work-dir", str(world.tmp / "part"), *flags]) == 0
        a = torch.load(world.tmp / "full" / "last.pt", map_location="cpu", weights_only=True)
        b = torch.load(world.tmp / "part" / "last.pt", map_location="cpu", weights_only=True)
        assert [h["train_loss"] for h in a["metrics"]["history"]] == [h["train_loss"] for h in b["metrics"]["history"]]
        assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])

    def test_score_validation_writes_only_the_checkpoints_own_validation_studies(self, world):  # noqa: F811
        manifest, rows, override, path, common = self.setup(world)
        assert cli(["train", *common, "--work-dir", str(world.tmp / "c1"), "--label-mask", "C1", "--label-mask-file", str(path)]) == 0
        ckpt = world.tmp / "c1" / "last.pt"
        before = sha(ckpt)
        out = world.tmp / "preds.csv"
        assert cli(["score-validation", *common, "--checkpoint", str(ckpt), "--output", str(out)]) == 0
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        lines = out.read_text().splitlines()
        studies = [l.split(",")[0] for l in lines[1:]]
        assert lines[0] == "study,ACL,Medial Meniscus,Lateral Meniscus" and studies == list(state["valid_study_ids"])
        assert not set(studies) & set(state["train_study_ids"]) and sha(ckpt) == before  # scoring never rewrites the checkpoint


class TestComparisonHarness:
    def reference(self, kind="weak", n=60, seed=0):
        rng = np.random.default_rng(seed)
        values = np.full((n, 12), np.nan)
        masks = np.zeros((n, 12), np.float32)
        for t in TIER1:
            masks[:, J[t]] = 1
            values[:, J[t]] = rng.integers(0, 2, n)
        return cmp_.ReferenceLabels(kind, tuple(f"r{i}" for i in range(n)), values, masks, "synthetic")

    def scores(self, ref, signal):
        rng = np.random.default_rng(1)
        y = np.stack([np.nan_to_num(ref.values[:, J[t]]) for t in TIER1], axis=1)
        return y * signal + rng.normal(size=y.shape)

    def test_auc_values_and_paired_difference_are_correct(self):
        ref = self.reference()
        y = np.stack([np.nan_to_num(ref.values[:, J[t]]) for t in TIER1], axis=1)
        perfect, inverted = y.copy(), -y
        res = cmp_.score_table(ref, {"good": perfect, "bad": inverted}, rounds=50, differences=[("good", "bad")])
        assert res["auc"]["good|macro"] == 1.0 and res["auc"]["bad|macro"] == 0.0 and res["auc"]["good|Medial Meniscus"] == 1.0
        diff = res["paired_difference"]["good|macro-bad|macro"]
        assert diff["point"] == 1.0 and diff["ci95"][0] == pytest.approx(1.0)
        assert res["primary_target"] == "Medial Meniscus" and res["secondary_targets"] == ["ACL", "Lateral Meniscus"]

    def test_weak_references_never_license_a_claim(self):
        ref = self.reference("weak")
        allowed, reasons = cmp_.claim_allowed(ref, {})
        assert not allowed and "weak" in reasons[0]
        res = cmp_.score_table(ref, {"a": self.scores(ref, 1.0)}, rounds=20)
        assert res["claim_allowed"] is False and res["reference_kind"] == "weak"

    def test_adjudicated_reference_must_be_disjoint_from_every_protected_set(self):
        ref = self.reference("adjudicated")
        assert cmp_.claim_allowed(ref, {"training": ["z1"], "gold": ["z2"]}) == (True, [])
        allowed, reasons = cmp_.claim_allowed(ref, {"phase16 hold-out pool": ["r1", "r2"], "gold": ["r3"]})
        assert not allowed and any("2 reference studies overlap the phase16 hold-out pool" in r for r in reasons)

    def test_misaligned_scores_are_rejected(self):
        ref = self.reference()
        with pytest.raises(ValueError, match="aligned"):
            cmp_.score_table(ref, {"a": np.zeros((5, 3))}, rounds=5)

    def test_across_seed_summary_is_descriptive(self):
        ref = self.reference()
        tables = [cmp_.score_table(ref, {"A": self.scores(ref, 2.0), "B": self.scores(ref, 0.0)}, rounds=30, seed=s, differences=[("A", "B")]) for s in range(3)]
        summary = cmp_.across_seeds(tables, [("A", "B")])
        assert summary["seeds"] == 3 and summary["arms"]["A"]["mean"] > summary["arms"]["B"]["mean"]
        assert summary["paired"]["A-B"]["of"] == 3 and "not a claim" in summary["note"]
