"""Resolution / slice-count configurations: supported shapes, ordering, masking, normalisation, and the one-decode multi-variant cache."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from mri_core.rsna_knee_dataset import SERIES_COLUMNS, TARGET_COLUMNS, load_rsna_metadata, load_series_volume
from mri_core.rsna_knee_train import RSNAKneeDicomDataset, _normalize
from mri_core.weak_label_data import WeakLabelDicomDataset
from mri_core.weak_label_models import build_model
from mri_core.weak_label_multicache import MultiVariantWarmDataset, parse_variants, variants_for_study
from scripts.kaggle_weak_label_experiment import main as cli

from .dicom_factory import AXIAL, CORONAL, SAGITTAL, write_series
from .test_weak_label_experiment import CFG
from .test_weak_label_real_smoke import build_manifest_for

pytestmark = pytest.mark.rsna

VARIANTS = [(128, 9), (192, 9), (128, 15), (192, 15)]
PLANES = (("Axial", AXIAL), ("Coronal", CORONAL), ("Sagittal", SAGITTAL))
SLICES_IN_VOLUME = {"Axial": 24, "Coronal": 12, "Sagittal": 17}  # the coronal series cannot fill 15 slots


@pytest.fixture
def world(tmp_path):
    meta, raw = tmp_path / "meta", tmp_path / "raw"
    meta.mkdir()
    ids = [f"1.2.826.{i}" for i in range(1, 7)]
    series_rows = []
    for uid in ids:
        for plane, orientation in PLANES:
            series_uid = f"{uid}.s.{plane}"
            series_rows.append((uid, series_uid, 1, 0, plane))
            write_series(raw / "train_series" / uid / series_uid, count=SLICES_IN_VOLUME[plane], orientation=orientation, spacing=3.0,
                         series_uid=series_uid, study_uid=uid, rows=10, columns=12, textured=True)
    pd.DataFrame([{"StudyInstanceUID": u, "Report": "r", **{t: np.nan for t in TARGET_COLUMNS}} for u in ids]).to_csv(meta / "train.csv", index=False)
    pd.DataFrame(series_rows, columns=SERIES_COLUMNS).to_csv(meta / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": []}).to_csv(meta / "test.csv", index=False)
    pd.DataFrame(columns=SERIES_COLUMNS).to_csv(meta / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [], **{t: [] for t in TARGET_COLUMNS}}).to_csv(meta / "sample_submission.csv", index=False)
    return SimpleNamespace(meta=meta, raw=raw, ids=ids, metadata=load_rsna_metadata(meta, raw), tmp=tmp_path)


def config(size, slices):
    return {**CFG, "image_size": size, "slices_per_series": slices}


def production_sample(world, size, slices, index=0):
    frame = pd.DataFrame({"StudyInstanceUID": world.ids, **{t: 0.0 for t in TARGET_COLUMNS}})
    return RSNAKneeDicomDataset(world.metadata, frame, config(size, slices), "train")[index]


@pytest.mark.parametrize("size,slices", VARIANTS)
class TestProductionPreprocessingSupportsEveryConfiguration:
    def test_shapes_and_masks(self, world, size, slices):
        s = production_sample(world, size, slices)
        assert tuple(s["images"].shape) == (3, slices, 1, size, size) and s["mask"].shape == (3, slices)
        expected_valid = [min(slices, SLICES_IN_VOLUME[p]) for p in ("Axial", "Coronal", "Sagittal")]
        assert s["mask"].sum(dim=1).tolist() == expected_valid
        assert torch.isfinite(s["images"]).all() and 0.0 <= float(s["images"].min()) and float(s["images"].max()) <= 1.0
        assert float(s["images"][~s["mask"]].abs().sum()) == 0.0  # padding stays exactly zero

    def test_matches_an_independent_recomputation_in_anatomical_order(self, world, size, slices):
        uid = world.ids[0]
        s = production_sample(world, size, slices)
        for plane_index, (plane, _) in enumerate(PLANES):
            volume = load_series_volume(world.metadata, uid, f"{uid}.s.{plane}")
            chosen = np.linspace(0, volume.num_slices - 1, min(slices, volume.num_slices), dtype=int)
            assert list(chosen) == sorted(set(chosen))  # strictly increasing along the position-validated slice order
            for k, source in enumerate(chosen):
                ref = F.interpolate(torch.from_numpy(_normalize(volume.get_slice(int(source)))).unsqueeze(0).unsqueeze(0),
                                    size=(size, size), mode="bilinear", align_corners=False)[0]
                assert torch.equal(s["images"][plane_index, k], ref.float())  # production stores the float64 result as float32

    def test_every_model_arm_accepts_the_configuration(self, world, size, slices):
        s = production_sample(world, size, slices)
        for arm in ("A", "B", "C"):
            out = build_model(arm, config(size, slices)).eval()(s["images"][None], s["mask"][None])
            assert out.shape == (1, 12) and torch.isfinite(out).all()


def test_normalisation_is_per_slice_and_independent_of_resolution_and_slice_count(world):
    a, b = production_sample(world, 128, 9), production_sample(world, 192, 15)
    # slice index 0 of each plane is the same source slice in both configurations: same 1-99 percentile window
    for plane_index in range(3):
        up = F.interpolate(a["images"][plane_index, 0].unsqueeze(0), size=(192, 192), mode="bilinear", align_corners=False)[0]
        assert float((up - b["images"][plane_index, 0]).abs().mean()) < 0.05
        assert float(a["images"][plane_index, 0].amax()) <= 1.0 and float(b["images"][plane_index, 0].amax()) <= 1.0


class TestMultiVariantCache:
    def test_parse_variants(self):
        assert parse_variants("128x9, 192X15") == [(128, 9), (192, 15)]
        with pytest.raises(ValueError):
            parse_variants("128x9,128x9")

    def test_one_decode_yields_tensors_bitwise_equal_to_the_production_pipeline(self, world):
        built = variants_for_study(world.metadata, world.ids[2], ("Axial", "Coronal", "Sagittal"), VARIANTS)
        for (size, slices), (images, mask) in built.items():
            ref = production_sample(world, size, slices, index=2)
            assert torch.equal(images, ref["images"]) and torch.equal(mask, ref["mask"])

    def test_cache_entries_written_by_the_builder_are_hits_for_the_training_dataset(self, world):
        cache = world.tmp / "cache"
        warm = MultiVariantWarmDataset(world.metadata, world.ids, VARIANTS, CFG, cache)
        rows = [warm[i] for i in range(len(world.ids))]
        assert all(r["ok"] and r["decoded"] for r in rows) and len(list(cache.glob("*.pt"))) == len(world.ids) * len(VARIANTS)
        assert all(not warm[i]["decoded"] for i in range(len(world.ids)))  # a second pass reuses valid entries
        for size, slices in VARIANTS:
            ds = WeakLabelDicomDataset(world.metadata, world.ids, np.zeros((6, 12), np.float32), np.zeros((6, 12), np.float32),
                                       config(size, slices), cache_dir=cache)
            sample = ds[3]
            ref = production_sample(world, size, slices, index=3)
            assert torch.equal(sample["images"], ref["images"].half().float()) and ds.cache_hits == 1 and ds.cache_misses == 0

    def test_unreadable_study_is_reported_not_raised(self, world):
        victim = world.ids[1]
        for f in (world.raw / "train_series" / victim / f"{victim}.s.Axial").iterdir():
            f.write_bytes(b"corrupt")
        warm = MultiVariantWarmDataset(world.metadata, world.ids, VARIANTS[:2], CFG, world.tmp / "cache")
        assert [warm[i]["ok"] for i in range(len(world.ids))] == [True, False, True, True, True, True]

    def test_cli_cache_warm_multi_then_training_on_the_included_set(self, world):
        manifest = build_manifest_for(SimpleNamespace(meta=world.meta, tmp=world.tmp))
        cfg = world.tmp / "cfg.json"
        cfg.write_text(json.dumps({**CFG, "epochs": 1, "val_fraction": 0.3, "num_workers": 0, "cache_dir": str(world.tmp / "cache"), "arm": "C"}))
        common = ["--data-dir", str(world.meta), "--dicom-root", str(world.raw), "--manifest", str(manifest), "--config", str(cfg)]
        assert cli(["cache-warm-multi", *common, "--work-dir", str(world.tmp / "warm"), "--smoke-subset", "5", "--min-per-class", "1",
                    "--variants", "16x3,16x5"]) == 0
        report = json.loads((world.tmp / "warm" / "cache_warm_multi.json").read_text())
        assert report["studies"] == 5 and set(report["variants"]) == {"16x3", "16x5"} and report["unreadable"] == 0
        assert all(v["entries"] == 5 for v in report["variants"].values())
        sel = world.tmp / "warm" / "selected_studies.json"
        cfg5 = world.tmp / "cfg5.json"
        cfg5.write_text(json.dumps({**json.loads(cfg.read_text()), "image_size": 16, "slices_per_series": 5}))
        assert cli(["train", "--data-dir", str(world.meta), "--dicom-root", str(world.raw), "--manifest", str(manifest), "--config", str(cfg5),
                    "--include-studies", str(sel), "--work-dir", str(world.tmp / "run")]) == 0
        state = torch.load(world.tmp / "run" / "last.pt", map_location="cpu", weights_only=True)
        assert sorted(state["train_study_ids"] + state["valid_study_ids"]) == json.loads(sel.read_text())
