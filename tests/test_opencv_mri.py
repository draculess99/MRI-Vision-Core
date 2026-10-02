import numpy as np
import pytest

from hackathon.opencv_aws.opencv_mri import (
    OpenCVMRIResult,
    create_overlay,
    detect_edges,
    enhance_contrast,
    normalize_to_uint8,
    process_mri_slice,
    save_result,
)


# ---------------------------------------------------------------------
# normalize_to_uint8
# ---------------------------------------------------------------------

def test_normalize_to_uint8():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)

    result = normalize_to_uint8(image)

    assert result.dtype == np.uint8
    assert result.shape == (10, 10)
    assert result.min() == 0
    assert result.max() == 255


def test_normalize_preserves_shape():
    image = np.random.default_rng(0).normal(size=(64, 96)).astype(np.float32)

    result = normalize_to_uint8(image)

    assert result.shape == (64, 96)


def test_constant_image():
    image = np.ones((32, 32), dtype=np.float32)

    result = normalize_to_uint8(image)

    assert np.all(result == 0)


def test_all_zero_image():
    image = np.zeros((32, 32), dtype=np.float32)

    result = normalize_to_uint8(image)

    assert np.all(result == 0)
    assert result.dtype == np.uint8


def test_negative_values_are_normalized():
    image = np.array(
        [
            [-10.0, -5.0],
            [0.0, 10.0],
        ],
        dtype=np.float32,
    )

    result = normalize_to_uint8(image)

    assert result.min() == 0
    assert result.max() == 255


def test_nan_and_inf_are_handled():
    image = np.array(
        [
            [0.0, np.nan],
            [np.inf, 10.0],
        ],
        dtype=np.float32,
    )

    result = normalize_to_uint8(image)

    assert result.dtype == np.uint8
    assert result.shape == (2, 2)
    assert np.isfinite(result).all()


def test_all_non_finite_returns_zeros():
    image = np.array(
        [
            [np.nan, np.inf],
            [-np.inf, np.nan],
        ],
        dtype=np.float32,
    )

    result = normalize_to_uint8(image)

    assert np.all(result == 0)


def test_invalid_3d_input_rejected():
    image = np.zeros((32, 32, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        normalize_to_uint8(image)


# ---------------------------------------------------------------------
# enhance_contrast
# ---------------------------------------------------------------------

def test_enhance_contrast():
    image = np.random.default_rng(0).normal(
        100,
        10,
        size=(128, 128),
    ).astype(np.float32)

    result = enhance_contrast(image)

    assert result.shape == (128, 128)
    assert result.dtype == np.uint8


def test_enhance_contrast_constant_image():
    image = np.ones((128, 128), dtype=np.float32)

    result = enhance_contrast(image)

    assert result.shape == (128, 128)
    assert result.dtype == np.uint8


def test_enhance_contrast_non_square_image():
    image = np.random.default_rng(3).normal(
        size=(80, 120),
    ).astype(np.float32)

    result = enhance_contrast(image)

    assert result.shape == (80, 120)


# ---------------------------------------------------------------------
# detect_edges
# ---------------------------------------------------------------------

def test_detect_edges():
    image = np.zeros((128, 128), dtype=np.float32)
    image[32:96, 32:96] = 1.0

    edges = detect_edges(image)

    assert edges.shape == (128, 128)
    assert edges.dtype == np.uint8
    assert np.any(edges > 0)


def test_detect_edges_blank_image():
    image = np.zeros((128, 128), dtype=np.float32)

    edges = detect_edges(image)

    assert edges.shape == image.shape
    assert edges.dtype == np.uint8


def test_detect_edges_binary_values():
    image = np.zeros((128, 128), dtype=np.float32)
    image[40:90, 40:90] = 100.0

    edges = detect_edges(image)

    unique = set(np.unique(edges).tolist())

    assert unique.issubset({0, 255})


# ---------------------------------------------------------------------
# create_overlay
# ---------------------------------------------------------------------

def test_create_overlay():
    image = np.zeros((128, 128), dtype=np.float32)
    image[32:96, 32:96] = 1.0

    overlay = create_overlay(image)

    assert overlay.shape == (128, 128, 3)
    assert overlay.dtype == np.uint8


def test_create_overlay_with_explicit_edges():
    image = np.zeros((64, 64), dtype=np.float32)

    edges = np.zeros((64, 64), dtype=np.uint8)
    edges[20:40, 20] = 255

    overlay = create_overlay(image, edges)

    assert overlay.shape == (64, 64, 3)

    edge_pixels = overlay[edges > 0]

    assert edge_pixels.shape[0] > 0
    assert np.all(edge_pixels[:, 0] == 0)
    assert np.all(edge_pixels[:, 1] == 255)
    assert np.all(edge_pixels[:, 2] == 0)


def test_overlay_contains_three_channels():
    image = np.random.default_rng(7).normal(
        size=(64, 64),
    ).astype(np.float32)

    overlay = create_overlay(image)

    assert overlay.ndim == 3
    assert overlay.shape[2] == 3


# ---------------------------------------------------------------------
# process_mri_slice
# ---------------------------------------------------------------------

def test_process_mri_slice():
    image = np.random.default_rng(1).normal(
        100,
        20,
        size=(128, 128),
    ).astype(np.float32)

    result = process_mri_slice(image)

    assert result.original.shape == (128, 128)
    assert result.enhanced.shape == (128, 128)
    assert result.edges.shape == (128, 128)
    assert result.overlay.shape == (128, 128, 3)


def test_process_returns_dataclass():
    image = np.random.default_rng(4).normal(
        size=(64, 64),
    ).astype(np.float32)

    result = process_mri_slice(image)

    assert isinstance(result, OpenCVMRIResult)


def test_process_output_dtypes():
    image = np.random.default_rng(5).normal(
        size=(64, 64),
    ).astype(np.float32)

    result = process_mri_slice(image)

    assert result.original.dtype == np.uint8
    assert result.enhanced.dtype == np.uint8
    assert result.edges.dtype == np.uint8
    assert result.overlay.dtype == np.uint8


def test_process_non_square_image():
    image = np.random.default_rng(6).normal(
        size=(80, 120),
    ).astype(np.float32)

    result = process_mri_slice(image)

    assert result.original.shape == (80, 120)
    assert result.enhanced.shape == (80, 120)
    assert result.edges.shape == (80, 120)
    assert result.overlay.shape == (80, 120, 3)


# ---------------------------------------------------------------------
# save_result
# ---------------------------------------------------------------------

def test_save_result_creates_all_files(tmp_path):
    image = np.random.default_rng(10).normal(
        size=(64, 64),
    ).astype(np.float32)

    result = process_mri_slice(image)

    paths = save_result(
        result,
        tmp_path,
        prefix="test_mri",
    )

    assert set(paths.keys()) == {
        "original",
        "enhanced",
        "edges",
        "overlay",
    }

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_save_result_uses_prefix(tmp_path):
    image = np.zeros((64, 64), dtype=np.float32)

    result = process_mri_slice(image)

    paths = save_result(
        result,
        tmp_path,
        prefix="knee_case_001",
    )

    assert paths["original"].name == "knee_case_001_original.png"
    assert paths["enhanced"].name == "knee_case_001_enhanced.png"
    assert paths["edges"].name == "knee_case_001_edges.png"
    assert paths["overlay"].name == "knee_case_001_overlay.png"


def test_save_result_creates_missing_directory(tmp_path):
    output_dir = tmp_path / "nested" / "results"

    image = np.zeros((32, 32), dtype=np.float32)
    result = process_mri_slice(image)

    paths = save_result(
        result,
        output_dir,
    )

    assert output_dir.exists()

    for path in paths.values():
        assert path.exists()


# ---------------------------------------------------------------------
# deterministic behavior
# ---------------------------------------------------------------------

def test_processing_is_deterministic():
    image = np.random.default_rng(11).normal(
        size=(128, 128),
    ).astype(np.float32)

    result1 = process_mri_slice(image)
    result2 = process_mri_slice(image)

    assert np.array_equal(result1.original, result2.original)
    assert np.array_equal(result1.enhanced, result2.enhanced)
    assert np.array_equal(result1.edges, result2.edges)
    assert np.array_equal(result1.overlay, result2.overlay)