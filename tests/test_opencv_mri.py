import numpy as np

from hackathon.opencv_aws.opencv_mri import (
    create_overlay,
    detect_edges,
    enhance_contrast,
    normalize_to_uint8,
    process_mri_slice,
)


def test_normalize_to_uint8():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)

    result = normalize_to_uint8(image)

    assert result.dtype == np.uint8
    assert result.shape == (10, 10)
    assert result.min() == 0
    assert result.max() == 255


def test_constant_image():
    image = np.ones((32, 32), dtype=np.float32)

    result = normalize_to_uint8(image)

    assert np.all(result == 0)


def test_enhance_contrast():
    image = np.random.default_rng(0).normal(
        100,
        10,
        size=(128, 128),
    ).astype(np.float32)

    result = enhance_contrast(image)

    assert result.shape == (128, 128)
    assert result.dtype == np.uint8


def test_detect_edges():
    image = np.zeros((128, 128), dtype=np.float32)
    image[32:96, 32:96] = 1.0

    edges = detect_edges(image)

    assert edges.shape == (128, 128)
    assert edges.dtype == np.uint8
    assert np.any(edges > 0)


def test_create_overlay():
    image = np.zeros((128, 128), dtype=np.float32)
    image[32:96, 32:96] = 1.0

    overlay = create_overlay(image)

    assert overlay.shape == (128, 128, 3)
    assert overlay.dtype == np.uint8


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