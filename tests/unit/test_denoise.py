"""Tests describing the behaviour of ``DenoiseStep``."""

import cv2
import numpy as np
import pytest

from flowimds.steps import DenoiseStep


def test_denoise_median_matches_opencv() -> None:
    """Median mode should match the output of ``cv2.medianBlur``."""

    image = np.array(
        [
            [0, 255, 0, 255, 0],
            [255, 0, 255, 0, 255],
            [0, 255, 0, 255, 0],
            [255, 0, 255, 0, 255],
            [0, 255, 0, 255, 0],
        ],
        dtype=np.uint8,
    )

    step = DenoiseStep(mode="median", kernel_size=3)
    actual = step.apply(image)
    expected = cv2.medianBlur(image, 3)

    assert np.array_equal(actual, expected)


def test_denoise_median_requires_odd_kernel_size() -> None:
    """Median mode must use an odd kernel size of at least three."""

    with pytest.raises(ValueError):
        DenoiseStep(mode="median", kernel_size=4)


def test_denoise_bilateral_matches_opencv() -> None:
    """Bilateral mode should match ``cv2.bilateralFilter`` output."""

    image = np.stack([np.linspace(0, 255, 9, dtype=np.uint8)] * 9, axis=0)
    image = np.dstack([image] * 3)

    step = DenoiseStep(
        mode="bilateral",
        diameter=5,
        sigma_color=25.0,
        sigma_space=25.0,
    )
    actual = step.apply(image)
    expected = cv2.bilateralFilter(image, 5, 25.0, 25.0)

    assert np.array_equal(actual, expected)


def test_denoise_bilateral_requires_positive_parameters() -> None:
    """Bilateral parameters must be strictly positive."""

    with pytest.raises(ValueError):
        DenoiseStep(mode="bilateral", diameter=0)

    with pytest.raises(ValueError):
        DenoiseStep(mode="bilateral", diameter=5, sigma_color=0)

    with pytest.raises(ValueError):
        DenoiseStep(
            mode="bilateral",
            diameter=5,
            sigma_color=1.0,
            sigma_space=0,
        )


def test_denoise_rejects_unknown_mode() -> None:
    """Only ``median`` and ``bilateral`` modes are supported."""

    with pytest.raises(ValueError):
        DenoiseStep(mode="gaussian")


def test_denoise_rejects_inputs_without_spatial_dimensions() -> None:
    """One-dimensional arrays must be rejected."""

    image = np.zeros(10, dtype=np.uint8)

    step = DenoiseStep(mode="median")

    with pytest.raises(ValueError):
        step.apply(image)
