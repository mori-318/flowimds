"""Tests describing the behaviour of ``BinarizeStep``."""

import cv2
import numpy as np
import pytest

from flowimds.steps import BinarizeStep


def test_binarize_fixed_threshold_matches_opencv() -> None:
    """Fixed thresholding should mirror OpenCV's behaviour exactly."""

    image = np.array(
        [
            [10, 120, 200],
            [50, 130, 220],
            [5, 125, 230],
        ],
        dtype=np.uint8,
    )

    step = BinarizeStep(mode="fixed", threshold=128)
    actual = step.apply(image)
    _, expected = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    assert np.array_equal(actual, expected)
    assert actual.dtype == image.dtype


def test_binarize_otsu_mode_converts_colour_images() -> None:
    """Otsu mode should handle colour inputs by converting to grayscale."""

    colour = np.dstack(
        [
            np.tile(np.array([0, 128, 255], dtype=np.uint8), (3, 1)),
            np.tile(np.array([255, 128, 0], dtype=np.uint8), (3, 1)),
            np.tile(np.array([64, 64, 64], dtype=np.uint8), (3, 1)),
        ]
    ).transpose(1, 0, 2)

    step = BinarizeStep(mode="otsu")
    actual = step.apply(colour)

    grayscale = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    _, expected = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    assert np.array_equal(actual, expected)


def test_binarize_fixed_requires_threshold() -> None:
    """The fixed mode must receive an explicit threshold value."""

    with pytest.raises(ValueError):
        BinarizeStep(mode="fixed")


@pytest.mark.parametrize("threshold", [-1, 1000])
def test_binarize_fixed_rejects_out_of_range_threshold(threshold: int) -> None:
    """Fixed mode thresholds must be between 0 and the max value."""

    with pytest.raises(ValueError):
        BinarizeStep(mode="fixed", threshold=threshold)


def test_binarize_otsu_rejects_manual_threshold() -> None:
    """Providing a threshold to Otsu mode is invalid."""

    with pytest.raises(ValueError):
        BinarizeStep(mode="otsu", threshold=100)


def test_binarize_rejects_unknown_mode() -> None:
    """Only ``otsu`` and ``fixed`` modes are supported."""

    with pytest.raises(ValueError):
        BinarizeStep(mode="adaptive")
