"""Tests describing the behaviour of ``GrayscaleStep``."""

import cv2
import numpy as np
import pytest

from flowimds.steps import GrayscaleStep


def test_grayscale_converts_colour_images_to_single_channel() -> None:
    """Colour inputs should be converted using OpenCV's BGR→gray conversion."""

    image = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [128, 128, 128]],
        ],
        dtype=np.uint8,
    )

    step = GrayscaleStep()
    actual = step.apply(image)
    expected = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    assert actual.ndim == 2
    assert actual.dtype == image.dtype
    assert np.array_equal(actual, expected)


def test_grayscale_leaves_grayscale_images_unchanged() -> None:
    """Already-grayscale inputs should pass through without modification."""

    image = np.arange(9, dtype=np.uint8).reshape(3, 3)

    step = GrayscaleStep()
    actual = step.apply(image)

    assert np.array_equal(actual, image)


def test_grayscale_rejects_inputs_without_spatial_dimensions() -> None:
    """One-dimensional arrays must be rejected."""

    image = np.zeros(10, dtype=np.uint8)

    step = GrayscaleStep()

    with pytest.raises(ValueError):
        step.apply(image)
