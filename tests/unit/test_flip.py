"""Tests describing the behaviour of ``FlipStep``."""

import numpy as np
import pytest

from flowimds.steps import FlipStep


def test_flip_horizontal_matches_numpy_fliplr() -> None:
    """Horizontal flips should mirror the image horizontally."""

    image = np.arange(12, dtype=np.uint8).reshape(3, 4)

    step = FlipStep(horizontal=True)
    actual = step.apply(image)
    expected = np.fliplr(image)

    assert np.array_equal(actual, expected)


def test_flip_vertical_matches_numpy_flipud() -> None:
    """Vertical flips should mirror the image vertically."""

    image = np.arange(24, dtype=np.uint8).reshape(3, 4, 2)

    step = FlipStep(vertical=True)
    actual = step.apply(image)
    expected = np.flipud(image)

    assert np.array_equal(actual, expected)


def test_flip_both_axes_matches_numpy_combination() -> None:
    """Enabling both axes should match sequential horizontal and vertical flips."""

    image = np.arange(16, dtype=np.uint8).reshape(4, 4)

    step = FlipStep(horizontal=True, vertical=True)
    actual = step.apply(image)
    expected = np.flipud(np.fliplr(image))

    assert np.array_equal(actual, expected)


def test_flip_requires_at_least_one_axis() -> None:
    """At least one axis must be enabled."""

    with pytest.raises(ValueError):
        FlipStep()


def test_flip_rejects_inputs_without_spatial_dimensions() -> None:
    """One-dimensional arrays must be rejected."""

    image = np.zeros(10, dtype=np.uint8)

    step = FlipStep(horizontal=True)

    with pytest.raises(ValueError):
        step.apply(image)
