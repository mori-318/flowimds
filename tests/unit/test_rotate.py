"""Tests describing the behaviour of ``RotateStep``."""

import numpy as np
import pytest
import cv2

from flowimds.steps import RotateStep


def _grid(width: int, height: int, *, channels: int = 1) -> np.ndarray:
    """Return an array with monotonically increasing values across the grid."""

    base = np.arange(width * height * channels, dtype=np.uint8)
    shape = (height, width, channels) if channels > 1 else (height, width)
    return base.reshape(shape)


def test_rotate_with_expand_matches_numpy_rot90() -> None:
    """A 90° rotation with expansion should match ``np.rot90`` output."""

    image = _grid(3, 2)

    step = RotateStep(90, expand=True, interpolation=cv2.INTER_NEAREST)
    actual = step.apply(image)
    expected = np.rot90(image, k=1)

    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)


def test_rotate_without_expand_preserves_shape() -> None:
    """Disabling expansion should keep the original image bounds."""

    image = _grid(4, 4, channels=3)

    step = RotateStep(180, expand=False, interpolation=cv2.INTER_NEAREST)
    actual = step.apply(image)
    expected = np.rot90(image, k=2)

    assert actual.shape == image.shape
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("angle", ["ninety", None])
def test_rotate_rejects_non_numeric_angles(angle: object) -> None:
    """Angles must be numeric values."""

    with pytest.raises(ValueError):
        RotateStep(angle)  # type: ignore[arg-type]


def test_rotate_rejects_images_without_spatial_dimensions() -> None:
    """Input arrays must provide at least two spatial dimensions."""

    image = np.zeros(10, dtype=np.uint8)

    step = RotateStep(45)

    with pytest.raises(ValueError):
        step.apply(image)
