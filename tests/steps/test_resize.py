from typing import Any

import numpy as np
import pytest

from flowimds.steps import ResizeStep


@pytest.mark.parametrize(
    "size, expected_shape",
    [
        ((200, 200), (200, 200, 3)),
        ((32, 64), (64, 32, 3)),
    ],
)
def test_resize_with_valid_size(
    size: tuple[int, int],
    expected_shape: tuple[int, ...],
) -> None:
    """Verify ``ResizeStep`` resizes colour images to the configured size."""

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    step = ResizeStep(size)
    resized_image = step.apply(image)

    assert resized_image.shape == expected_shape
    assert resized_image.dtype == image.dtype


def test_resize_preserves_dtype_for_grayscale_images() -> None:
    """Ensure grayscale images retain shape and dtype after resizing."""

    size = (16, 16)
    image = np.zeros((8, 8), dtype=np.uint16)

    step = ResizeStep(size)
    resized_image = step.apply(image)

    assert resized_image.shape == (16, 16)
    assert resized_image.dtype == image.dtype


@pytest.mark.parametrize(
    "size",
    [
        (-10, 200),
        (0, 100),
        (100, 0),
        (100,),
        (100, 100, 100),
        ("wide", 100),
        (100, "tall"),
    ],
)
def test_resize_with_invalid_size(size: Any) -> None:
    """Ensure ``ResizeStep`` rejects invalid size definitions."""

    with pytest.raises(ValueError):
        ResizeStep(size)


def test_resize_rejects_invalid_image_dimensions() -> None:
    """Ensure ``ResizeStep`` raises when provided images lack spatial dims."""

    size = (10, 10)
    image = np.zeros((100,), dtype=np.uint8)

    step = ResizeStep(size)

    with pytest.raises(ValueError):
        step.apply(image)
