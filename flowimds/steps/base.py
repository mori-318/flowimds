"""Protocol and helper utilities for pipeline steps."""

from typing import Protocol, Sequence

import cv2
import numpy as np


class PipelineStep(Protocol):
    """Protocol describing the contract for pipeline steps."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Transform the provided image and return the result."""


def validate_positive_int(value: int, *, argument_name: str) -> int:
    """Validate that ``value`` is a positive integer.

    Args:
        value: Candidate value.
        argument_name: Name used in the exception message.

    Returns:
        Normalised integer value.

    Raises:
        ValueError: If ``value`` is not a positive integer.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{argument_name} must be a positive integer"
        raise ValueError(msg)
    if value <= 0:
        msg = f"{argument_name} must be greater than zero"
        raise ValueError(msg)
    return int(value)


def validate_odd_integer(
    value: int,
    *,
    argument_name: str,
    minimum: int = 1,
) -> int:
    """Validate that ``value`` is an odd integer.

    Ensures ``value`` is at least ``minimum``.
    """

    candidate = validate_positive_int(value, argument_name=argument_name)
    if candidate % 2 == 0:
        msg = f"{argument_name} must be an odd integer"
        raise ValueError(msg)
    if candidate < minimum:
        msg = f"{argument_name} must be at least {minimum}"
        raise ValueError(msg)
    return candidate


def validate_size_pair(
    size: Sequence[int],
    *,
    argument_name: str = "size",
) -> tuple[int, int]:
    """Ensure ``size`` is a ``(width, height)`` pair of positive integers.

    Args:
        size: Candidate sequence representing ``(width, height)``.
        argument_name: Name used in the exception message.

    Returns:
        Tuple of two positive integers.

    Raises:
        ValueError: If the sequence does not contain two positive integers.
    """

    if len(size) != 2 or not all(isinstance(dimension, int) for dimension in size):
        msg = f"{argument_name} must be a pair of integers (width, height)"
        raise ValueError(msg)
    width, height = size
    if width <= 0 or height <= 0:
        msg = f"{argument_name} dimensions must be positive integers"
        raise ValueError(msg)
    return int(width), int(height)


def ensure_image_has_spatial_dims(
    image: np.ndarray,
    *,
    argument_name: str = "image",
) -> None:
    """Validate that ``image`` is a 2D or 3D numpy array.

    Args:
        image: Array to validate.
        argument_name: Name used in the exception message.

    Raises:
        ValueError: If ``image`` lacks spatial dimensions.
    """

    if image.ndim not in {2, 3}:
        msg = f"{argument_name} must be a 2D or 3D numpy array"
        raise ValueError(msg)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert ``image`` to grayscale if necessary."""

    ensure_image_has_spatial_dims(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
