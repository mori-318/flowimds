"""Protocol definitions for pipeline steps."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class PipelineStep(Protocol):
    """Protocol describing the contract for pipeline steps."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Transform the provided image and return the result."""


def validate_size_pair(size: Sequence[int], *, argument_name: str = "size") -> tuple[int, int]:
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


def ensure_image_has_spatial_dims(image: np.ndarray, *, argument_name: str = "image") -> None:
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
