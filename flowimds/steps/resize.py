"""Resize step implementations."""

import cv2
import numpy as np

from flowimds.steps.base import (
    PipelineStep,
    ensure_image_has_spatial_dims,
    validate_size_pair,
)


class ResizeStep(PipelineStep):
    """Pipeline step that resizes images to a fixed size using OpenCV."""

    def __init__(self, size: tuple[int, int]) -> None:
        """Initialise the resize step with the output ``(width, height)``.

        Args:
            size: Pair of positive integers representing ``(width, height)``.

        Raises:
            ValueError: If ``size`` is not a tuple of two positive integers.
        """

        self._size = validate_size_pair(size)

    @property
    def size(self) -> tuple[int, int]:
        """Return the configured ``(width, height)`` pair."""

        return self._size

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Resize the provided image to the configured size.

        Args:
            image: Image to resize.

        Returns:
            Resized image.

        Raises:
            ValueError: If ``image`` is not a 2D or 3D numpy array.
        """

        ensure_image_has_spatial_dims(image)
        return cv2.resize(image, self._size)
