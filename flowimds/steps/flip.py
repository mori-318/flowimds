"""Flip step implementation."""

import cv2
import numpy as np

from flowimds.steps.base import PipelineStep, ensure_image_has_spatial_dims


class FlipStep(PipelineStep):
    """Pipeline step that flips images horizontally and/or vertically."""

    def __init__(self, *, horizontal: bool = False, vertical: bool = False) -> None:
        """Initialise the flip step.

        Args:
            horizontal: Whether to flip along the horizontal axis.
            vertical: Whether to flip along the vertical axis.

        Raises:
            ValueError: If neither axis is enabled.
        """

        if not horizontal and not vertical:
            raise ValueError("At least one of horizontal or vertical must be True")
        self._horizontal = bool(horizontal)
        self._vertical = bool(vertical)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Flip the provided image according to the configured axes."""

        ensure_image_has_spatial_dims(image)
        if self._horizontal and self._vertical:
            flip_code = -1
        elif self._horizontal:
            flip_code = 1
        else:
            flip_code = 0
        return cv2.flip(image, flip_code)
