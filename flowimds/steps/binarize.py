"""Binarisation step implementation."""

from typing import Literal, cast

import cv2
import numpy as np

from flowimds.steps.base import (
    PipelineStep,
    ensure_image_has_spatial_dims,
    to_grayscale,
    validate_positive_int,
)

BinarizeMode = Literal["otsu", "fixed"]


class BinarizeStep(PipelineStep):
    """Pipeline step that thresholds images into binary form."""

    def __init__(
        self,
        *,
        mode: str = "otsu",
        threshold: int | None = None,
        max_value: int = 255,
    ) -> None:
        """Initialise the binarisation step.

        Args:
            mode: Binarisation strategy (``"otsu"`` or ``"fixed"``).
            threshold: Threshold used for ``"fixed"`` mode.
            max_value: Value assigned to pixels above the threshold.

        Raises:
            ValueError: If configuration values are invalid.
        """

        normalised_mode = mode.lower()
        if normalised_mode not in {"otsu", "fixed"}:
            raise ValueError("mode must be either 'otsu' or 'fixed'")
        self._mode: BinarizeMode = cast(BinarizeMode, normalised_mode)

        self._max_value = validate_positive_int(max_value, argument_name="max_value")
        self._threshold: int | None

        if self._mode == "fixed":
            if threshold is None:
                raise ValueError("threshold must be provided when mode='fixed'")
            if not isinstance(threshold, int):
                raise ValueError("threshold must be an integer value")
            if not 0 <= threshold <= self._max_value:
                raise ValueError("threshold must be between 0 and max_value")
            self._threshold = threshold
        else:
            if threshold is not None:
                raise ValueError("threshold cannot be provided when mode='otsu'")
            self._threshold = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Threshold the provided image according to the configured mode."""

        ensure_image_has_spatial_dims(image)
        grayscale = to_grayscale(image)

        if self._mode == "otsu":
            _, result = cv2.threshold(
                grayscale,
                0.0,
                self._max_value,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        else:
            if self._threshold is None:  # pragma: no cover - defensive
                msg = "threshold must be configured for fixed mode"
                raise ValueError(msg)
            _, result = cv2.threshold(
                grayscale,
                float(self._threshold),
                self._max_value,
                cv2.THRESH_BINARY,
            )
        return result.astype(grayscale.dtype, copy=False)
