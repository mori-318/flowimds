"""Noise reduction step implementations."""

import cv2
import numpy as np

from flowimds.steps.base import (
    PipelineStep,
    ensure_image_has_spatial_dims,
    validate_odd_integer,
    validate_positive_int,
)


class DenoiseStep(PipelineStep):
    """Pipeline step that applies median or bilateral filtering."""

    def __init__(
        self,
        *,
        mode: str = "median",
        kernel_size: int = 3,
        diameter: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
    ) -> None:
        """Initialise the denoising step.

        Args:
            mode: Denoising strategy (``"median"`` or ``"bilateral"``).
            kernel_size: Median filter kernel size (odd integer >= 3).
            diameter: Diameter of the pixel neighbourhood for bilateral mode.
            sigma_color: Filter sigma in the colour space for bilateral mode.
            sigma_space: Filter sigma in coordinate space for bilateral mode.

        Raises:
            ValueError: If configuration values are invalid.
        """

        normalised_mode = mode.lower()
        if normalised_mode not in {"median", "bilateral"}:
            raise ValueError("mode must be either 'median' or 'bilateral'")
        self._mode = normalised_mode

        if self._mode == "median":
            self._kernel_size = validate_odd_integer(
                kernel_size,
                argument_name="kernel_size",
                minimum=3,
            )
        else:
            self._diameter = validate_positive_int(diameter, argument_name="diameter")
            if sigma_color <= 0:
                raise ValueError("sigma_color must be greater than zero")
            if sigma_space <= 0:
                raise ValueError("sigma_space must be greater than zero")
            self._sigma_color = float(sigma_color)
            self._sigma_space = float(sigma_space)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the configured denoising operation to ``image``."""

        ensure_image_has_spatial_dims(image)
        if self._mode == "median":
            return cv2.medianBlur(image, self._kernel_size)
        return cv2.bilateralFilter(
            image,
            self._diameter,
            self._sigma_color,
            self._sigma_space,
        )
