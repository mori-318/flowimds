"""Rotate step implementation."""

from typing import Union
import math

import cv2
import numpy as np

from flowimds.steps.base import PipelineStep, ensure_image_has_spatial_dims

Number = Union[int, float]


class RotateStep(PipelineStep):
    """Pipeline step that rotates images around their centre."""

    def __init__(
        self,
        angle: Number,
        *,
        expand: bool = True,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
    ) -> None:
        """Initialise the rotation step.

        Args:
            angle: Rotation angle in degrees (counter-clockwise).
            expand: Whether to expand the canvas to avoid cropping.
            interpolation: Interpolation mode used by OpenCV.
            border_mode: Border handling strategy for uncovered pixels.

        Raises:
            ValueError: If ``angle`` is not numeric.
        """

        if not isinstance(angle, (int, float)):
            raise ValueError("angle must be a numeric value")
        self._angle = float(angle)
        self._expand = bool(expand)
        self._interpolation = interpolation
        self._border_mode = border_mode

    @property
    def angle(self) -> float:
        """Return the configured rotation angle in degrees."""

        return self._angle

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Rotate the provided image by the configured angle."""

        ensure_image_has_spatial_dims(image)
        height, width = image.shape[:2]

        right_angle_steps = self._right_angle_steps()
        if right_angle_steps is not None:
            rotated = np.rot90(image, k=right_angle_steps)
            if self._expand or rotated.shape[:2] == (height, width):
                return rotated

        centre = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(centre, self._angle, 1.0)

        if self._expand:
            new_width, new_height = self._compute_expanded_dimensions(
                matrix,
                width,
                height,
            )
        else:
            new_width = width
            new_height = height

        return cv2.warpAffine(
            image,
            matrix,
            (new_width, new_height),
            flags=self._interpolation,
            borderMode=self._border_mode,
        )

    def _right_angle_steps(self) -> int | None:
        """Return the number of 90° steps if ``angle`` is a right angle."""

        steps_float = self._angle / 90.0
        steps = int(round(steps_float))
        if not math.isclose(steps_float, steps, abs_tol=1e-6):
            return None
        return steps % 4

    def _compute_expanded_dimensions(
        self,
        matrix: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        """Return the canvas size required to fit the rotated image."""

        corners = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )
        ones = np.ones((corners.shape[0], 1), dtype=np.float32)
        homogenous = np.hstack([corners, ones])
        affine = np.vstack([matrix, [0.0, 0.0, 1.0]])
        transformed = homogenous @ affine.T

        min_x = transformed[:, 0].min()
        max_x = transformed[:, 0].max()
        min_y = transformed[:, 1].min()
        max_y = transformed[:, 1].max()

        new_width = int(math.ceil(max_x - min_x + 1.0))
        new_height = int(math.ceil(max_y - min_y + 1.0))

        matrix[0, 2] -= min_x
        matrix[1, 2] -= min_y

        return new_width, new_height
