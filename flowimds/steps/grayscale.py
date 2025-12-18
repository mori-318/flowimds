"""Grayscale conversion step."""

import numpy as np

from flowimds.steps.base import PipelineStep, to_grayscale


class GrayscaleStep(PipelineStep):
    """Pipeline step that converts images to grayscale."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Convert the provided image to grayscale, preserving dtype."""

        return to_grayscale(image)
