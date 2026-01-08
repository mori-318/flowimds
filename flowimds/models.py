"""Data models for pipeline processing."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np


@dataclass
class OutputMapping:
    """Mapping between an input file path and the persisted output path."""

    input_path: Path
    output_path: Path


@dataclass
class ProcessedImage:
    """Container that keeps transformed image data before persistence."""

    input_path: Path | None
    image: np.ndarray


class PipelineSettings(TypedDict):
    """Typed mapping representing pipeline configuration for a run."""

    input_path: str | None
    output_path: str | None
    recursive: bool
    worker_count: int
    log_enabled: bool
