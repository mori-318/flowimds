"""Pipeline core implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Protocol

import numpy as np

from flowimds.utils.utils import read_image, write_image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


class PipelineStep(Protocol):
    """Protocol describing a pipeline step."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Transform the provided image."""


@dataclass
class OutputMapping:
    """Mapping between an input file path and the persisted output path."""

    input_path: Path
    output_path: Path


@dataclass
class PipelineResult:
    """Result of a pipeline run.

    Attributes:
        processed_count: Total number of successfully processed images.
        failed_count: Total number of images that failed to process.
        failed_files: Paths of the files that failed to process.
        output_mappings: Mapping objects that describe output destinations.
        duration_seconds: Execution time in seconds.
        settings: Settings that were in effect for the run.
    """

    processed_count: int
    failed_count: int
    failed_files: list[str]
    output_mappings: list[OutputMapping]
    duration_seconds: float
    settings: dict[str, Any]


class Pipeline:
    """Image processing pipeline that orchestrates a sequence of steps."""

    def __init__(
        self,
        steps: Iterable[PipelineStep],
        input_path: Path | str,
        output_path: Path | str,
        recursive: bool = False,
        preserve_structure: bool = False,
    ) -> None:
        """Initialise the pipeline with the provided configuration.

        Args:
            steps: Iterable of processing steps that expose ``apply``.
            input_path: Directory that stores the source images.
            output_path: Directory where processed images are written.
            recursive: Whether to traverse the input directory recursively.
            preserve_structure: Whether to mirror the input directory structure.
        """

        self._steps = list(steps)
        self._input_path = Path(input_path)
        self._output_path = Path(output_path)
        self._recursive = recursive
        self._preserve_structure = preserve_structure

    def run(self) -> PipelineResult:
        """Execute the pipeline and return the aggregated result."""

        image_paths = self._collect_image_paths()
        start = perf_counter()
        processed, failed, mappings = self._process_images(image_paths)
        duration = perf_counter() - start

        return PipelineResult(
            processed_count=processed,
            failed_count=len(failed),
            failed_files=[str(path) for path in failed],
            output_mappings=mappings,
            duration_seconds=duration,
            settings=self._build_settings(),
        )

    def _process_images(
        self,
        image_paths: Iterable[Path],
    ) -> tuple[int, list[Path], list[OutputMapping]]:
        """Process the provided image paths and persist the results."""

        processed_count = 0
        failed_files: list[Path] = []
        output_mappings: list[OutputMapping] = []

        for image_path in image_paths:
            try:
                image = read_image(str(image_path))
                if image is None:
                    failed_files.append(image_path)
                    continue
                image = self._apply_steps(image)
                destination = self._resolve_destination(image_path)
                if not write_image(str(destination), image):
                    failed_files.append(image_path)
                    continue
                output_mappings.append(OutputMapping(image_path, destination))
                processed_count += 1
            except Exception:  # pragma: no cover - defensive
                failed_files.append(image_path)

        failed_files = list(dict.fromkeys(failed_files))
        return processed_count, failed_files, output_mappings

    def _build_settings(self) -> dict[str, Any]:
        """Return a dictionary that summarises the run configuration."""

        return {
            "input_path": str(self._input_path),
            "output_path": str(self._output_path),
            "recursive": self._recursive,
            "preserve_structure": self._preserve_structure,
        }

    def _apply_steps(self, image: np.ndarray) -> np.ndarray:
        """Apply pipeline steps to the provided image in sequence."""

        transformed = image
        for step in self._steps:
            transformed = step.apply(transformed)
        return transformed

    def _collect_image_paths(self) -> list[Path]:
        """Collect eligible image paths from the input directory."""

        if not self._input_path.exists():
            msg = f"Input path '{self._input_path}' does not exist."
            raise FileNotFoundError(msg)

        iterator: Iterable[Path]
        if self._recursive:
            iterator = self._input_path.rglob("*")
        else:
            iterator = self._input_path.glob("*")

        image_paths = [
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        return sorted(image_paths, key=lambda path: path.as_posix())

    def _resolve_destination(self, source: Path) -> Path:
        """Resolve the output destination path for the given source.

        Args:
            source: Path to the source file.

        Returns:
            Path to the destination file.
        """

        destination_root = self._output_path
        if self._preserve_structure:
            relative = source.relative_to(self._input_path)
            return destination_root / relative
        return destination_root / source.name
