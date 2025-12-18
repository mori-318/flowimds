"""Pipeline core implementation prior to version 0.2.0."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, TypedDict

import numpy as np

from flowimds.steps import PipelineStep
from flowimds.utils.image_discovery import IMAGE_SUFFIXES, collect_image_paths
from flowimds.utils.image_io import read_image, write_image


@dataclass
class OutputMapping:
    """Mapping between an input file path and the persisted output path."""

    input_path: Path
    output_path: Path


class PipelineSettings(TypedDict):
    """Typed mapping representing pipeline configuration for a run."""

    input_path: str | None
    output_path: str | None
    recursive: bool
    preserve_structure: bool


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
    settings: PipelineSettings


class Pipeline:
    """Image processing pipeline that orchestrates a sequence of steps."""

    def __init__(
        self,
        steps: Iterable[PipelineStep],
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
        recursive: bool = False,
        preserve_structure: bool = False,
    ) -> None:
        """Initialise the pipeline with the provided configuration.

        Args:
            steps: Iterable of processing steps that expose ``apply``.
            input_path: Directory that stores the source images. Required when
                using :meth:`run`.
            output_path: Directory where processed images are written. Required
                when using :meth:`run` or :meth:`run_on_paths`.
            recursive: Whether to traverse the input directory recursively.
            preserve_structure: Whether to mirror the input directory structure.
        """

        self._steps = list(steps)
        self._input_path = Path(input_path) if input_path is not None else None
        self._output_path = Path(output_path) if output_path is not None else None
        self._recursive = recursive
        self._preserve_structure = preserve_structure

    def run(self) -> PipelineResult:
        """Execute the pipeline and return the aggregated result."""

        if self._input_path is None:
            msg = "input_path must be provided to use run()."
            raise ValueError(msg)
        if self._output_path is None:
            msg = "output_path must be provided to use run()."
            raise ValueError(msg)

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

    def run_on_paths(self, paths: Iterable[Path | str]) -> PipelineResult:
        """Process an explicit iterable of image paths.

        Args:
            paths: Iterable of filesystem paths pointing to images.

        Returns:
            ``PipelineResult`` describing the execution outcome.
        """

        if self._output_path is None:
            msg = "output_path must be provided to use run_on_paths()."
            raise ValueError(msg)

        image_paths = [Path(path) for path in paths]
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

    def run_on_arrays(self, images: Iterable[np.ndarray]) -> list[np.ndarray]:
        """Apply pipeline steps to a collection of in-memory images.

        Args:
            images: Iterable of numpy arrays representing images.

        Returns:
            List of transformed images, in the same order as ``images``.

        Raises:
            TypeError: If any element is not a numpy array.
        """

        results: list[np.ndarray] = []
        for index, image in enumerate(images):
            normalised = self._ensure_array(image, index)
            results.append(self._apply_steps(normalised))
        return results

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

    def _build_settings(self) -> PipelineSettings:
        """Return a typed dictionary that summarises the run configuration."""

        return PipelineSettings(
            input_path=str(self._input_path) if self._input_path is not None else None,
            output_path=(
                str(self._output_path) if self._output_path is not None else None
            ),
            recursive=self._recursive,
            preserve_structure=self._preserve_structure,
        )

    def _apply_steps(self, image: np.ndarray) -> np.ndarray:
        """Apply pipeline steps to the provided image in sequence."""

        transformed = image
        for step in self._steps:
            transformed = step.apply(transformed)
        return transformed

    def _collect_image_paths(self) -> list[Path]:
        """Collect eligible image paths from the input directory."""

        if self._input_path is None:
            msg = "input_path must be provided to collect image paths."
            raise ValueError(msg)

        return collect_image_paths(
            self._input_path,
            recursive=self._recursive,
            suffixes=IMAGE_SUFFIXES,
        )

    def _resolve_destination(self, source: Path) -> Path:
        """Resolve the output destination path for the given source.

        Args:
            source: Path to the source file.

        Returns:
            Path to the destination file.
        """

        destination_root = self._output_path
        if destination_root is None:
            msg = "output_path must be provided to persist results."
            raise ValueError(msg)

        if self._preserve_structure and self._input_path is not None:
            try:
                relative = source.relative_to(self._input_path)
            except ValueError:
                relative = Path(source.name)
            return destination_root / relative
        return destination_root / source.name

    @staticmethod
    def _ensure_array(image: np.ndarray, index: int) -> np.ndarray:
        """Validate that ``image`` is a numpy array and return it."""

        if not isinstance(image, np.ndarray):
            msg = f"images[{index}] must be a numpy.ndarray"
            raise TypeError(msg)
        return image
