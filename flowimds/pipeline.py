"""The newest pipeline core implementation after version 0.2.0."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Iterable, TypedDict

import numpy as np
from tqdm import tqdm

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
    worker_count: int
    log_enabled: bool


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
        worker_count: int | None = None,
        log: bool = False,
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
            worker_count: Maximum number of worker threads used to process
                images in parallel. ``None`` defaults to roughly 70% of the
                CPU cores reported by :func:`os.cpu_count`.
            log: Whether to emit informational logs and display progress
                updates during processing.
        """

        self._steps = list(steps)
        self._input_path = Path(input_path) if input_path is not None else None
        self._output_path = Path(output_path) if output_path is not None else None
        self._recursive = recursive
        self._preserve_structure = preserve_structure
        self._worker_count = worker_count
        self._log_enabled = log
        self._destination_lock = Lock()
        self._flattened_destination_registry: set[str] = set()

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
        if self._input_path is not None:
            msg = "input_path must not be provided to use run_on_paths()."
            raise ValueError(msg)

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

        materialised_paths = list(image_paths)
        if not materialised_paths:
            return 0, [], []

        self._reset_destination_registry()

        worker_count = self._resolve_worker_count()

        if self._log_enabled:
            logical_cores = os.cpu_count() or worker_count
            print(
                "[flowimds] Starting pipeline with "
                f"{len(materialised_paths)} images | "
                f"workers: {worker_count} / logical cores: {logical_cores}"
            )

        progress_bar = self._create_progress_bar(len(materialised_paths))
        try:
            if worker_count <= 1:
                return self._process_images_sequential(
                    materialised_paths,
                    progress_bar,
                )
            return self._process_images_parallel(
                materialised_paths,
                worker_count,
                progress_bar,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()

    def _process_images_sequential(
        self,
        image_paths: Iterable[Path],
        progress_bar,
    ) -> tuple[int, list[Path], list[OutputMapping]]:
        """Process images sequentially and return execution statistics."""

        processed_count = 0
        failed_files: list[Path] = []
        output_mappings: list[OutputMapping] = []

        total = len(image_paths)
        progress_tracker = self._create_progress_tracker(total, progress_bar)
        for index, image_path in enumerate(image_paths, start=1):
            success, destination = self._process_single_image(image_path)
            if success and destination is not None:
                output_mappings.append(OutputMapping(image_path, destination))
                processed_count += 1
            else:
                failed_files.append(image_path)
            progress_tracker(index)

        failed_files = list(dict.fromkeys(failed_files))
        return processed_count, failed_files, output_mappings

    def _process_images_parallel(
        self,
        image_paths: Iterable[Path],
        worker_count: int,
        progress_bar,
    ) -> tuple[int, list[Path], list[OutputMapping]]:
        """Process images in parallel using a thread pool."""

        processed_count = 0
        failed_files: list[Path] = []
        output_mappings: list[OutputMapping] = []

        total = len(image_paths)
        completed = 0
        progress_tracker = self._create_progress_tracker(total, progress_bar)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self._process_single_image, path)
                for path in image_paths
            ]
            for image_path, future in zip(image_paths, futures):
                try:
                    success, destination = future.result()
                except Exception:  # pragma: no cover - defensive
                    success, destination = False, None
                if success and destination is not None:
                    output_mappings.append(OutputMapping(image_path, destination))
                    processed_count += 1
                else:
                    failed_files.append(image_path)
                completed += 1
                progress_tracker(completed)

        failed_files = list(dict.fromkeys(failed_files))
        return processed_count, failed_files, output_mappings

    def _process_single_image(
        self,
        image_path: Path,
    ) -> tuple[bool, Path | None]:
        """Process a single image and return success flag and destination."""

        try:
            image = read_image(str(image_path))
            if image is None:
                return False, None
            image = self._apply_steps(image)
            destination = self._resolve_destination(image_path)
            if not write_image(str(destination), image):
                return False, None
            return True, destination
        except Exception:  # pragma: no cover - defensive
            return False, None

    def _resolve_worker_count(self) -> int:
        """Return the effective worker count for image processing."""

        if self._worker_count is not None and self._worker_count > 0:
            return self._worker_count
        cpu_count = os.cpu_count() or 1
        recommended = round(cpu_count * 0.7)
        return max(1, min(cpu_count, recommended))

    def _create_progress_tracker(self, total: int, progress_bar):
        """Return a callable that updates progress with minimal branching."""

        if not self._log_enabled or total <= 0:
            return lambda _completed: None

        if progress_bar is not None:
            return lambda completed: progress_bar.update(1)

        threshold = max(1, total // 10)

        def _log_progress(completed: int) -> None:
            if completed == 1 or completed == total or completed % threshold == 0:
                print(f"[flowimds] Progress: {completed}/{total} images processed")

        return _log_progress

    def _create_progress_bar(self, total: int):
        """Create and return a tqdm progress bar if available."""

        if not self._log_enabled or total <= 0:
            return None

        return tqdm(total=total, desc="flowimds", unit="image", leave=False)

    def _build_settings(self) -> PipelineSettings:
        """Return a typed dictionary that summarises the run configuration."""

        return PipelineSettings(
            input_path=str(self._input_path) if self._input_path is not None else None,
            output_path=(
                str(self._output_path) if self._output_path is not None else None
            ),
            recursive=self._recursive,
            preserve_structure=self._preserve_structure,
            worker_count=self._resolve_worker_count(),
            log_enabled=self._log_enabled,
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
        unique_name = self._ensure_unique_flat_filename(source.name)
        return destination_root / unique_name

    def _reset_destination_registry(self) -> None:
        """Reset the registry that tracks flattened destination filenames."""

        with self._destination_lock:
            self._flattened_destination_registry.clear()

    def _ensure_unique_flat_filename(self, filename: str) -> str:
        """Return a unique filename for flattened outputs.

        Args:
            filename: Original filename including extension.

        Returns:
            Unique filename that appends ``_no{n}`` on collisions.
        """

        stem, suffix = os.path.splitext(filename)
        counter = 1
        candidate = filename
        with self._destination_lock:
            while candidate in self._flattened_destination_registry:
                counter += 1
                candidate = f"{stem}_no{counter}{suffix}"
            self._flattened_destination_registry.add(candidate)
        return candidate

    @staticmethod
    def _ensure_array(image: np.ndarray, index: int) -> np.ndarray:
        """Validate that ``image`` is a numpy array and return it."""

        if not isinstance(image, np.ndarray):
            msg = f"images[{index}] must be a numpy.ndarray"
            raise TypeError(msg)
        return image
