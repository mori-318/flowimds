"""The newest pipeline core implementation after version 0.2.0."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Callable, Iterable, TypedDict, cast

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
        processed_images: In-memory processed images for deferred saving.
        source_root: Base path used for structure preservation when saving.
    """

    processed_count: int
    failed_count: int
    failed_files: list[str]
    output_mappings: list[OutputMapping]
    duration_seconds: float
    settings: PipelineSettings
    processed_images: list[ProcessedImage]
    source_root: Path | None

    def save(self, output_dir: Path | str) -> None:
        """Persist processed images to ``output_dir``.

        Args:
            output_dir: Destination directory where processed images are written.
        """

        if not self.processed_images:
            return

        destination_root = Path(output_dir)
        destination_root.mkdir(parents=True, exist_ok=True)

        preserve_structure = bool(self.settings.get("preserve_structure", False))
        source_root = self.source_root
        input_setting = self.settings.get("input_path")
        if source_root is None and input_setting:
            source_root = Path(input_setting)

        used_names: set[str] = set()
        used_names_lock = Lock()

        def _allocate_filename(base_name: str) -> str:
            with used_names_lock:
                return self._ensure_unique_name(base_name, used_names)

        def _resolve_destination(index: int, source: Path | None) -> Path:
            if (
                preserve_structure
                and source is not None
                and source_root is not None
                and source.is_relative_to(source_root)
            ):
                relative_path = source.relative_to(source_root)
                return destination_root / relative_path
            if source is not None:
                filename = _allocate_filename(source.name)
            else:
                filename = _allocate_filename(f"image_{index}.png")
            return destination_root / filename

        def _persist(
            index: int, processed: ProcessedImage
        ) -> tuple[bool, OutputMapping | str]:
            source = processed.input_path
            image = processed.image

            destination_path = _resolve_destination(index, source)
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            if write_image(str(destination_path), image):
                return True, OutputMapping(
                    input_path=source or Path(f"array_{index}"),
                    output_path=destination_path,
                )
            return False, str(source) if source else f"array_{index}"

        total = len(self.processed_images)
        worker_count = max(1, int(self.settings.get("worker_count", 1)))
        log_enabled = bool(self.settings.get("log_enabled", False))

        if log_enabled:
            logical_cores = os.cpu_count()
            print(
                "[flowimds] Saving "
                f"{total} images | workers: {worker_count} / logical cores: "
                f"{logical_cores}"
            )

        progress_bar = (
            tqdm(total=total, desc="flowimds (save)", unit="image", leave=False)
            if log_enabled and total > 0
            else None
        )

        def _update_progress() -> None:
            if progress_bar is not None:
                progress_bar.update(1)

        tasks = list(enumerate(self.processed_images, start=1))

        try:
            if worker_count <= 1 or total <= 1:
                for index, processed in tasks:
                    success, result = _persist(index, processed)
                    if success:
                        mapping = cast(OutputMapping, result)
                        self.output_mappings.append(mapping)
                    else:
                        failed = cast(str, result)
                        self.failed_files.append(failed)
                    _update_progress()
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_to_payload = {
                        executor.submit(_persist, index, processed): (index, processed)
                        for index, processed in tasks
                    }
                    for future in as_completed(future_to_payload):
                        index, processed = future_to_payload[future]
                        try:
                            success, result = future.result()
                        except Exception:
                            success = False
                            result = (
                                str(processed.input_path)
                                if processed.input_path is not None
                                else f"array_{index}"
                            )
                        if success:
                            mapping = cast(OutputMapping, result)
                            self.output_mappings.append(mapping)
                        else:
                            failed = cast(str, result)
                            self.failed_files.append(failed)
                        _update_progress()
        finally:
            if progress_bar is not None:
                progress_bar.close()

        self.processed_count = len(self.output_mappings)
        self.failed_count = len(self.failed_files)

    @staticmethod
    def _ensure_unique_name(filename: str, used_names: set[str]) -> str:
        """Return a unique filename for flattened saves."""

        stem, suffix = os.path.splitext(filename)
        candidate = filename
        counter = 1
        while candidate in used_names:
            counter += 1
            candidate = f"{stem}_no{counter}{suffix}"
        used_names.add(candidate)
        return candidate


class Pipeline:
    """Image processing pipeline that orchestrates a sequence of steps."""

    def __init__(
        self,
        steps: Iterable[PipelineStep],
        recursive: bool = False,
        preserve_structure: bool = False,
        worker_count: int | None = None,
        log: bool = False,
    ) -> None:
        """Initialise the pipeline with the provided configuration.

        Args:
            steps: Iterable of processing steps that expose ``apply``.
            recursive: Whether to traverse the input directory recursively.
            preserve_structure: Whether to mirror the input directory structure.
            worker_count: Maximum number of worker threads used to process
                images in parallel. ``None`` defaults to roughly 70% of the
                CPU cores reported by :func:`os.cpu_count`.
            log: Whether to emit informational logs and display progress
                updates during processing.
        """

        self._steps = list(steps)
        self._recursive = recursive
        self._preserve_structure = preserve_structure
        self._worker_count = worker_count
        self._log_enabled = log
        self._input_path: Path | None = None
        self._output_path: Path | None = None
        self._destination_lock = Lock()
        self._flattened_destination_registry: set[str] = set()

    def run(
        self,
        *,
        input_path: Path | str | None = None,
        input_paths: Iterable[Path | str] | None = None,
        input_arrays: Iterable[np.ndarray] | None = None,
    ) -> PipelineResult:
        """Execute the pipeline and return the aggregated result.

        Exactly one of ``input_path``, ``input_paths``, or ``input_arrays`` may be
        provided. When none are provided, the instance-level ``input_path`` is used.
        """

        self._validate_input_selection(
            input_path=input_path,
            input_paths=input_paths,
            input_arrays=input_arrays,
        )
        if input_arrays is not None:
            return self._run_on_arrays(input_arrays)

        image_paths, source_root = self._resolve_image_sources(
            directory=input_path,
            explicit_paths=input_paths,
        )

        return self._run_in_memory(image_paths, source_root)

    def _resolve_image_sources(
        self,
        directory: Path | str | None,
        explicit_paths: Iterable[Path | str] | None,
    ) -> tuple[list[Path], Path | None]:
        """Resolve image sources and return a list of paths and source root.

        Args:
            directory: Directory path to search.
            explicit_paths: List of explicitly provided image paths.

        Returns:
            Tuple of image path list and source root path.

        Raises:
            ValueError: When no valid input source is provided.
            FileNotFoundError: When the specified path does not exist.
        """
        if explicit_paths is not None:
            image_paths = [Path(path) for path in explicit_paths]
            source_root = self._determine_source_root(image_paths)
            return image_paths, source_root

        if directory is not None:
            dir_path = Path(directory)
            if not dir_path.exists():
                msg = f"Input path '{dir_path}' does not exist."
                raise FileNotFoundError(msg)

            self._input_path = dir_path
            image_paths = collect_image_paths(
                dir_path,
                recursive=self._recursive,
                suffixes=IMAGE_SUFFIXES,
            )
            return image_paths, dir_path

        msg = "input_path, input_paths, or input_arrays must be specified."
        raise ValueError(msg)

    def _run_in_memory(
        self,
        image_paths: list[Path],
        source_root: Path | None,
    ) -> PipelineResult:
        """Process images and keep transformed data in memory.

        Args:
            image_paths: Paths to source images to process without saving.
            source_root: Base directory used when mirroring structure at save time.

        Returns:
            ``PipelineResult`` that stores transformed images for deferred save.
        """

        start = perf_counter()
        processed_images, failed_files = self._process_images(image_paths)
        duration = perf_counter() - start

        return PipelineResult(
            processed_count=len(processed_images),
            failed_count=len(failed_files),
            failed_files=failed_files,
            output_mappings=[],
            duration_seconds=duration,
            settings=self._build_settings(),
            processed_images=processed_images,
            source_root=source_root,
        )

    def _run_on_arrays(
        self,
        images: Iterable[np.ndarray],
    ) -> PipelineResult:
        """Process in-memory images and return a result with deferred outputs.

        Args:
            images: Iterable of numpy arrays representing images.

        Returns:
            ``PipelineResult`` containing transformed images ready for saving.
        """

        start = perf_counter()
        processed_images: list[ProcessedImage] = []
        failed_files: list[str] = []

        for index, image in enumerate(images, start=1):
            try:
                normalised = self._ensure_array(image, index - 1)
                transformed = self._apply_steps(normalised)
                processed_images.append(
                    ProcessedImage(input_path=None, image=transformed),
                )
            except Exception:  # pragma: no cover - defensive
                failed_files.append(f"array_{index}")

        duration = perf_counter() - start

        return PipelineResult(
            processed_count=len(processed_images),
            failed_count=len(failed_files),
            failed_files=failed_files,
            output_mappings=[],
            duration_seconds=duration,
            settings=self._build_settings(),
            processed_images=processed_images,
            source_root=None,
        )

    def _process_images(
        self,
        image_paths: Iterable[Path],
    ) -> tuple[list[ProcessedImage], list[str]]:
        """Process the provided image paths and keep outputs in memory."""

        materialised_paths = list(image_paths)
        if not materialised_paths:
            return [], []

        worker_count = self._resolve_worker_count()

        if self._log_enabled:
            logical_cores = os.cpu_count() or worker_count
            print(
                "[flowimds] Starting pipeline with "
                f"{len(materialised_paths)} images | "
                f"workers: {worker_count} / logical cores: {logical_cores}"
            )

        progress_bar, progress_tracker = self._prepare_progress(len(materialised_paths))
        try:
            if worker_count <= 1:
                return self._process_images_sequential(
                    materialised_paths,
                    progress_tracker,
                )
            return self._process_images_parallel(
                materialised_paths,
                worker_count,
                progress_tracker,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()

    def _process_images_sequential(
        self,
        image_paths: Iterable[Path],
        progress_tracker: Callable[[int], None],
    ) -> tuple[list[ProcessedImage], list[str]]:
        """Process images sequentially and return execution statistics."""

        processed_images: list[ProcessedImage] = []
        failed_files: list[str] = []

        paths = list(image_paths)
        total = len(paths)
        for index, image_path in enumerate(paths, start=1):
            success, processed = self._process_single_image(image_path)
            if success and processed is not None:
                processed_images.append(processed)
            else:
                failed_files.append(str(image_path))
            progress_tracker(index)

        failed_files = list(dict.fromkeys(failed_files))
        return processed_images, failed_files

    def _process_images_parallel(
        self,
        image_paths: Iterable[Path],
        worker_count: int,
        progress_tracker: Callable[[int], None],
    ) -> tuple[list[ProcessedImage], list[str]]:
        """Process images in parallel using a thread pool."""

        processed_images: list[ProcessedImage] = []
        failed_files: list[str] = []

        paths = list(image_paths)
        total = len(paths)
        completed = 0

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_path = {
                executor.submit(self._process_single_image, path): path
                for path in paths
            }
            for future in as_completed(future_to_path):
                image_path = future_to_path[future]
                try:
                    success, processed = future.result()
                except Exception:  # pragma: no cover - defensive
                    success, processed = False, None
                if success and processed is not None:
                    processed_images.append(processed)
                else:
                    failed_files.append(str(image_path))
                completed += 1
                progress_tracker(completed)

        failed_files = list(dict.fromkeys(failed_files))
        return processed_images, failed_files

    def _process_single_image(
        self,
        image_path: Path,
    ) -> tuple[bool, ProcessedImage | None]:
        """Process a single image and return success flag and data."""

        try:
            image = read_image(str(image_path))
            if image is None:
                return False, None
            transformed = self._apply_steps(image)
            return True, ProcessedImage(input_path=image_path, image=transformed)
        except Exception:  # pragma: no cover - defensive
            return False, None

    def _resolve_worker_count(self) -> int:
        """Return the effective worker count for image processing."""

        if self._worker_count is not None and self._worker_count > 0:
            return self._worker_count
        cpu_count = os.cpu_count() or 1
        recommended = round(cpu_count * 0.7)
        return max(1, min(cpu_count, recommended))

    def _prepare_progress(
        self,
        total: int,
    ) -> tuple[tqdm | None, Callable[[int], None]]:
        """Create progress bar and tracker with consistent lifecycle handling."""

        progress_bar = self._create_progress_bar(total)
        progress_tracker = self._create_progress_tracker(total, progress_bar)
        return progress_bar, progress_tracker

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

    def _resolve_image_paths(
        self,
        explicit_paths: Iterable[Path | str] | None,
    ) -> list[Path]:
        """Return image paths gathered from arguments or ``input_path``."""

        if explicit_paths is not None:
            return [Path(path) for path in explicit_paths]
        if self._input_path is None:
            msg = "input_path must be provided to use run()."
            raise ValueError(msg)
        return self._collect_image_paths()

    @contextmanager
    def _temporary_io_overrides(
        self,
        input_path: Path | str | None,
        output_path: Path | str | None,
    ):
        """Temporarily override ``input_path``/``output_path`` during a run."""

        original_input_path = self._input_path
        original_output_path = self._output_path

        if input_path is not None:
            self._input_path = Path(input_path)
        if output_path is not None:
            self._output_path = Path(output_path)

        try:
            yield
        finally:
            self._input_path = original_input_path
            self._output_path = original_output_path

    def _determine_source_root(self, image_paths: list[Path]) -> Path | None:
        """Return a base directory used to preserve structure on save."""

        if self._input_path is not None:
            return self._input_path
        if not image_paths:
            return None
        try:
            common = os.path.commonpath([path.resolve() for path in image_paths])
        except ValueError:
            return None
        return Path(common)

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

    def _validate_input_selection(
        self,
        *,
        input_path: Path | str | None,
        input_paths: Iterable[Path | str] | None,
        input_arrays: Iterable[np.ndarray] | None,
    ) -> None:
        """Validate that only one input source kind is provided."""

        input_options = [
            input_path is not None,
            input_paths is not None,
            input_arrays is not None,
        ]
        if sum(input_options) > 1:
            msg = "Specify only one of input_path, input_paths, or input_arrays."
            raise ValueError(msg)
