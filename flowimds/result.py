"""Pipeline result container with save functionality."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
from threading import Lock
from typing import cast

from tqdm import tqdm

from flowimds.models import OutputMapping, PipelineSettings, ProcessedImage
from flowimds.utils.image_io import write_image


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

    def save(
        self,
        output_path: Path | str,
        preserve_structure: bool = False,
    ) -> None:
        """Persist processed images to ``output_path``.

        Args:
            output_path: Destination directory where processed images are written.
            preserve_structure: Whether to mirror the input directory structure.
        """

        if not self.processed_images:
            return

        destination_root = Path(output_path)
        destination_root.mkdir(parents=True, exist_ok=True)

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
