"""Unit tests that define the expected behaviour of ``Pipeline``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import pytest

from flowimds.pipeline import Pipeline, PipelineResult
from flowimds.steps import ResizeStep

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _collect_images(directory: Path, recursive: bool) -> list[Path]:
    """Return a sorted list of image files under ``directory``."""

    iterator: Iterable[Path]
    if recursive:
        iterator = directory.rglob("*")
    else:
        iterator = directory.glob("*")
    paths = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(paths)


def _normalise_mapping(mapping: object) -> tuple[Path, Path]:
    """Convert mapping objects into ``(input_path, output_path)`` tuples."""

    if hasattr(mapping, "input_path") and hasattr(mapping, "output_path"):
        return Path(mapping.input_path), Path(mapping.output_path)
    if isinstance(mapping, dict):  # pragma: no cover - defensive branch
        return Path(mapping["input_path"]), Path(mapping["output_path"])
    input_path, output_path = mapping  # type: ignore[misc]
    return Path(input_path), Path(output_path)


@dataclass
class RecordingStep:
    """Test double that records how many images it processed."""

    transform: Callable[[np.ndarray], np.ndarray]
    call_count: int = 0

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the configured transform and count the invocation."""

        self.call_count += 1
        return self.transform(image)


class FailingStep:
    """Test double that raises an error for every invocation."""

    def __init__(self) -> None:
        """Initialise the failing step."""

        self.call_count = 0

    def apply(self, image: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Raise an error to emulate a processing failure."""

        self.call_count += 1
        raise RuntimeError("intentional step failure")


def test_pipeline_applies_steps_and_generates_results(
    simple_input_dir: Path,
    output_dir: Path,
) -> None:
    """`Pipeline` should transform every image and persist resized copies."""

    resize_step = RecordingStep(
        transform=lambda image: cv2.resize(image, (40, 40)),
    )
    pipeline = Pipeline(
        input_path=simple_input_dir,
        output_path=output_dir,
        steps=[resize_step],
        recursive=False,
        preserve_structure=True,
    )

    result = pipeline.run()

    input_images = _collect_images(simple_input_dir, recursive=False)

    assert isinstance(result, PipelineResult)
    assert result.processed_count == len(input_images)
    assert result.failed_count == 0
    assert not result.failed_files
    assert resize_step.call_count == len(input_images)

    normalised_mappings = [
        _normalise_mapping(mapping) for mapping in result.output_mappings
    ]
    assert len(normalised_mappings) == len(input_images)
    for source_path, output_path in normalised_mappings:
        assert Path(source_path) in input_images
        assert output_path.is_relative_to(output_dir)
        assert output_path.exists()
        height, width = cv2.imread(str(output_path)).shape[:2]
        assert (width, height) == (40, 40)
    assert result.settings["recursive"] is False
    assert Path(result.settings["input_path"]) == simple_input_dir
    assert Path(result.settings["output_path"]) == output_dir
    assert result.duration_seconds >= 0


def test_pipeline_records_failures_and_continues(
    simple_input_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """`Pipeline` should capture failing files without aborting the run."""

    failing_step = FailingStep()
    output_path = tmp_path_factory.mktemp("failures")
    pipeline = Pipeline(
        input_path=simple_input_dir,
        output_path=output_path,
        steps=[failing_step],
        recursive=False,
        preserve_structure=False,
    )

    result = pipeline.run()

    input_images = _collect_images(simple_input_dir, recursive=False)

    assert isinstance(result, PipelineResult)
    assert result.processed_count == 0
    assert result.failed_count == len(input_images)
    assert sorted(Path(path) for path in result.failed_files) == input_images
    assert not any(output_path.iterdir())
    assert failing_step.call_count == len(input_images)


def test_pipeline_honours_recursive_flag(
    recursive_input_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """`Pipeline` should toggle recursive discovery based on the flag."""

    top_level_images = _collect_images(recursive_input_dir, recursive=False)
    all_images = _collect_images(recursive_input_dir, recursive=True)

    non_recursive_step = RecordingStep(transform=lambda image: image)
    non_recursive_pipeline = Pipeline(
        input_path=recursive_input_dir,
        output_path=tmp_path_factory.mktemp("non_recursive"),
        steps=[non_recursive_step],
        recursive=False,
        preserve_structure=False,
    )
    non_recursive_result = non_recursive_pipeline.run()

    recursive_step = RecordingStep(transform=lambda image: image)
    recursive_pipeline = Pipeline(
        input_path=recursive_input_dir,
        output_path=tmp_path_factory.mktemp("recursive"),
        steps=[recursive_step],
        recursive=True,
        preserve_structure=False,
    )
    recursive_result = recursive_pipeline.run()

    assert non_recursive_result.processed_count == len(top_level_images)
    assert recursive_result.processed_count == len(all_images)
    assert non_recursive_step.call_count == len(top_level_images)
    assert recursive_step.call_count == len(all_images)


def test_pipeline_run_on_paths_processes_explicit_list(
    simple_input_dir: Path,
    output_dir: Path,
) -> None:
    """`Pipeline.run_on_paths` should process the provided file list only."""

    target_size = (28, 28)
    image_paths = _collect_images(simple_input_dir, recursive=False)

    pipeline = Pipeline(
        input_path=simple_input_dir,
        output_path=output_dir,
        steps=[ResizeStep(target_size)],
        recursive=False,
        preserve_structure=False,
    )

    result = pipeline.run_on_paths(image_paths)

    assert result.processed_count == len(image_paths)
    assert result.failed_count == 0
    assert not result.failed_files
    assert all(mapping.output_path.exists() for mapping in result.output_mappings)
    for mapping in result.output_mappings:
        image = cv2.imread(str(mapping.output_path))
        assert image is not None
        height, width = image.shape[:2]
        assert (width, height) == target_size


def test_pipeline_run_on_arrays_returns_transformed_images(
    simple_input_dir: Path,
) -> None:
    """`Pipeline.run_on_arrays` should return transformed images in memory."""

    image_paths = _collect_images(simple_input_dir, recursive=False)
    arrays = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in image_paths]

    pipeline = Pipeline(
        input_path=simple_input_dir,
        output_path=simple_input_dir,
        steps=[ResizeStep((16, 16))],
        recursive=False,
        preserve_structure=False,
    )

    transformed_images = pipeline.run_on_arrays(arrays)

    assert len(transformed_images) == len(arrays)
    for transformed in transformed_images:
        assert transformed.shape[:2] == (16, 16)
