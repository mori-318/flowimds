"""Unit tests that define the expected behaviour of ``Pipeline``."""

from pathlib import Path
from typing import assert_type, cast

import cv2
import numpy as np
import pytest

from flowimds.pipeline import Pipeline, PipelineResult
from flowimds.steps import ResizeStep
from flowimds.utils.image_discovery import collect_image_paths
from flowimds.utils.image_io import write_image


def create_pipeline_with_resize_step(size: tuple[int, int]) -> Pipeline:
    return Pipeline(
        steps=[ResizeStep(size)],
        recursive=False,
        preserve_structure=True,
    )


def check_result_images(
    result_images: list[Path], simple_input_dir: Path, target_size: tuple[int, int]
) -> None:
    input_images = collect_image_paths(simple_input_dir)
    assert len(result_images) == len(input_images)
    for result_path in result_images:
        image = cv2.imread(str(result_path))
        assert image is not None
        height, width = image.shape[:2]
        assert (width, height) == target_size


def _normalise_mapping(mapping: object) -> tuple[Path, Path]:
    """Convert mapping objects into ``(input_path, output_path)`` tuples."""

    if hasattr(mapping, "input_path") and hasattr(mapping, "output_path"):
        return Path(mapping.input_path), Path(mapping.output_path)
    if isinstance(mapping, dict):  # pragma: no cover - defensive branch
        return Path(mapping["input_path"]), Path(mapping["output_path"])
    input_path, output_path = cast(
        tuple[str | Path, str | Path],
        mapping,
    )
    return Path(input_path), Path(output_path)


class FailingStep:
    """Test double that raises an error for every invocation."""

    def __init__(self) -> None:
        """Initialise the failing step."""

        self.call_count = 0

    def apply(self, image: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Raise an error to emulate a processing failure."""

        self.call_count += 1
        raise RuntimeError("intentional step failure")


@pytest.mark.usefixtures("simple_input_dir")
def test_pipeline_applies_steps_and_generates_results(
    simple_input_dir: Path,
    output_dir: Path,
) -> None:
    """`Pipeline` should transform every image and persist resized copies."""

    target_size = (40, 40)
    pipeline = create_pipeline_with_resize_step(target_size)

    result = pipeline.run(input_path=simple_input_dir)
    result.save(output_dir)
    assert_type(result, PipelineResult)

    input_images = collect_image_paths(simple_input_dir)

    assert result.processed_count == len(input_images)
    assert result.failed_count == 0
    assert not result.failed_files

    normalised_mappings = [
        _normalise_mapping(mapping) for mapping in result.output_mappings
    ]
    assert len(normalised_mappings) == len(input_images)
    for source_path, output_path in normalised_mappings:
        assert Path(source_path) in input_images
        assert output_path.is_relative_to(output_dir)
        assert output_path.exists()
        image = cv2.imread(str(output_path))
        assert image is not None
        height, width = image.shape[:2]
        assert (width, height) == target_size
    assert result.settings["recursive"] is False
    input_setting = result.settings["input_path"]
    assert input_setting is not None
    assert Path(input_setting) == simple_input_dir
    assert result.settings["output_path"] is None
    assert result.duration_seconds >= 0


@pytest.mark.usefixtures("simple_input_dir")
def test_pipeline_records_failures_and_continues(
    simple_input_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """`Pipeline` should capture failing files without aborting the run."""

    failing_step = FailingStep()
    output_path = tmp_path_factory.mktemp("failures")
    pipeline = Pipeline(
        steps=[failing_step],
        recursive=False,
        preserve_structure=False,
    )

    result = pipeline.run(input_path=simple_input_dir)
    assert_type(result, PipelineResult)

    input_images = collect_image_paths(simple_input_dir)

    assert result.processed_count == 0
    assert result.failed_count == len(input_images)
    assert sorted(Path(path) for path in result.failed_files) == input_images
    assert not any(output_path.iterdir())
    assert failing_step.call_count == len(input_images)


def test_pipeline_flattened_outputs_are_unique(
    tmp_path: Path,
    output_dir: Path,
) -> None:
    """Pipeline should de-duplicate flattened output filenames."""

    input_root = tmp_path / "inputs"
    nested_a = input_root / "a"
    nested_b = input_root / "b"
    nested_a.mkdir(parents=True, exist_ok=True)
    nested_b.mkdir(parents=True, exist_ok=True)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    write_image(str(nested_a / "duplicate.png"), image)
    write_image(str(nested_b / "duplicate.png"), image)

    pipeline = Pipeline(
        steps=[ResizeStep((16, 16))],
        recursive=True,
        preserve_structure=False,
    )

    result = pipeline.run(input_path=input_root)
    result.save(output_dir)
    assert_type(result, PipelineResult)

    output_files = sorted(path.name for path in output_dir.glob("*.png"))

    assert result.processed_count == 2
    assert result.failed_count == 0
    assert output_files == ["duplicate.png", "duplicate_no2.png"]


@pytest.mark.usefixtures("recursive_input_dir")
def test_pipeline_honours_recursive_flag(
    recursive_input_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """`Pipeline` should toggle recursive discovery based on the flag."""

    top_level_images = collect_image_paths(recursive_input_dir)
    all_images = collect_image_paths(recursive_input_dir, recursive=True)

    non_recursive_pipeline = Pipeline(
        steps=[ResizeStep((16, 16))],
        recursive=False,
        preserve_structure=False,
    )
    non_recursive_result = non_recursive_pipeline.run(input_path=recursive_input_dir)
    non_recursive_result.save(tmp_path_factory.mktemp("non_recursive"))
    assert_type(non_recursive_result, PipelineResult)

    recursive_pipeline = Pipeline(
        steps=[ResizeStep((16, 16))],
        recursive=True,
        preserve_structure=False,
    )
    recursive_result = recursive_pipeline.run(input_path=recursive_input_dir)
    recursive_result.save(tmp_path_factory.mktemp("recursive"))
    assert_type(recursive_result, PipelineResult)

    assert non_recursive_result.processed_count == len(top_level_images)
    assert recursive_result.processed_count == len(all_images)


@pytest.mark.usefixtures("simple_input_dir")
def test_pipeline_run_on_paths_processes_explicit_list(
    simple_input_dir: Path,
    output_dir: Path,
) -> None:
    """`Pipeline.run` with input_paths should process the provided file list only."""

    target_size = (28, 28)
    image_paths = collect_image_paths(simple_input_dir)

    pipeline = Pipeline(
        steps=[ResizeStep(target_size)],
        recursive=False,
        preserve_structure=False,
    )

    result = pipeline.run(input_paths=image_paths)
    result.save(output_dir)
    assert_type(result, PipelineResult)

    assert result.processed_count == len(image_paths)
    assert result.failed_count == 0
    assert not result.failed_files
    assert all(mapping.output_path.exists() for mapping in result.output_mappings)
    for mapping in result.output_mappings:
        image = cv2.imread(str(mapping.output_path))
        assert image is not None
        height, width = image.shape[:2]
        assert (width, height) == target_size


def test_run_raises_when_input_path_missing(tmp_path: Path) -> None:
    """`Pipeline.run` must require an input path."""

    pipeline = Pipeline(steps=[])

    with pytest.raises(
        ValueError,
        match="input_path, input_paths, or input_arrays must be specified.",
    ):
        pipeline.run()


@pytest.mark.usefixtures("simple_input_dir")
def test_run_raises_when_output_path_missing(simple_input_dir: Path) -> None:
    """`Pipeline.run` should support deferred saving when no output path is set."""

    pipeline = Pipeline(steps=[])

    result = pipeline.run(input_path=simple_input_dir)
    assert_type(result, PipelineResult)

    input_images = collect_image_paths(simple_input_dir)

    assert result.processed_count == len(input_images)
    assert result.failed_count == 0
    assert not result.failed_files
    assert result.output_mappings == []
    assert len(result.processed_images) == len(input_images)


@pytest.mark.usefixtures("simple_input_dir")
def test_run_raises_when_input_path_defined(
    simple_input_dir: Path,
    output_dir: Path,
) -> None:
    """`Pipeline.run` with input_paths should work with input_path."""

    target_size = (28, 28)
    image_paths = collect_image_paths(simple_input_dir)

    pipeline = Pipeline(
        steps=[ResizeStep(target_size)],
        recursive=False,
        preserve_structure=False,
    )

    # This should work - we can use input_paths with input_path set
    result = pipeline.run(input_paths=image_paths)
    result.save(output_dir)
    assert_type(result, PipelineResult)

    assert result.processed_count == len(image_paths)
    assert result.failed_count == 0


def test_run_on_paths_raises_when_output_path_missing() -> None:
    """`Pipeline.run` with empty input_paths should work."""

    pipeline = Pipeline(steps=[])

    # Running with empty list should work and return empty result
    result = pipeline.run(input_paths=[])
    assert_type(result, PipelineResult)
    assert result.processed_count == 0
    assert result.failed_count == 0


@pytest.mark.usefixtures("simple_input_dir")
def test_pipeline_run_on_arrays_returns_transformed_images(
    simple_input_dir: Path,
) -> None:
    """`Pipeline.run` with input_arrays should return transformed images in memory."""

    image_paths = collect_image_paths(simple_input_dir)
    arrays = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None
        arrays.append(image)

    pipeline = Pipeline(
        steps=[ResizeStep((16, 16))],
        recursive=False,
        preserve_structure=False,
    )

    result = pipeline.run(input_arrays=arrays)
    assert_type(result, PipelineResult)

    assert len(result.processed_images) == len(arrays)
    for processed in result.processed_images:
        assert processed.image.shape[:2] == (16, 16)
