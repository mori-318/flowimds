from pathlib import Path
from typing import assert_type

import cv2
import numpy as np
import pytest

from flowimds.pipeline import Pipeline, PipelineResult, ProcessedImage
from flowimds.steps.resize import ResizeStep
from flowimds.utils.image_discovery import collect_image_paths
from flowimds.utils.image_io import write_image


class FailingStep:
    """Test step that always raises."""

    def __init__(self) -> None:
        self.call_count = 0

    def apply(self, image: np.ndarray) -> np.ndarray:  # noqa: ARG002
        self.call_count += 1
        raise RuntimeError("intentional step failure")


@pytest.mark.usefixtures("simple_input_dir")
def test_pipeline_applies_steps_and_generates_results(
    simple_input_dir: Path,
    output_dir: Path,
) -> None:
    """Pipeline should transform all images and save resized copies."""
    target_size = (40, 40)
    pipeline = Pipeline(steps=[ResizeStep(target_size)])

    result = pipeline.run(input_path=simple_input_dir)
    result.save(output_dir)
    assert_type(result, PipelineResult)

    input_images = collect_image_paths(simple_input_dir)
    assert result.processed_count == len(input_images)
    assert result.failed_count == 0

    for mapping in result.output_mappings:
        assert mapping.output_path.exists()
        img = cv2.imread(str(mapping.output_path))
        assert img is not None
        h, w = img.shape[:2]
        assert (w, h) == target_size

    assert result.duration_seconds >= 0


@pytest.mark.usefixtures("simple_input_dir")
def test_pipeline_records_failures_and_continues(
    simple_input_dir: Path,
) -> None:
    """Pipeline records failed files while continuing."""
    failing_step = FailingStep()
    pipeline = Pipeline(steps=[failing_step])

    result = pipeline.run(input_path=simple_input_dir)
    assert_type(result, PipelineResult)

    input_images = collect_image_paths(simple_input_dir)
    assert result.processed_count == 0
    assert result.failed_count == len(input_images)
    assert sorted(Path(p) for p in result.failed_files) == input_images
    assert failing_step.call_count == len(input_images)


def test_pipeline_flattened_outputs_are_unique(
    tmp_path: Path,
    output_dir: Path,
) -> None:
    """Flattened outputs should disambiguate duplicate filenames."""
    input_root = tmp_path / "inputs"
    (input_root / "a").mkdir(parents=True)
    (input_root / "b").mkdir(parents=True)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    write_image(str(input_root / "a" / "duplicate.png"), image)
    write_image(str(input_root / "b" / "duplicate.png"), image)

    pipeline = Pipeline(steps=[ResizeStep((16, 16))])
    result = pipeline.run(input_path=input_root, recursive=True)
    result.save(output_dir)

    output_files = sorted(p.name for p in output_dir.glob("*.png"))
    assert result.processed_count == 2
    assert output_files == ["duplicate.png", "duplicate_no2.png"]


@pytest.mark.usefixtures("recursive_input_dir")
def test_pipeline_honours_recursive_flag(
    recursive_input_dir: Path,
) -> None:
    """Recursive flag controls whether subdirectories are scanned."""
    top_level = collect_image_paths(recursive_input_dir, recursive=False)
    all_images = collect_image_paths(recursive_input_dir, recursive=True)

    non_rec = Pipeline(steps=[]).run(input_path=recursive_input_dir, recursive=False)
    rec = Pipeline(steps=[]).run(input_path=recursive_input_dir, recursive=True)

    assert non_rec.processed_count == len(top_level)
    assert rec.processed_count == len(all_images)


@pytest.mark.usefixtures("simple_input_dir")
def test_pipeline_run_on_paths_processes_explicit_list(
    simple_input_dir: Path,
    output_dir: Path,
) -> None:
    """input_paths should process only the provided file list."""
    target_size = (28, 28)
    image_paths = collect_image_paths(simple_input_dir)
    pipeline = Pipeline(steps=[ResizeStep(target_size)])

    result = pipeline.run(input_paths=image_paths)
    result.save(output_dir)

    assert result.processed_count == len(image_paths)
    assert result.failed_count == 0
    for mapping in result.output_mappings:
        img = cv2.imread(str(mapping.output_path))
        assert img is not None
        h, w = img.shape[:2]
        assert (w, h) == target_size


def test_run_raises_when_input_path_missing() -> None:
    """Raise ValueError when no input is provided."""
    pipeline = Pipeline(steps=[])
    with pytest.raises(ValueError, match="input_path, input_paths, or input_arrays"):
        pipeline.run()


@pytest.mark.usefixtures("simple_input_dir")
def test_run_without_output_directory_returns_in_memory_results(
    simple_input_dir: Path,
) -> None:
    """Keep results in memory when no output path is given."""
    pipeline = Pipeline(steps=[])
    result = pipeline.run(input_path=simple_input_dir)
    assert_type(result, PipelineResult)

    input_images = collect_image_paths(simple_input_dir)
    assert result.processed_count == len(input_images)
    assert result.output_mappings == []
    assert len(result.processed_images) == len(input_images)


@pytest.mark.usefixtures("simple_input_dir")
def test_run_rejects_input_paths_when_input_directory_provided(
    simple_input_dir: Path,
) -> None:
    """Error when input_path and input_paths are provided together."""
    image_paths = collect_image_paths(simple_input_dir)
    pipeline = Pipeline(steps=[])

    with pytest.raises(ValueError, match="Specify only one of"):
        pipeline.run(input_path=simple_input_dir, input_paths=image_paths)


def test_run_with_empty_input_paths_returns_empty_result() -> None:
    """Empty list yields an empty result."""
    result = Pipeline(steps=[]).run(input_paths=[])
    assert result.processed_count == 0
    assert result.failed_count == 0


@pytest.mark.usefixtures("simple_input_dir")
def test_pipeline_run_on_arrays_returns_transformed_images(
    simple_input_dir: Path,
) -> None:
    """input_arrays should return transformed images in memory."""
    image_paths = collect_image_paths(simple_input_dir)
    arrays = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in image_paths]

    result = Pipeline(steps=[ResizeStep((16, 16))]).run(input_arrays=arrays)  # type: ignore
    assert len(result.processed_images) == len(arrays)
    for processed in result.processed_images:
        assert processed.image.shape[:2] == (16, 16)


def test_run_raises_when_multiple_input_sources_provided(
    simple_input_dir: Path,
) -> None:
    """Error when multiple input source kinds are mixed."""
    image_paths = collect_image_paths(simple_input_dir)
    pipeline = Pipeline(steps=[])

    with pytest.raises(ValueError, match="Specify only one of"):
        pipeline.run(
            input_paths=image_paths,
            input_arrays=[np.zeros((4, 4, 3), dtype=np.uint8)],
        )


def test_run_raises_when_input_directory_missing(tmp_path: Path) -> None:
    """FileNotFoundError when input directory is missing."""
    missing_dir = tmp_path / "not_there"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Pipeline(steps=[]).run(input_path=missing_dir)


def test_pipeline_result_save_is_noop_without_processed_images(tmp_path: Path) -> None:
    """save is a no-op when there are no processed images."""
    result = PipelineResult(
        processed_count=0,
        failed_count=0,
        failed_files=[],
        output_mappings=[],
        duration_seconds=0.0,
        settings={
            "input_path": None,
            "output_path": None,
            "recursive": False,
            "worker_count": 1,
            "log_enabled": False,
        },
        processed_images=[],
        source_root=None,
    )
    result.save(tmp_path, preserve_structure=False)
    assert not any(tmp_path.iterdir())


def test_pipeline_result_save_preserves_structure(tmp_path: Path) -> None:
    """When preserve_structure=True, save mirrors the source tree."""
    source_root = tmp_path / "source"
    nested_dir = source_root / "nested"
    nested_dir.mkdir(parents=True)
    sample_path = nested_dir / "image.png"
    write_image(str(sample_path), np.zeros((4, 4, 3), dtype=np.uint8))

    processed = ProcessedImage(
        input_path=sample_path,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    result = PipelineResult(
        processed_count=1,
        failed_count=0,
        failed_files=[],
        output_mappings=[],
        duration_seconds=0.0,
        settings={
            "input_path": str(source_root),
            "output_path": None,
            "recursive": True,
            "worker_count": 1,
            "log_enabled": False,
        },
        processed_images=[processed],
        source_root=source_root,
    )

    destination = tmp_path / "output"
    result.save(destination, preserve_structure=True)
    assert (destination / "nested" / "image.png").exists()
