"""Integration tests for Pipeline working with concrete steps."""

from pathlib import Path
from typing import assert_type

import cv2
import pytest

from flowimds.pipeline import Pipeline, PipelineResult
from flowimds.steps import ResizeStep
from flowimds.utils.image_discovery import collect_image_paths

def create_pipeline_with_resize_step(size: tuple[int, int]) -> Pipeline:
    return Pipeline(
        steps=[ResizeStep(size)],
        recursive=False,
        preserve_structure=True,
    )

def check_result_images(result_images: list[Path], simple_input_dir: Path, target_size: tuple[int, int]) -> None:
    input_images = collect_image_paths(simple_input_dir)
    assert len(result_images) == len(input_images)
    for result_path in result_images:
        image = cv2.imread(str(result_path))
        assert image is not None
        height, width = image.shape[:2]
        assert (width, height) == target_size

@pytest.mark.usefixtures("simple_input_dir")
def test_pipeline_run_and_save(simple_input_dir: Path, output_dir: Path) -> None:
    """Pipeline should run first, then persist via ``PipelineResult.save``."""

    # define pipeline
    target_size = (24, 24)
    pipeline = create_pipeline_with_resize_step(target_size)

    # run pipeline (no output path configured yet)
    result = pipeline.run(input_path=simple_input_dir)
    assert_type(result, PipelineResult)

    # verify run outcome before any persistence
    input_images = collect_image_paths(simple_input_dir)
    assert result.processed_count == len(input_images)
    assert result.failed_count == 0
    assert not result.failed_files

    # save results after inspecting the run statistics
    result.save(output_dir)

    # compare results saved on disk
    output_images = collect_image_paths(output_dir)
    check_result_images(output_images, simple_input_dir, target_size)


def test_pipeline_run_on_paths_and_save(simple_input_dir: Path, output_dir: Path) -> None:
    """Pipeline should run first, then persist via ``PipelineResult.save``."""
    input_paths = collect_image_paths(simple_input_dir)

    # define pipeline
    target_size = (24, 24)
    pipeline = create_pipeline_with_resize_step(target_size)

    # run pipeline (no output path configured yet)
    result = pipeline.run(input_paths=input_paths)
    assert_type(result, PipelineResult)

    # save results after inspecting the run statistics
    result.save(output_dir)

    # compare results saved on disk
    output_images = collect_image_paths(output_dir)
    check_result_images(output_images, simple_input_dir, target_size)


def test_pipeline_run_on_arrays_and_save(simple_input_dir: Path, output_dir: Path) -> None:
    input_paths = collect_image_paths(simple_input_dir)
    input_image_arrays = [cv2.imread(str(path)) for path in input_paths]

    # define pipeline
    target_size = (24, 24)
    pipeline = create_pipeline_with_resize_step(target_size)

    # run pipeline (no output path configured yet)
    result = pipeline.run(input_arrays=input_image_arrays)
    assert_type(result, PipelineResult)

    # verify run outcome before any persistence
    assert result.processed_count == len(input_image_arrays)
    assert result.failed_count == 0
    assert not result.failed_files

    # save results after inspecting the run statistics
    result.save(output_dir)

    # compare results saved on disk
    output_images = collect_image_paths(output_dir)
    check_result_images(output_images, simple_input_dir, target_size)

