"""Integration tests for Pipeline working with concrete steps."""

from pathlib import Path

import cv2
import pytest

from flowimds.pipeline import Pipeline
from flowimds.steps import ResizeStep
from flowimds.utils.image_discovery import collect_image_paths


@pytest.mark.usefixtures("simple_input_dir")
def test_pipeline_with_resize_step(simple_input_dir: Path, output_dir: Path) -> None:
    """Ensure ``Pipeline`` and ``ResizeStep`` produce resized outputs end-to-end."""

    target_size = (24, 24)
    pipeline = Pipeline(
        steps=[ResizeStep(target_size)],
        input_path=simple_input_dir,
        output_path=output_dir,
        recursive=False,
        preserve_structure=True,
    )

    result = pipeline.run()

    input_images = collect_image_paths(simple_input_dir)
    output_images = collect_image_paths(output_dir)

    assert result.processed_count == len(input_images)
    assert result.failed_count == 0
    assert not result.failed_files
    assert len(output_images) == len(input_images)

    for output_path in output_images:
        image = cv2.imread(str(output_path))
        assert image is not None
        height, width = image.shape[:2]
        assert (width, height) == target_size
