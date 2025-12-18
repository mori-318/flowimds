"""Tests for flowimds.utils helper functions."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from flowimds.utils.image_io import read_image, write_image


@pytest.mark.parametrize("fixture_name", ["jp_filename", "no_jp_filename"])
def test_read_image_handles_various_paths(
    request: pytest.FixtureRequest,
    fixture_name: str,
) -> None:
    """Ensure ``read_image`` can load images with Unicode characters in paths."""

    sample_path: Path = request.getfixturevalue(fixture_name)
    actual = read_image(str(sample_path))
    assert actual is not None
    assert actual.dtype == np.uint8


@pytest.mark.parametrize("fixture_name", ["jp_filename", "no_jp_filename"])
def test_write_image_handles_unicode_paths(
    request: pytest.FixtureRequest,
    fixture_name: str,
) -> None:
    """Verify ``write_image`` can persist images to Unicode file paths."""

    sample_path: Path = request.getfixturevalue(fixture_name)
    image = read_image(str(sample_path))
    assert image is not None
    destination = request.getfixturevalue("output_dir") / "出力/結果_日本語ファイル.png"

    succeeded = write_image(str(destination), image)

    assert succeeded
    assert destination.exists()
    saved = cv2.imread(str(destination), cv2.IMREAD_COLOR)
    assert saved is not None
    assert saved.shape == image.shape
    assert np.array_equal(saved, image)
