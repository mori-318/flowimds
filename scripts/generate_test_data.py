"""Generate fixture images used by the automated tests.

This script rebuilds the directory structure under ``tests/data`` with
synthetic images. Run it whenever the fixture assets need to be recreated
(e.g., after a clean checkout).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from flowimds.utils.image_io import write_image


TESTS_ROOT = Path(__file__).resolve().parent.parent / "tests"
DATA_ROOT = TESTS_ROOT / "data"


def _ensure_directory(path: Path) -> None:
    """Create ``path`` if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def _rng(seed: int) -> np.random.Generator:
    """Return a NumPy random number generator initialised with ``seed``."""

    return np.random.default_rng(seed)


def _random_shape(
    generator: np.random.Generator,
    *,
    channels: int = 3,
    min_size: int = 32,
    max_size: int = 160,
) -> tuple[int, ...]:
    """Return a random image shape within the provided bounds.

    Args:
        generator: Random number generator used for sampling.
        channels: Number of colour channels to include. ``0`` indicates
            grayscale.
        min_size: Inclusive lower bound for width/height.
        max_size: Exclusive upper bound for width/height.

    Returns:
        Tuple describing the sampled image dimensions.
    """

    height = int(generator.integers(min_size, max_size))
    width = int(generator.integers(min_size, max_size))
    if channels <= 0:
        return (height, width)
    return (height, width, channels)


def _random_image(shape: tuple[int, ...], generator: np.random.Generator) -> np.ndarray:
    """Return pseudo-random image data for a given ``shape``."""

    return generator.integers(0, 256, size=shape, dtype=np.uint8)


def _generate_fixed_colour(
    shape: tuple[int, int],
    *,
    colour: Iterable[int],
) -> np.ndarray:
    """Return an RGB image filled with ``colour``.

    Args:
        shape: Image height and width.
        colour: Iterable describing the (B, G, R) values.

    Returns:
        Image filled with the requested colour.
    """

    array = np.zeros((*shape, 3), dtype=np.uint8)
    array[...] = np.array(tuple(colour), dtype=np.uint8)
    return array


def _generate_simple_dataset() -> None:
    """Populate ``tests/data/simple`` with flat input fixtures."""

    input_dir = DATA_ROOT / "simple" / "input"
    _ensure_directory(input_dir)

    filenames: Sequence[str] = [
        "image_01.png",
        "image_02.png",
        "image_03.png",
        "image_04.png",
        "image_05.png",
        "image_06.png",
        "image_07.png",
        "image_08.png",
        "image_09.png",
        "image_10.png",
    ]

    generator = _rng(100)
    for filename in filenames:
        shape = _random_shape(generator)
        image = _random_image(shape, generator)
        write_image(str(input_dir / filename), image)

    sample_blue = _generate_fixed_colour((24, 24), colour=(255, 0, 0))
    write_image(str(input_dir / "sample_blue.png"), sample_blue)

    gray_generator = _rng(200)
    sample_gray = _random_image((32, 32), gray_generator)
    write_image(str(input_dir / "sample_gray.jpg"), sample_gray)


def _generate_recursive_dataset() -> None:
    """Populate ``tests/data/recursive`` with nested input fixtures."""

    base = DATA_ROOT / "recursive" / "input"
    level1 = base / "level1"
    level2 = level1 / "level2"

    for path in (base, level1, level2):
        _ensure_directory(path)

    generator = _rng(500)

    definitions: Sequence[tuple[Path, str]] = [
        (base, "image_01.png"),
        (base, "image_02.png"),
        (base, "image_03.png"),
        (base, "image_04.png"),
        (base, "image_05.png"),
        (base, "image_06.png"),
        (base, "image_07.png"),
        (base, "image_08.png"),
        (base, "image_09.png"),
        (base, "image_10.png"),
        (level1, "image_nested_01.png"),
        (level1, "image_nested_02.png"),
        (level2, "image_deep_01.png"),
    ]

    for directory, filename in definitions:
        shape = _random_shape(generator)
        image = _random_image(shape, generator)
        write_image(str(directory / filename), image)

    for index in range(2):
        nested = _random_image(_random_shape(generator), generator)
        write_image(str(level2 / f"sub_{index}.png"), nested)


def _generate_others_dataset() -> None:
    """Populate ``tests/data/others`` with miscellaneous assets."""

    others_dir = DATA_ROOT / "others"
    _ensure_directory(others_dir)

    unicode_image = _generate_fixed_colour((32, 32), colour=(0, 255, 255))
    write_image(str(others_dir / "日本語含むパス.png"), unicode_image)

    ascii_image = _generate_fixed_colour((32, 32), colour=(0, 0, 255))
    write_image(str(others_dir / "no_jp.png"), ascii_image)


def main() -> None:
    """Generate all fixture datasets under ``tests/data``."""

    _generate_simple_dataset()
    _generate_recursive_dataset()
    _generate_others_dataset()


if __name__ == "__main__":
    main()
