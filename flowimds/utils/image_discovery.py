"""Helper functions for discovering image files within directories."""

from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES: set[str] = {".png", ".jpg", ".jpeg"}


def collect_image_paths(
    root: Path,
    *,
    recursive: bool = False,
    suffixes: Iterable[str] | None = None,
) -> list[Path]:
    """Return sorted image file paths under ``root``.

    Args:
        root: Path to the directory to search.
        recursive: Whether to traverse subdirectories recursively.
        suffixes: Optional iterable of file suffixes to include.

    Returns:
        Sorted list of image paths that match the provided suffixes.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """

    if not root.exists():
        msg = f"Input path '{root}' does not exist."
        raise FileNotFoundError(msg)

    suffix_set = {suffix.lower() for suffix in (suffixes or IMAGE_SUFFIXES)}

    iterator = root.rglob("*") if recursive else root.glob("*")
    image_paths = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in suffix_set
    ]
    return sorted(image_paths, key=lambda path: path.as_posix())


__all__ = ["collect_image_paths", "IMAGE_SUFFIXES"]
