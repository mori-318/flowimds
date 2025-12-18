"""Image I/O helper functions."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt


def read_image(
    path: str,
    flags: int = cv2.IMREAD_COLOR,
    dtype: npt.DTypeLike = np.uint8,
) -> np.ndarray | None:
    """Read an image from the given path.

    Supports paths that contain Japanese characters.

    Args:
        path: Path to the image file.
        flags: OpenCV flags used to control the reading mode.
        dtype: Data type for the buffer read from disk.

    Returns:
        The read image, or ``None`` if reading failed.
    """

    try:
        buffer = np.fromfile(path, dtype)
        image = cv2.imdecode(buffer, flags)
        return image
    except Exception as exc:  # pragma: no cover - defensive
        print(exc)
        return None


def write_image(
    path: str,
    image: np.ndarray,
    params: Sequence[int] | None = None,
) -> bool:
    """Write an image to the given path.

    Args:
        path: Path to the image file.
        image: The image to write.
        params: Parameters to pass to ``cv2.imencode``.

    Returns:
        ``True`` if the image was written successfully, ``False`` otherwise.
    """

    parameters = list(params or [])

    try:
        parent = Path(path).parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        extension = Path(path).suffix
        success, encoded = cv2.imencode(extension, image, parameters)
        if not success:
            return False

        with Path(path).open(mode="w+b") as destination:
            encoded.tofile(destination)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        print(exc)
        return False
