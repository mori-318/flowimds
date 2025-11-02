import os
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np


def read_image(
    path: str,
    flags: int = cv2.IMREAD_COLOR,
    dtype: np.dtype = np.uint8,
) -> np.ndarray | None:
    """Read an image from the given path.

    Supports paths that contain Japanese characters.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray | None: The read image.
    """

    try:
        n = np.fromfile(path, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def write_image(path: str, img: np.ndarray, params: List[Tuple[int, int]] = []) -> bool:
    """Write an image to the given path.

    Args:
        path: Path to the image file.
        img: The image to write.
        params: Parameters to pass to ``cv2.imencode``.

    Returns:
        bool: ``True`` if the image was written successfully, ``False`` otherwise.
    """

    try:
        # If the parent directory does not exist, create it.
        parent_dir = Path(path).parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)

        ext = os.path.splitext(path)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(path, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
