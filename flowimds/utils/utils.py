import os
import cv2
import numpy as np

def read_image(path: str) -> np.ndarray:
    """ Read an image from the given path.
    Supports paths that contain Japanese characters.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: The read image.
    """
    return cv2.imdecode(
        np.fromfile(path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

def write_image(path: str, img: np.ndarray) -> bool:
    """Write an image to the given path.

    Args:
        path: Path to the image file.
        img: The image to write.

    Returns:
        bool: ``True`` if the image was written successfully, ``False`` otherwise.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    path = path if os.path.splitext(path)[1] else path + ext
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:
        return False
    return True
