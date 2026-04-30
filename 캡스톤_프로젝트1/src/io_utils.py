"""
io_utils.py
===========
한국어 경로 등 비ASCII 경로에서 cv2.imread/imwrite가 실패하는 문제 우회.
numpy.fromfile + cv2.imdecode 방식 사용.
"""

from pathlib import Path
import numpy as np
import cv2


def imread_unicode(path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """비ASCII 경로 안전 imread."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path, image: np.ndarray, ext: str | None = None) -> bool:
    """비ASCII 경로 안전 imwrite."""
    p = Path(path)
    ext = ext or p.suffix or ".png"
    if not ext.startswith("."):
        ext = "." + ext
    try:
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        buf.tofile(str(p))
        return True
    except Exception:
        return False
