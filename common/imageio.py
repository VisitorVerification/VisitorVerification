import os

import cv2
import numpy as np


def imread_unicode(path: str) -> np.ndarray | None:
    """한글 경로에서도 되는 이미지 읽기. 실패하면 None을 돌려준다."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def imwrite_unicode(path: str, image: np.ndarray) -> bool:
    """한글 경로에서도 되는 이미지 쓰기. 폴더가 없으면 만든다. 성공 여부를 돌려준다."""
    ext = os.path.splitext(path)[1]
    if not ext:
        return False
    try:
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        buf.tofile(path)
        return True
    except (OSError, cv2.error):
        return False
