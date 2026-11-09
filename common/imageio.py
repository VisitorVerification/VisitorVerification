"""한글 경로에서도 되는 이미지 읽기·쓰기 (B 담당 파일, A가 먼저 쓰려고 만든 기본 구현)."""
import os

import cv2
import numpy as np


def imread_unicode(path: str) -> np.ndarray | None:
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, image: np.ndarray) -> bool:
    """폴더가 없으면 만든다."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(path)
    return True
