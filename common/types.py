from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PersonBox:
    """YOLO가 찾은 사람 한 명 (B가 만든다)."""
    track_id: int                      # 추적 번호. 같은 사람이면 프레임이 바뀌어도 같음
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2), 원본 프레임 픽셀 좌표
    conf: float                        # 검출 신뢰도


@dataclass
class RecognitionResult:
    """한 프레임의 얼굴 인식 결과 (A가 만든다)."""
    face_found: bool
    person_id: Optional[int] = None    # 기준값 이내로 일치한 등록자. 없으면 None
    name: Optional[str] = None
    distance: Optional[float] = None   # 가장 가까운 등록 얼굴과의 거리 (작을수록 비슷)
    face_image: Optional[np.ndarray] = None   # BGR 얼굴 이미지
    embedding: Optional[np.ndarray] = None    # float32 1차원 특징 벡터


@dataclass
class Decision:
    """다수결로 확정된 판정 (A가 만든다)."""
    category: str                      # "registered" | "unregistered" | "unknown"
    track_id: int
    person_id: Optional[int] = None
    name: Optional[str] = None
    distance: Optional[float] = None
    face_image: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None    # 미등록자 중복 방지에 씀
