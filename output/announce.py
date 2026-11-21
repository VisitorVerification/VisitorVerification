"""쿨다운·미등록자 중복 방지 (B, 계획서 5.6절).

should_announce(decision, now)가 True일 때만 안내·문 명령·기록을 한다.
True를 돌려줄 때 시각을 기억한다.

| 판정       | 같은 사람으로 보는 기준                                      | 기간               |
| 등록자     | 같은 person_id                                               | COOLDOWN_SEC       |
| 미등록자   | 얼굴 벡터 거리 ≤ MATCH_THRESHOLD (메모리 + C의 DB)            | UNKNOWN_DEDUP_MIN  |
| 인식 불가  | 같은 track_id                                                | COOLDOWN_SEC       |
"""
import logging

import numpy as np

from common.types import Decision
from config import COOLDOWN_SEC, MATCH_THRESHOLD, UNKNOWN_DEDUP_MIN
from vision.face import embedding_distance

log = logging.getLogger("announce")

_registered: dict[int, float] = {}                 # person_id → 마지막 안내 시각
_unknown: dict[int, float] = {}                    # track_id → 마지막 안내 시각
_unregistered: list[tuple[float, np.ndarray]] = [] # (안내 시각, 얼굴 벡터)

# C의 recent_unknown_embeddings(minutes) → 벡터 목록. C의 함수가 오면 main.py에서 넣는다 (②)
recent_unknown_embeddings = None


def reset() -> None:
    """기억한 안내 기록을 모두 지운다 (l 키, 시험용)."""
    _registered.clear()
    _unknown.clear()
    _unregistered.clear()


def _cleanup(now: float) -> None:
    for memo in (_registered, _unknown):
        for key in [k for k, t in memo.items() if now - t >= COOLDOWN_SEC]:
            del memo[key]
    _unregistered[:] = [(t, e) for t, e in _unregistered if now - t < UNKNOWN_DEDUP_MIN * 60]


def _same_face(embedding: np.ndarray, others) -> bool:
    return any(embedding_distance(embedding, o) <= MATCH_THRESHOLD for o in others)


def _seen_unregistered(embedding: np.ndarray) -> bool:
    """① 메모리의 최근 미등록자, ② C의 DB 중 하나라도 같은 사람이면 True."""
    if _same_face(embedding, (e for _, e in _unregistered)):
        return True
    if recent_unknown_embeddings is not None:
        try:
            if _same_face(embedding, recent_unknown_embeddings(minutes=UNKNOWN_DEDUP_MIN)):
                return True
        except Exception:   # DB 오류여도 메모리 확인만으로 계속 진행
            log.exception("최근 미등록자 벡터를 불러오지 못했습니다")
    return False


def should_announce(decision: Decision, now: float) -> bool:
    """이 판정을 안내·기록해야 하면 True (그리고 시각을 기억), 최근에 안내한 사람이면 False."""
    _cleanup(now)
    if decision.category == "registered":
        if decision.person_id is None:          # 번호가 없으면 비교할 수 없으므로 안내
            return True
        if decision.person_id in _registered:
            return False
        _registered[decision.person_id] = now
        return True

    if decision.category == "unregistered":
        if decision.embedding is None:          # 벡터가 없으면 비교할 수 없으므로 안내
            return True
        embedding = np.asarray(decision.embedding, dtype=np.float32).ravel()
        if _seen_unregistered(embedding):
            return False
        _unregistered.append((now, embedding))
        return True

    # "unknown": 벡터가 없으므로 추적 번호로
    if decision.track_id in _unknown:
        return False
    _unknown[decision.track_id] = now
    return True
