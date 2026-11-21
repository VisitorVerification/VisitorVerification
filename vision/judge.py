"""다수결 판정 (A 담당, 인수인계서 5.5).

추적 번호(track_id)마다 최근 인식 결과를 모아
등록자 / 미등록자 / 인식 불가 를 확정한다.

B의 메인 루프 사용법 (4.4)
    decision = vote(person.track_id, result, now)    # 얼굴 인식을 한 프레임에서만
    if needs_face_hint(person.track_id, now): ...    # 매 프레임 불러도 됨

같은 사람에게 반복해서 안내하지 않는 일(쿨다운)은 B가 맡는다.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from common.types import Decision, RecognitionResult
from config import FACE_HINT_SEC, UNKNOWN_TIMEOUT_SEC, VOTE_MIN, VOTE_WINDOW

STALE_SEC = 10          # 이 시간(초) 넘게 갱신되지 않은 추적 번호는 지운다 (5.5)
UNREG = "UNREG"         # 최근 결과 목록에서 미등록자를 나타내는 값


@dataclass
class _Vote:
    label: object                 # 등록자면 person_id(int), 미등록이면 UNREG
    name: Optional[str]
    distance: Optional[float]


@dataclass
class _TrackState:
    first_seen: float                              # 처음 vote()가 불린 시각 (판정 후에는 다시 잼)
    last_update: float                             # 마지막 vote() 시각
    last_face: Optional[float] = None              # 마지막으로 얼굴을 찾은 시각
    votes: deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    face_image: Optional[np.ndarray] = None        # 최근 얼굴 정보
    embedding: Optional[np.ndarray] = None
    distance: Optional[float] = None


_tracks: dict[int, _TrackState] = {}


# ──────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────
def _cleanup(now: float) -> None:
    """STALE_SEC 넘게 갱신되지 않은 추적 번호를 지운다."""
    for tid in [t for t, s in _tracks.items() if now - s.last_update > STALE_SEC]:
        del _tracks[tid]


def _no_face_seconds(state: _TrackState, now: float) -> float:
    """얼굴을 못 찾은 지 몇 초인지. 한 번도 못 찾았으면 처음 본 시각부터 잰다."""
    base = state.last_face if state.last_face is not None else state.first_seen
    return now - base


def _reset(state: _TrackState, now: float) -> None:
    """판정을 돌려준 뒤 최근 결과 목록과 시각 기준을 초기화한다."""
    state.votes.clear()
    state.first_seen = now
    state.last_face = None
    state.face_image = state.embedding = state.distance = None


def _reset_all() -> None:
    """모든 추적 상태를 지운다 (테스트용)."""
    _tracks.clear()


# ──────────────────────────────────────────────
# 공통 명세 함수 (3.5)
# ──────────────────────────────────────────────
def vote(track_id: int, result: RecognitionResult, now: float) -> Decision | None:
    """한 프레임의 인식 결과를 넣고, 확정되면 Decision을, 아니면 None을 돌려준다."""
    _cleanup(now)
    state = _tracks.get(track_id)
    if state is None:
        state = _tracks[track_id] = _TrackState(first_seen=now, last_update=now)
    state.last_update = now

    # 얼굴을 찾은 결과만 최근 결과 목록에 넣는다
    if result.face_found:
        state.last_face = now
        label = result.person_id if result.person_id is not None else UNREG
        state.votes.append(_Vote(label, result.name, result.distance))
        state.face_image = result.face_image
        state.embedding = result.embedding
        state.distance = result.distance

    counts = Counter(v.label for v in state.votes)

    # 1) 등록자: 같은 person_id가 VOTE_MIN개 이상
    for label, count in counts.most_common():
        if label != UNREG and count >= VOTE_MIN:
            matched = [v for v in state.votes if v.label == label]
            distances = [v.distance for v in matched if v.distance is not None]
            decision = Decision(category="registered", track_id=track_id,
                                person_id=int(label), name=matched[-1].name,
                                distance=min(distances) if distances else None,
                                face_image=state.face_image, embedding=state.embedding)
            _reset(state, now)
            return decision

    # 2) 미등록자: UNREG가 VOTE_MIN개 이상
    if counts.get(UNREG, 0) >= VOTE_MIN:
        decision = Decision(category="unregistered", track_id=track_id,
                            distance=state.distance,
                            face_image=state.face_image, embedding=state.embedding)
        _reset(state, now)
        return decision

    # 3) 인식 불가: 얼굴을 못 찾은 지 UNKNOWN_TIMEOUT_SEC 이상
    if _no_face_seconds(state, now) >= UNKNOWN_TIMEOUT_SEC:
        decision = Decision(category="unknown", track_id=track_id)
        _reset(state, now)
        return decision

    # 4) 보류
    return None


def needs_face_hint(track_id: int, now: float) -> bool:
    """얼굴을 못 찾은 지 FACE_HINT_SEC 이상, UNKNOWN_TIMEOUT_SEC 미만이면 True."""
    state = _tracks.get(track_id)
    if state is None or now - state.last_update > STALE_SEC:
        return False
    elapsed = _no_face_seconds(state, now)
    return FACE_HINT_SEC <= elapsed < UNKNOWN_TIMEOUT_SEC
