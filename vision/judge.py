"""다수결 판정 (A 담당, 5.5).

추적 번호(track_id)마다 최근 VOTE_WINDOW번의 얼굴 인식 결과를 모아
같은 결과가 VOTE_MIN번 이상이면 확정한다.
"""
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

import config
from common.types import Decision, RecognitionResult

UNREG = "UNREG"


@dataclass
class _TrackState:
    first_seen: float                          # 처음 본 시각 (판정 후에는 판정 시각으로 초기화)
    last_update: float
    last_face: float | None = None             # 마지막으로 얼굴을 찾은 시각
    results: deque = field(default_factory=lambda: deque(maxlen=config.VOTE_WINDOW))
    # results 항목: (label, distance, name)  label = person_id 또는 "UNREG"
    face_image: np.ndarray | None = None
    embedding: np.ndarray | None = None
    distance: float | None = None

    def reset(self, now: float) -> None:
        self.results.clear()
        self.first_seen = now
        self.last_face = None
        self.face_image = None
        self.embedding = None
        self.distance = None


_tracks: dict[int, _TrackState] = {}


def _purge(now: float) -> None:
    stale = [tid for tid, s in _tracks.items() if now - s.last_update > config.TRACK_EXPIRE_SEC]
    for tid in stale:
        del _tracks[tid]


def _no_face_since(state: _TrackState) -> float:
    return state.last_face if state.last_face is not None else state.first_seen


def vote(track_id: int, result: RecognitionResult, now: float) -> Decision | None:
    """얼굴 인식 결과 하나를 넣고, 확정되면 Decision을 돌려준다. 보류면 None."""
    _purge(now)
    state = _tracks.get(track_id)
    if state is None:
        state = _TrackState(first_seen=now, last_update=now)
        _tracks[track_id] = state
    state.last_update = now

    if result.face_found:
        label = result.person_id if result.person_id is not None else UNREG
        state.results.append((label, result.distance, result.name))
        state.last_face = now
        state.face_image = result.face_image
        state.embedding = result.embedding
        state.distance = result.distance

    counts = Counter(label for label, _, _ in state.results)

    # 1) 등록자
    for label, n in counts.most_common():
        if label != UNREG and n >= config.VOTE_MIN:
            hits = [(d, name) for lb, d, name in state.results if lb == label]
            dists = [d for d, _ in hits if d is not None]
            decision = Decision(
                category="registered",
                track_id=track_id,
                person_id=int(label),
                name=hits[-1][1],
                distance=min(dists) if dists else None,
                face_image=state.face_image,
                embedding=state.embedding,
            )
            state.reset(now)
            return decision

    # 2) 미등록자
    if counts.get(UNREG, 0) >= config.VOTE_MIN:
        decision = Decision(
            category="unregistered",
            track_id=track_id,
            distance=state.distance,
            face_image=state.face_image,
            embedding=state.embedding,
        )
        state.reset(now)
        return decision

    # 3) 인식 불가
    if now - _no_face_since(state) >= config.UNKNOWN_TIMEOUT_SEC:
        decision = Decision(category="unknown", track_id=track_id)
        state.reset(now)
        return decision

    # 4) 보류
    return None


def needs_face_hint(track_id: int, now: float) -> bool:
    """얼굴을 못 찾은 지 FACE_HINT_SEC 이상, UNKNOWN_TIMEOUT_SEC 미만이면 True."""
    state = _tracks.get(track_id)
    if state is None:
        return False
    elapsed = now - _no_face_since(state)
    return config.FACE_HINT_SEC <= elapsed < config.UNKNOWN_TIMEOUT_SEC


def reset_all() -> None:
    """모든 추적 상태를 지운다 (테스트용)."""
    _tracks.clear()
