"""다수결 판정 (A 담당, 인수인계서 5.5).

추적 번호(track_id)마다 최근 인식 결과를 모아
등록자 / 미등록자 / 인식 불가 를 확정한다.

B의 메인 루프 사용법 (4.4)
    decision = vote(person.track_id, result, now)    # 얼굴 인식을 한 프레임에서만
    if needs_face_hint(person.track_id, now): ...    # 매 프레임 불러도 됨

같은 사람에게 반복해서 안내하지 않는 일(쿨다운)은 B가 맡는다.

고개 돌리기 확인 (사진 대기 대책)
    등록자로 확정되면 바로 돌려주지 않고 "고개를 내 왼쪽/오른쪽으로 돌려 주세요"(방향 무작위)를 요구한다.
    TURN_SEC 안에 그 방향으로 코 위치 비율 r이 TURN_DELTA 이상 움직이면 registered,
    반대로 움직이거나 시간이 지나면 unknown(얼굴 벡터 없음)으로 돌려준다.
    대기 중 띄울 문구는 turn_prompt(track_id, now)로 B의 메인 루프가 받아 간다.
    통과한 사람은 COOLDOWN_SEC 동안 다시 시키지 않는다 (B의 쿨다운과 같은 기간).
"""
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from common.types import Decision, RecognitionResult
from config import COOLDOWN_SEC, FACE_HINT_SEC, UNKNOWN_TIMEOUT_SEC, VOTE_MIN, VOTE_WINDOW

STALE_SEC = 10          # 이 시간(초) 넘게 갱신되지 않은 추적 번호는 지운다 (5.5)
UNREG = "UNREG"         # 최근 결과 목록에서 미등록자를 나타내는 값

# ── 고개 돌리기 확인 (사진 대기 대책) ──
TURN_CHECK = True       # False면 예전처럼 등록자를 바로 확정한다
TURN_SEC = 3.0          # 안내가 뜬 뒤 고개를 돌릴 시간 (초). 2초는 안내를 읽고 돌리기에 짧아서 3초로
TURN_DELTA = 0.15       # 기준보다 r이 이만큼 움직이면 "돌렸다" (임시값, measure_yaw.py로 조정)
LEFT_SIGN = +1          # r이 커지는 쪽이 "내 왼쪽"이면 +1, 반대면 -1 (가설, 직접 확인 후 조정)
TURN_MESSAGES = {
    "left": "고개를 내 왼쪽으로 돌려 주세요.",
    "right": "고개를 내 오른쪽으로 돌려 주세요.",
}


@dataclass
class _Vote:
    label: object                 # 등록자면 person_id(int), 미등록이면 UNREG
    name: Optional[str]
    distance: Optional[float]


@dataclass
class _Turn:
    """고개 돌리기 대기 상태."""
    direction: str                # "left" | "right" (내 몸 기준)
    deadline: float               # 이 시각까지 돌려야 통과
    baseline: Optional[float]     # 대기를 시작할 때의 r (없으면 대기 중 처음 잰 r)
    pending: Decision             # 통과하면 돌려줄 registered 판정


@dataclass
class _TrackState:
    first_seen: float                              # 처음 vote()가 불린 시각 (판정 후에는 다시 잼)
    last_update: float                             # 마지막 vote() 시각
    last_face: Optional[float] = None              # 마지막으로 얼굴을 찾은 시각
    votes: deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    face_image: Optional[np.ndarray] = None        # 최근 얼굴 정보
    embedding: Optional[np.ndarray] = None
    distance: Optional[float] = None
    turn: Optional[_Turn] = None                   # 고개 돌리기 대기 중이면 그 상태


_tracks: dict[int, _TrackState] = {}
_passed: dict[int, float] = {}          # person_id → 고개 돌리기를 통과한 시각 (COOLDOWN_SEC 동안 다시 안 시킴)

# r을 받아 오는 함수. 기본은 vision.face.last_head_ratio (시험에서는 가짜 함수로 바꾼다)
_ratio_source = None


def _ratio(track_id: int) -> Optional[float]:
    if _ratio_source is not None:
        return _ratio_source(track_id)
    from vision import face   # 시험에서 모델을 안 불러오도록 필요할 때만 가져온다
    return face.last_head_ratio(track_id)


def _pick_direction() -> str:
    """방향을 무작위로 고른다 (시험에서는 바꿔 끼운다)."""
    return random.choice(("left", "right"))


# ──────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────
def _cleanup(now: float) -> None:
    """STALE_SEC 넘게 갱신되지 않은 추적 번호를 지운다."""
    for tid in [t for t, s in _tracks.items() if now - s.last_update > STALE_SEC]:
        del _tracks[tid]
    for pid in [p for p, at in _passed.items() if now - at >= COOLDOWN_SEC]:
        del _passed[pid]


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
    state.turn = None


def _reset_all() -> None:
    """모든 추적 상태를 지운다 (테스트용)."""
    _tracks.clear()
    _passed.clear()


def _start_turn(state: _TrackState, track_id: int, decision: Decision, now: float) -> None:
    """등록자 판정을 보류하고 고개 돌리기 대기를 시작한다."""
    state.turn = _Turn(direction=_pick_direction(), deadline=now + TURN_SEC,
                       baseline=_ratio(track_id), pending=decision)


def _check_turn(state: _TrackState, track_id: int, result: RecognitionResult,
                now: float) -> Decision | None:
    """고개 돌리기 대기 중일 때. 통과 → registered, 실패 → unknown, 아직이면 None."""
    turn = state.turn
    r = _ratio(track_id) if result.face_found else None
    if r is not None:
        if turn.baseline is None:
            turn.baseline = r
        else:
            moved_left = (r - turn.baseline) * LEFT_SIGN        # +면 내 왼쪽, -면 내 오른쪽으로 돌림
            moved = moved_left if turn.direction == "left" else -moved_left
            if moved >= TURN_DELTA:                              # 요구한 방향으로 돌림 → 통과
                decision = turn.pending
                if decision.person_id is not None:
                    _passed[decision.person_id] = now
                _reset(state, now)
                return decision
            if moved <= -TURN_DELTA:                             # 반대 방향 → 바로 실패 (영상 대책)
                return _fail_turn(state, track_id, now)
    if now >= turn.deadline:                                     # 시간 안에 못 돌림 → 실패
        return _fail_turn(state, track_id, now)
    return None


def _fail_turn(state: _TrackState, track_id: int, now: float) -> Decision:
    """고개 돌리기 실패 → unknown. C의 DB 규칙대로 얼굴 벡터는 넣지 않는다."""
    face_image = state.turn.pending.face_image
    _reset(state, now)
    return Decision(category="unknown", track_id=track_id, face_image=face_image)


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

    # 0) 고개 돌리기 대기 중이면 그것만 확인한다
    if state.turn is not None:
        if result.face_found:
            state.last_face = now
        return _check_turn(state, track_id, result, now)

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
            # 고개 돌리기: 켜져 있고, 최근에 통과한 사람이 아니면 판정을 보류하고 요구한다
            if TURN_CHECK and decision.person_id not in _passed:
                _start_turn(state, track_id, decision, now)
                return None
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
    if state.turn is not None:          # 고개 돌리기 안내 중에는 얼굴 안내를 띄우지 않는다
        return False
    elapsed = _no_face_seconds(state, now)
    return FACE_HINT_SEC <= elapsed < UNKNOWN_TIMEOUT_SEC


def turn_prompt(track_id: int, now: float) -> str | None:
    """고개 돌리기 대기 중이면 화면에 띄울 문구, 아니면 None. 매 프레임 불러도 된다."""
    state = _tracks.get(track_id)
    if state is None or state.turn is None or now - state.last_update > STALE_SEC:
        return None
    if now >= state.turn.deadline:      # 시간이 지났으면 다음 vote()에서 실패로 확정된다
        return None
    return TURN_MESSAGES[state.turn.direction]
