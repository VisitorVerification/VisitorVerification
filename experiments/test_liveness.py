"""vision/judge.py 고개 돌리기 확인(사진 대기 대책) 규칙 확인.

가짜 RecognitionResult와 가짜 코 위치 비율 r을 넣어
vote(), turn_prompt(), needs_face_hint()가 규칙대로 동작하는지 본다.
카메라·모델 없이 실행된다.

실행 (프로젝트 루트에서):
    python experiments/test_liveness.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 프로젝트 루트

import numpy as np

from common.types import RecognitionResult
from config import COOLDOWN_SEC, VOTE_MIN
from vision import judge

STEP = 0.2                     # 가짜 프레임 간격 (초)
FRONT = 0.5                    # 정면 r
_ratios: dict[int, float | None] = {}      # 추적 번호 → 지금 프레임의 가짜 r
_results: list[tuple[str, bool, str]] = []


# ──────────────────────────────────────────────
# 준비
# ──────────────────────────────────────────────
def setup(direction: str = "left", left_sign: int = +1) -> None:
    """매 시험 전에 상태를 비우고 가짜 r·방향을 끼운다."""
    judge._reset_all()
    judge.TURN_CHECK = True
    judge.LEFT_SIGN = left_sign
    judge._ratio_source = lambda tid: _ratios.get(tid)
    judge._pick_direction = lambda: direction
    _ratios.clear()


def reg(pid: int = 7, name: str = "홍길동") -> RecognitionResult:
    return RecognitionResult(face_found=True, person_id=pid, name=name, distance=0.3,
                             face_image=np.zeros((50, 50, 3), np.uint8),
                             embedding=np.full(128, pid, np.float32))


def unreg() -> RecognitionResult:
    return RecognitionResult(face_found=True, distance=0.8,
                             face_image=np.zeros((50, 50, 3), np.uint8),
                             embedding=np.full(128, 0.5, np.float32))


def noface() -> RecognitionResult:
    return RecognitionResult(face_found=False)


def step(tid: int, result: RecognitionResult, t: float, r: float | None = FRONT):
    """r을 정하고 vote()를 한 번 부른다. 얼굴이 없으면 r도 None (face.py와 같음)."""
    _ratios[tid] = r if result.face_found else None
    return judge.vote(tid, result, t)


def start_turn(tid: int = 1, t0: float = 0.0, pid: int = 7) -> float:
    """등록자 결과를 VOTE_MIN번 넣어 고개 돌리기 대기를 시작시키고, 마지막 시각을 돌려준다."""
    t = t0
    for _ in range(VOTE_MIN):
        d = step(tid, reg(pid), t)
        t += STEP
    assert d is None, "고개 돌리기 대기가 시작되지 않았습니다"
    return t - STEP


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))


# ──────────────────────────────────────────────
# 시험
# ──────────────────────────────────────────────
def test_starts_turn_instead_of_registered():
    setup("left")
    t = start_turn()
    check("등록자 다수결이 나와도 바로 확정하지 않고 고개 돌리기 대기", judge._tracks[1].turn is not None)
    check("대기 중에는 안내 문구를 돌려줌", judge.turn_prompt(1, t) == judge.TURN_MESSAGES["left"],
          f"{judge.turn_prompt(1, t)}")


def test_turn_left_passes():
    setup("left")
    t = start_turn()
    d1 = step(1, reg(), t + STEP, FRONT + 0.05)                 # 조금 돌림 → 아직
    d2 = step(1, reg(), t + 2 * STEP, FRONT + judge.TURN_DELTA + 0.02)   # 충분히 돌림 → 통과
    check("왼쪽 요구: 조금 돌리면 아직 보류", d1 is None, f"{d1}")
    ok = d2 is not None and d2.category == "registered" and d2.person_id == 7
    check("왼쪽 요구: 내 왼쪽으로 충분히 돌리면 registered", ok, f"{d2}")
    if ok:
        check("통과한 판정에 이름·얼굴·벡터가 그대로 있음",
              d2.name == "홍길동" and d2.face_image is not None and d2.embedding is not None)
    check("통과 후에는 안내 문구가 사라짐", judge.turn_prompt(1, t + 2 * STEP) is None)


def test_turn_right_passes():
    setup("right")
    t = start_turn()
    d = step(1, reg(), t + STEP, FRONT - judge.TURN_DELTA - 0.02)
    check("오른쪽 요구: 내 오른쪽으로 돌리면 registered", d is not None and d.category == "registered", f"{d}")


def test_wrong_direction_fails_now():
    setup("left")
    t = start_turn()
    d = step(1, reg(), t + STEP, FRONT - judge.TURN_DELTA - 0.02)   # 반대(오른쪽)로 돌림
    ok = d is not None and d.category == "unknown"
    check("반대 방향으로 돌리면 시간이 남아도 바로 unknown", ok, f"{d}")
    if ok:
        check("실패 판정에는 얼굴 벡터가 없음 (C의 DB 규칙), 얼굴 사진은 남김",
              d.embedding is None and d.face_image is not None and d.person_id is None)


def test_photo_times_out():
    setup("left")
    t = start_turn()
    jitter = [0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01, -0.005, 0.0, 0.01, 0.02, -0.02]
    outs, k = [], 1
    while t + k * STEP < t + judge.TURN_SEC - 1e-9:                 # 2초가 되기 전까지
        outs.append(step(1, reg(), t + k * STEP, FRONT + jitter[k % len(jitter)]))
        k += 1
    check("사진(r이 거의 그대로): 시간 안에는 보류", all(o is None for o in outs), f"{outs}")
    d = step(1, reg(), t + judge.TURN_SEC, FRONT + 0.01)
    check("사진(r이 거의 그대로): TURN_SEC이 지나면 unknown", d is not None and d.category == "unknown", f"{d}")


def test_baseline_from_first_ratio():
    setup("left")
    t = 0.0
    for i in range(VOTE_MIN):
        r = None if i == VOTE_MIN - 1 else FRONT                   # 확정 순간에 r을 못 잼
        _ratios[1] = r
        judge.vote(1, reg(), t)
        t += STEP
    t -= STEP
    check("확정 순간 r이 없으면 기준값은 비어 있음", judge._tracks[1].turn.baseline is None)
    step(1, reg(), t + STEP, 0.6)                                   # 처음 잰 값이 기준값
    check("대기 중 처음 잰 r이 기준값이 됨", judge._tracks[1].turn.baseline == 0.6)
    d = step(1, reg(), t + 2 * STEP, 0.6 + judge.TURN_DELTA + 0.01)
    check("그 기준값에서 돌리면 통과", d is not None and d.category == "registered", f"{d}")


def test_no_face_during_turn():
    setup("left")
    t = start_turn()
    d1 = step(1, noface(), t + STEP)
    check("대기 중 얼굴을 놓쳐도 바로 실패하지 않음", d1 is None, f"{d1}")
    check("고개 돌리기 안내 중에는 '얼굴이 보이도록' 안내를 띄우지 않음",
          judge.needs_face_hint(1, t + STEP) is False)
    d2 = step(1, noface(), t + judge.TURN_SEC)
    check("얼굴을 계속 못 찾으면 TURN_SEC에 unknown", d2 is not None and d2.category == "unknown", f"{d2}")


def test_prompt_after_deadline():
    setup("right")
    t = start_turn()
    check("안내 문구는 오른쪽 문구", judge.turn_prompt(1, t + 0.5) == judge.TURN_MESSAGES["right"])
    check("시간이 지나면 다음 vote() 전이어도 안내 문구를 안 돌려줌",
          judge.turn_prompt(1, t + judge.TURN_SEC + 0.01) is None)


def test_passed_person_skips_within_cooldown():
    setup("left")
    t = start_turn()
    step(1, reg(), t + STEP, FRONT + judge.TURN_DELTA + 0.05)       # 통과
    t2 = t + 2 * STEP
    out = [step(1, reg(), t2 + i * STEP) for i in range(VOTE_MIN)]
    d = out[-1]
    check("통과한 사람은 COOLDOWN_SEC 안에 다시 시키지 않고 바로 registered",
          d is not None and d.category == "registered", f"{d}")
    t3 = t + COOLDOWN_SEC + 1.0
    out = [step(1, reg(), t3 + i * STEP) for i in range(VOTE_MIN)]
    check("COOLDOWN_SEC이 지나면 다시 고개 돌리기를 시킴",
          out[-1] is None and judge._tracks[1].turn is not None, f"{out[-1]}")


def test_failed_person_asked_again():
    setup("left")
    t = start_turn()
    step(1, reg(), t + judge.TURN_SEC)                              # 시간 초과 → 실패
    out = [step(1, reg(), t + judge.TURN_SEC + (i + 1) * STEP) for i in range(VOTE_MIN)]
    check("실패한 사람(사진)은 다시 시도하면 또 고개 돌리기를 시킴",
          out[-1] is None and judge._tracks[1].turn is not None, f"{out[-1]}")


def test_unregistered_not_affected():
    setup("left")
    out = [step(2, unreg(), i * STEP) for i in range(VOTE_MIN)]
    d = out[-1]
    check("미등록자는 고개 돌리기 없이 바로 unregistered", d is not None and d.category == "unregistered", f"{d}")
    check("미등록자에게는 안내 문구 없음", judge.turn_prompt(2, VOTE_MIN * STEP) is None)


def test_turn_check_off():
    setup("left")
    judge.TURN_CHECK = False
    out = [step(1, reg(), i * STEP) for i in range(VOTE_MIN)]
    check("TURN_CHECK = False면 예전처럼 바로 registered",
          out[-1] is not None and out[-1].category == "registered", f"{out[-1]}")


def test_left_sign_flip():
    setup("left", left_sign=-1)
    t = start_turn()
    d = step(1, reg(), t + STEP, FRONT - judge.TURN_DELTA - 0.02)   # r이 작아지는 쪽 = 내 왼쪽
    check("LEFT_SIGN = -1이면 r이 작아지는 쪽이 내 왼쪽", d is not None and d.category == "registered", f"{d}")


def test_multi_track():
    setup("left")
    t = start_turn(tid=1, pid=7)
    out = [step(2, unreg(), t + (i + 1) * STEP) for i in range(VOTE_MIN)]
    check("한 사람이 대기 중이어도 다른 추적 번호는 따로 판정",
          out[-1] is not None and out[-1].category == "unregistered" and judge._tracks[1].turn is not None)


# ──────────────────────────────────────────────
def main() -> int:
    print(f"설정: TURN_SEC={judge.TURN_SEC}, TURN_DELTA={judge.TURN_DELTA}, LEFT_SIGN={judge.LEFT_SIGN:+d}, "
          f"VOTE_MIN={VOTE_MIN}, COOLDOWN_SEC={COOLDOWN_SEC}\n")
    tests = [test_starts_turn_instead_of_registered, test_turn_left_passes, test_turn_right_passes,
             test_wrong_direction_fails_now, test_photo_times_out, test_baseline_from_first_ratio,
             test_no_face_during_turn, test_prompt_after_deadline, test_passed_person_skips_within_cooldown,
             test_failed_person_asked_again, test_unregistered_not_affected, test_turn_check_off,
             test_left_sign_flip, test_multi_track]
    for t in tests:
        print(f"▶ {t.__name__}")
        t()
    failed = [r for r in _results if not r[1]]
    print(f"\n결과: {len(_results) - len(failed)}/{len(_results)} 통과")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
