"""vision/judge.py 규칙 확인 (인수인계서 5.5).

가짜 RecognitionResult를 넣어 vote(), needs_face_hint()가 규칙대로 동작하는지 본다.
카메라·모델 없이 실행된다.

실행 (프로젝트 루트에서):
    python experiments/test_judge.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 프로젝트 루트

import numpy as np

from common.types import RecognitionResult
from config import FACE_HINT_SEC, UNKNOWN_TIMEOUT_SEC, VOTE_MIN, VOTE_WINDOW
from vision import judge
from vision.judge import needs_face_hint, vote

judge.TURN_CHECK = False   # 다수결 규칙만 본다. 고개 돌리기 확인은 test_liveness.py에서 시험

STEP = 0.2   # 가짜 프레임 간격 (초)
_results: list[tuple[str, bool, str]] = []


# ──────────────────────────────────────────────
# 가짜 결과 만들기
# ──────────────────────────────────────────────
def reg(pid: int, name: str, dist: float = 0.3) -> RecognitionResult:
    """등록자로 인식된 결과."""
    return RecognitionResult(face_found=True, person_id=pid, name=name, distance=dist,
                             face_image=np.zeros((50, 50, 3), np.uint8),
                             embedding=np.full(128, pid, np.float32))


def unreg(dist: float = 0.8) -> RecognitionResult:
    """얼굴은 찾았지만 기준값을 넘은 결과 (미등록자)."""
    return RecognitionResult(face_found=True, distance=dist,
                             face_image=np.zeros((50, 50, 3), np.uint8),
                             embedding=np.full(128, 0.5, np.float32))


def noface() -> RecognitionResult:
    return RecognitionResult(face_found=False)


def feed(track_id: int, results: list[RecognitionResult], start: float = 0.0):
    """결과를 STEP 간격으로 차례로 넣고, (각 결과의 vote 반환값 목록, 마지막 시각)을 돌려준다."""
    out, t = [], start
    for r in results:
        out.append(vote(track_id, r, t))
        t += STEP
    return out, t - STEP


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))


# ──────────────────────────────────────────────
# 테스트
# ──────────────────────────────────────────────
def test_registered():
    judge._reset_all()
    dists = [0.35, 0.28, 0.40, 0.31]
    out, _ = feed(1, [reg(7, "홍길동", d) for d in dists[:VOTE_MIN]])
    d = out[-1]
    check("등록자: VOTE_MIN개 전까지는 보류", all(x is None for x in out[:-1]))
    check("등록자: VOTE_MIN개째에 registered 확정", d is not None and d.category == "registered",
          f"받은 값 {d}")
    if d:
        check("등록자: id·이름이 맞음", d.person_id == 7 and d.name == "홍길동")
        check("등록자: 거리는 그중 가장 작은 값", d.distance == min(dists[:VOTE_MIN]), f"{d.distance}")
        check("등록자: 최근 얼굴·벡터가 들어 있음", d.face_image is not None and d.embedding is not None)


def test_unregistered():
    judge._reset_all()
    out, _ = feed(1, [unreg() for _ in range(VOTE_MIN)])
    d = out[-1]
    check("미등록자: VOTE_MIN개째에 unregistered 확정",
          d is not None and d.category == "unregistered" and all(x is None for x in out[:-1]), f"{d}")
    if d:
        check("미등록자: id·이름은 None, 벡터는 있음 (중복 방지용)",
              d.person_id is None and d.name is None and d.embedding is not None)


def test_mixed_stays_pending():
    judge._reset_all()
    results = [reg(1, "A"), unreg(), reg(2, "B"), unreg(), reg(1, "A")]   # 어느 쪽도 VOTE_MIN 미만
    out, _ = feed(1, results[:VOTE_WINDOW])
    check("섞인 결과: 확정하지 않고 보류", all(x is None for x in out), f"{out}")


def test_flicker_and_sliding():
    if (VOTE_WINDOW, VOTE_MIN) != (5, 4):
        print(f"  ⏭️  SKIP 깜빡임·밀려나기 (VOTE_WINDOW={VOTE_WINDOW}, VOTE_MIN={VOTE_MIN}일 때는 건너뜀)")
        return
    judge._reset_all()
    out, _ = feed(1, [reg(1, "A"), reg(1, "A"), unreg(), reg(1, "A"), reg(1, "A")])
    check("깜빡임: 5번 중 4번 같으면 5번째에 확정",
          out[-1] is not None and out[-1].category == "registered" and all(x is None for x in out[:-1]))

    judge._reset_all()
    out, _ = feed(1, [unreg(), unreg(), unreg(), reg(1, "A"), reg(1, "A"), reg(1, "A"), reg(1, "A")])
    first = next((i for i, x in enumerate(out) if x is not None), None)
    check("밀려나기: 오래된 결과는 창 밖으로 빠지고 7번째에 등록자 확정",
          first == 6 and out[6].category == "registered", f"처음 확정된 순서 {first}")


def test_reset_after_decision():
    judge._reset_all()
    _, t = feed(1, [reg(1, "A") for _ in range(VOTE_MIN)])
    d = vote(1, reg(1, "A"), t + STEP)
    check("판정 후 초기화: 바로 다음 결과 하나로는 다시 확정하지 않음", d is None, f"{d}")


def test_unknown_never_found():
    judge._reset_all()
    t, first_unknown, hint_ok = 0.0, None, True
    while t <= UNKNOWN_TIMEOUT_SEC + 1:
        expected_hint = FACE_HINT_SEC <= t < UNKNOWN_TIMEOUT_SEC
        d = vote(1, noface(), t)
        if d is not None and first_unknown is None:
            first_unknown = (t, d)
            break
        if needs_face_hint(1, t) != expected_hint:
            hint_ok = False
        t = round(t + STEP, 6)
    check("얼굴 못 찾음: FACE_HINT_SEC부터 UNKNOWN_TIMEOUT_SEC 전까지만 안내", hint_ok)
    ok = (first_unknown is not None and first_unknown[1].category == "unknown"
          and abs(first_unknown[0] - UNKNOWN_TIMEOUT_SEC) < STEP)
    check("얼굴 못 찾음: UNKNOWN_TIMEOUT_SEC에 unknown 확정", ok, f"{first_unknown}")
    if first_unknown:
        d = first_unknown[1]
        check("인식 불가: 얼굴·벡터 없음", d.face_image is None and d.embedding is None)


def test_unknown_after_face_lost():
    judge._reset_all()
    vote(1, noface(), 0.0)
    vote(1, unreg(), 4.0)          # 4초에 얼굴을 찾음 → 시계가 다시 시작
    d1 = vote(1, noface(), UNKNOWN_TIMEOUT_SEC + 0.1)               # 처음 본 지 5초 넘음
    d2 = vote(1, noface(), 4.0 + UNKNOWN_TIMEOUT_SEC + 0.1)         # 얼굴 찾은 지 5초 넘음
    check("얼굴을 놓친 경우: 처음 본 지가 아니라 얼굴 찾은 지 기준으로 unknown",
          d1 is None and d2 is not None and d2.category == "unknown", f"d1={d1}, d2={d2}")


def test_face_found_no_hint():
    judge._reset_all()
    for i in range(10):
        vote(1, reg(1, "A") if i % 2 == 0 else unreg(), i * STEP)   # 얼굴은 계속 보임
    check("얼굴이 보이는 동안은 안내하지 않음", not needs_face_hint(1, 10 * STEP))


def test_multi_track():
    judge._reset_all()
    out1, out2, t = [], [], 0.0
    for _ in range(VOTE_MIN):
        out1.append(vote(1, reg(3, "C"), t))
        out2.append(vote(2, unreg(), t))
        t += STEP
    ok = (out1[-1] is not None and out1[-1].category == "registered" and out1[-1].track_id == 1
          and out2[-1] is not None and out2[-1].category == "unregistered" and out2[-1].track_id == 2)
    check("여러 명: 추적 번호별 결과가 섞이지 않음", ok, f"{out1[-1]}, {out2[-1]}")


def test_stale_cleanup_and_unknown_track():
    judge._reset_all()
    vote(1, noface(), 0.0)
    vote(2, noface(), judge.STALE_SEC + 1)          # 다른 사람 때문에 정리가 일어남
    check("오래된 추적 번호는 지워짐", 1 not in judge._tracks)
    check("모르는 추적 번호는 안내하지 않음", needs_face_hint(99, 0.0) is False)


# ──────────────────────────────────────────────
def main() -> int:
    print(f"설정: VOTE_WINDOW={VOTE_WINDOW}, VOTE_MIN={VOTE_MIN}, "
          f"FACE_HINT_SEC={FACE_HINT_SEC}, UNKNOWN_TIMEOUT_SEC={UNKNOWN_TIMEOUT_SEC}\n")
    tests = [test_registered, test_unregistered, test_mixed_stays_pending, test_flicker_and_sliding,
             test_reset_after_decision, test_unknown_never_found, test_unknown_after_face_lost,
             test_face_found_no_hint, test_multi_track, test_stale_cleanup_and_unknown_track]
    for t in tests:
        print(f"▶ {t.__name__}")
        t()
    failed = [r for r in _results if not r[1]]
    print(f"\n결과: {len(_results) - len(failed)}/{len(_results)} 통과")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
