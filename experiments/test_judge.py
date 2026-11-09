"""가짜 RecognitionResult로 vision/judge.py의 규칙을 확인한다.

실행:  python experiments/test_judge.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from common.types import RecognitionResult
from vision import judge


def reg(pid, name, d=0.3):
    return RecognitionResult(face_found=True, person_id=pid, name=name, distance=d,
                             face_image=np.zeros((4, 4, 3), np.uint8),
                             embedding=np.ones(128, np.float32))


def unreg(d=0.8):
    return RecognitionResult(face_found=True, distance=d,
                             face_image=np.zeros((4, 4, 3), np.uint8),
                             embedding=np.full(128, 0.5, np.float32))


NOFACE = RecognitionResult(face_found=False)


def feed(track, results, t0=0.0, step=0.2):
    """결과를 차례로 넣고 (시각, 판정) 목록을 돌려준다."""
    out = []
    for i, r in enumerate(results):
        t = t0 + i * step
        out.append((t, judge.vote(track, r, t)))
    return out


def test_registered_after_vote_min():
    judge.reset_all()
    res = feed(1, [reg(7, "홍길동", 0.35), reg(7, "홍길동", 0.30),
                   reg(7, "홍길동", 0.40), reg(7, "홍길동", 0.33)])
    assert all(d is None for _, d in res[:3]), "VOTE_MIN 전에 판정되면 안 됨"
    d = res[3][1]
    assert d and d.category == "registered" and d.person_id == 7 and d.name == "홍길동"
    assert abs(d.distance - 0.30) < 1e-9, "결과들 중 가장 작은 거리"
    assert d.face_image is not None and d.embedding is not None


def test_mixed_stays_pending():
    judge.reset_all()
    res = feed(1, [reg(1, "a"), reg(2, "b"), reg(1, "a"), unreg(), reg(1, "a"), reg(2, "b")])
    assert all(d is None for _, d in res), "섞이면 보류"


def test_window_slides():
    judge.reset_all()
    # 창 크기 5: [2,1,1,1,1] 에서 1이 4개 → 확정
    res = feed(1, [reg(2, "b"), reg(1, "a"), reg(1, "a"), reg(1, "a"), reg(1, "a")])
    assert res[-1][1] and res[-1][1].person_id == 1
    judge.reset_all()
    # [1,U,1,1,U,1] → 마지막 창 [U,1,1,U,1] 은 1이 3개 → 보류
    res = feed(1, [reg(1, "a"), unreg(), reg(1, "a"), reg(1, "a"), unreg(), reg(1, "a")])
    assert all(d is None for _, d in res), "창 안에 1이 3개뿐이면 보류"
    # 1을 하나 더 → 창 [1,1,U,1,1] 에서 1이 4개 → 확정
    d = judge.vote(1, reg(1, "a"), 1.2)
    assert d and d.person_id == 1, "창 [1,1,U,1,1] 에서 1이 4개면 확정"


def test_unregistered():
    judge.reset_all()
    res = feed(3, [unreg(0.7), unreg(0.9), unreg(0.8), unreg(0.75)])
    d = res[-1][1]
    assert d and d.category == "unregistered" and d.person_id is None
    assert d.embedding is not None and abs(d.distance - 0.75) < 1e-9


def test_no_face_hint_then_unknown():
    judge.reset_all()
    assert judge.vote(5, NOFACE, 0.0) is None
    assert not judge.needs_face_hint(5, 1.9)
    assert judge.needs_face_hint(5, config.FACE_HINT_SEC)
    assert judge.needs_face_hint(5, 4.9)
    assert judge.vote(5, NOFACE, 4.9) is None
    d = judge.vote(5, NOFACE, config.UNKNOWN_TIMEOUT_SEC)
    assert d and d.category == "unknown" and d.face_image is None and d.embedding is None
    assert not judge.needs_face_hint(5, config.UNKNOWN_TIMEOUT_SEC), "판정 후 초기화"


def test_face_lost_after_found():
    judge.reset_all()
    judge.vote(6, unreg(), 0.0)
    judge.vote(6, reg(1, "a"), 1.0)          # 마지막으로 얼굴 찾은 시각 = 1.0
    assert judge.vote(6, NOFACE, 3.0) is None
    assert judge.needs_face_hint(6, 3.0)
    assert judge.vote(6, NOFACE, 5.9) is None
    d = judge.vote(6, NOFACE, 6.0)
    assert d and d.category == "unknown"


def test_tracks_independent():
    judge.reset_all()
    for i in range(3):
        assert judge.vote(10, reg(1, "a"), i * 0.2) is None
        assert judge.vote(11, unreg(), i * 0.2 + 0.1) is None
    d10 = judge.vote(10, reg(1, "a"), 0.6)
    d11 = judge.vote(11, unreg(), 0.7)
    assert d10.category == "registered" and d10.track_id == 10
    assert d11.category == "unregistered" and d11.track_id == 11


def test_reset_after_decision():
    judge.reset_all()
    feed(1, [reg(1, "a")] * 4)
    res = feed(1, [reg(1, "a")] * 3, t0=1.0)
    assert all(d is None for _, d in res), "판정 뒤에는 처음부터 다시 모은다"


def test_stale_track_removed():
    judge.reset_all()
    feed(1, [reg(1, "a")] * 3)
    judge.vote(2, NOFACE, 20.0)               # 다른 추적 번호 호출 → 오래된 1번 정리
    assert 1 not in judge._tracks
    assert judge.vote(1, reg(1, "a"), 20.2) is None, "지워진 뒤 새로 시작"


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  통과  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  실패  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 통과 "
          f"(VOTE_WINDOW={config.VOTE_WINDOW}, VOTE_MIN={config.VOTE_MIN}, "
          f"FACE_HINT_SEC={config.FACE_HINT_SEC}, UNKNOWN_TIMEOUT_SEC={config.UNKNOWN_TIMEOUT_SEC})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
