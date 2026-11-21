"""vision/face.py 얼굴 고르기·거울 화면 확인 (카메라·모델 없이, 가짜 얼굴 좌표 사용).

- _pick_face(): 사람 박스 위쪽 가운데(머리 자리)에 가장 가까운 얼굴을 고르는지
  두 사람이 겹쳐 서 있어도 각자 자기 얼굴을 고르는지
- 거울 화면: 화면 좌표만 뒤집히고 원본 frame은 그대로인지

실행 (프로젝트 루트에서):
    python experiments/test_face_pick.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 프로젝트 루트

import numpy as np

from vision import face as F

_results: list[bool] = []


def face_row(x, y, w, h) -> np.ndarray:
    """YuNet 결과 한 줄 모양 (x, y, w, h, 눈·코·입 5점, 점수). 점은 박스 안 대략적인 위치."""
    pts = [x + .3 * w, y + .4 * h, x + .7 * w, y + .4 * h, x + .5 * w, y + .6 * h,
           x + .35 * w, y + .8 * h, x + .65 * w, y + .8 * h]
    return np.array([x, y, w, h, *pts, 0.95], dtype=np.float32)


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(ok)
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))


def same(a, b) -> bool:
    return a is not None and b is not None and np.allclose(a, b)


def test_single():
    print("▶ 한 사람")
    box = (100, 50, 300, 450)
    f = face_row(160, 70, 80, 90)
    check("얼굴 1개 → 그 얼굴", same(F._pick_face(np.stack([f]), box), f))
    check("얼굴 0개 → None", F._pick_face(np.empty((0, 15), np.float32), box) is None)


def test_overlap_bigger_neighbor():
    print("▶ 두 사람이 겹쳐 섬 (옆 사람 얼굴이 더 큼)")
    # 1번: 왼쪽 사람, 2번: 오른쪽 사람(카메라에 더 가까워 얼굴이 큼). 박스가 겹친다
    box1, box2 = (100, 60, 320, 470), (240, 30, 520, 470)
    face1 = face_row(170, 80, 70, 80)       # 1번 머리 자리
    face2 = face_row(330, 50, 110, 125)     # 2번 머리 자리, 더 큼. 중심 x=385 → 1번 박스 밖
    face2_in = face_row(270, 50, 110, 125)  # 2번 얼굴이 1번 박스 오른쪽 끝 안쪽까지 들어온 경우 (중심 x=325)
    faces = np.stack([face1, face2])
    check("가장 큰 얼굴 방식이면 1번도 2번 얼굴을 고름 (고치기 전 문제 재현)", same(F._largest(faces), face2))
    check("1번 박스 → 1번 얼굴", same(F._pick_face(faces, box1), face1))
    check("2번 박스 → 2번 얼굴", same(F._pick_face(faces, box2), face2))
    box1_wide = (100, 60, 340, 470)
    faces_in = np.stack([face1, face2_in])
    check("옆 사람 얼굴이 내 박스 안에 걸쳐도 → 내 얼굴",
          same(F._pick_face(faces_in, box1_wide), face1))
    check("두 박스가 서로 다른 얼굴을 고름",
          not same(F._pick_face(faces, box1), F._pick_face(faces, box2)))


def test_outside_and_lower():
    print("▶ 위치 규칙")
    box = (200, 100, 400, 500)
    inside_top = face_row(260, 120, 80, 90)
    outside = face_row(420, 120, 90, 100)          # 중심이 박스 오른쪽 밖
    lower = face_row(260, 380, 80, 90)             # 박스 아래쪽 (손에 든 사진·포스터 등)
    check("박스 가로 범위 밖 얼굴은 고르지 않음",
          F._pick_face(np.stack([outside]), box) is None)
    check("위쪽 얼굴과 아래쪽 얼굴 중 위쪽(머리 자리)을 고름",
          same(F._pick_face(np.stack([lower, inside_top]), box), inside_top))


def test_far_full_body():
    print("▶ 멀리 선 사람 (전신 박스, 얼굴이 박스 맨 위)")
    box = (280, 40, 360, 460)                      # 좁고 긴 전신 박스
    head = face_row(302, 45, 40, 44)
    check("전신 박스에서도 머리 얼굴을 고름", same(F._pick_face(np.stack([head]), box), head))


def test_mirror():
    print("▶ 거울 화면 (눈은 거울, 머리는 원본)")
    width = 640
    frame = np.zeros((480, width, 3), np.uint8)
    frame[:, :10] = 255                            # 왼쪽 끝에 흰 줄
    f = face_row(100, 50, 80, 90)
    old = getattr(F._config, "MIRROR", None)
    try:
        F._config.MIRROR = True
        view = F._view_of(frame)
        check("화면은 좌우 반전 (흰 줄이 오른쪽 끝으로)", view[:, -1].min() == 255 and view[:, 0].max() == 0)
        check("원본 frame은 그대로", frame[:, 0].min() == 255 and frame[:, -1].max() == 0)
        check("박스 x 좌표도 뒤집힘 (640 − 100 − 80 = 460)", F._view_box(f, width) == (460, 50, 80, 90),
              str(F._view_box(f, width)))
        check("점 좌표도 뒤집힘", F._view_point(100, 50, width) == (539, 50), str(F._view_point(100, 50, width)))
        F._config.MIRROR = False
        check("MIRROR=False면 화면도 원본 방향", np.array_equal(F._view_of(frame), frame))
        check("MIRROR=False면 박스 좌표 그대로", F._view_box(f, width) == (100, 50, 80, 90))
        del F._config.MIRROR
        check("config에 MIRROR가 없으면 꺼짐으로 처리", F._mirror_on() is False)
    finally:
        if old is not None:
            F._config.MIRROR = old


def main() -> int:
    for t in (test_single, test_overlap_bigger_neighbor, test_outside_and_lower, test_far_full_body, test_mirror):
        t()
    print(f"\n결과: {sum(_results)}/{len(_results)} 통과")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
