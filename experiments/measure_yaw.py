"""고개 돌리기 확인 실험: 코 위치 비율 r 측정 (A 담당, 사진 대기 대책 1단계).

YuNet이 주는 두 눈·코 점으로 r을 계산해 화면에 실시간으로 보여 준다.
    r = 코를 두 눈을 잇는 선에 비춘 위치 (왼쪽 눈 = 0, 오른쪽 눈 = 1, 원본 화면 기준)
    정면이면 약 0.5, 고개를 돌리면 한쪽으로 움직이고, 평평한 사진은 기울여도 거의 그대로일 것으로 예상한다.
눈을 잇는 선에 비춰서 재므로 고개를 옆으로 갸웃(기울임)해도 값이 덜 흔들린다.

실행 (프로젝트 루트에서):
    python experiments/measure_yaw.py
    python experiments/measure_yaw.py --seconds 3 --delta 0.15

조작:
    b       지금 r을 정면 기준값으로 정하기 (정면을 보고 누르기)
    스페이스 --seconds(기본 judge.TURN_SEC) 동안 r을 모아 콘솔에 결과 출력 (진짜 얼굴 / 사진 비교용)
    q / Esc 끝내기

화면은 config.MIRROR를 따라 거울로 보이지만, r은 항상 원본 화면으로 계산한다 (main.py와 같음).
화면 글자는 영문만 쓰고, 측정 결과는 콘솔에 한글로 출력한다.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 프로젝트 루트

import cv2
import numpy as np

from config import CAMERA_INDEX
from vision import face as F

from vision import judge as J

head_ratio = F.head_ratio   # main.py의 판정과 같은 계산을 쓴다


def turn_label(delta: float, threshold: float) -> str:
    """기준값과의 차이로 방향을 표시한다. judge.LEFT_SIGN(가설)을 따른다. 실험으로 맞는지 확인한다."""
    moved_left = delta * J.LEFT_SIGN
    if moved_left >= threshold:
        return "MY LEFT ?"
    if moved_left <= -threshold:
        return "MY RIGHT ?"
    return "FRONT"


def report(samples: list[tuple[float, float]], baseline: float | None, seconds: float) -> None:
    """스페이스 측정 결과를 콘솔에 출력한다. samples: [(시각, r), ...]"""
    if not samples:
        print(f"[측정] {seconds:.1f}초 동안 얼굴을 한 번도 못 찾았습니다.")
        return
    values = np.array([r for _, r in samples])
    base = baseline if baseline is not None else float(values[0])
    base_note = "b로 정한 값" if baseline is not None else "첫 값 (b로 기준값을 정하면 더 정확해요)"
    print(f"[측정] {seconds:.1f}초, 샘플 {len(values)}개 (초당 {len(values) / seconds:.1f}개)")
    print(f"       기준값 {base:.3f} ({base_note})")
    print(f"       최소 {values.min():.3f}  최대 {values.max():.3f}  폭 {values.max() - values.min():.3f}")
    print(f"       기준보다 최대 +{values.max() - base:.3f} / {values.min() - base:.3f}")


def run(camera_index: int, seconds: float, threshold: float) -> None:
    F._load_models()
    cap = F.open_camera(camera_index)
    if cap is None:
        return

    print("정면을 보고 b → 기준값,  스페이스 → "
          f"{seconds:.0f}초 측정,  q/Esc → 끝내기")
    print(f"방향 표시는 judge.LEFT_SIGN = {J.LEFT_SIGN:+d}을 따라요 "
          f"(+1: r이 커지면 'MY LEFT'), 기준 ±{threshold}")

    window = "measure yaw"
    font = cv2.FONT_HERSHEY_SIMPLEX
    baseline = None
    recording = None          # 측정 중이면 끝나는 시각
    samples: list[tuple[float, float]] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라에서 영상을 읽지 못했습니다")
                break
            now = time.time()
            h, w = frame.shape[:2]

            faces = F._detect_faces(frame)
            main = F._pick_face(faces, (0, 0, w, h))   # recognize()와 같은 방법으로 고른 얼굴
            r = head_ratio(main) if main is not None else None

            if recording is not None:
                if r is not None:
                    samples.append((now, r))
                if now >= recording:
                    report(samples, baseline, seconds)
                    recording = None

            view = F._view_of(frame)
            if main is not None:
                x, y, fw, fh = F._view_box(main, w)
                cv2.rectangle(view, (x, y), (x + fw, y + fh), (0, 200, 0), 2)
                for i in (0, 1, 2):        # 두 눈, 코
                    cv2.circle(view, F._view_point(main[4 + 2 * i], main[5 + 2 * i], w), 3, (255, 0, 255), -1)

            if r is None:
                lines = [("NO FACE (or too far)", (0, 200, 255))]
            else:
                lines = [(f"r = {r:.3f}", (0, 255, 255))]
                if baseline is not None:
                    delta = r - baseline
                    lines.append((f"base {baseline:.3f}  delta {delta:+.3f}  {turn_label(delta, threshold)}",
                                  (0, 255, 255)))
                else:
                    lines.append(("press b while facing front", (0, 200, 255)))
            if recording is not None:
                lines.append((f"RECORDING {max(recording - now, 0):.1f}s  n={len(samples)}", (0, 0, 255)))
            for i, (text, color) in enumerate(lines):
                cv2.putText(view, text, (10, 30 + 28 * i), font, 0.7, color, 2)
            cv2.putText(view, "b: baseline  space: record  q: quit", (10, h - 12), font, 0.6, (0, 255, 255), 2)

            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                if r is None:
                    print("얼굴을 못 찾아서 기준값을 정하지 못했습니다")
                else:
                    baseline = r
                    print(f"[기준값] r = {r:.3f}")
            elif key == ord(" ") and recording is None:
                samples = []
                recording = now + seconds
                print(f"[측정 시작] {seconds:.1f}초")
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:   # 창 닫기 버튼
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="고개 돌리기 확인 실험: 코 위치 비율 r 측정")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="카메라 번호")
    parser.add_argument("--seconds", type=float, default=J.TURN_SEC,
                        help=f"스페이스로 측정할 시간 (기본 judge.TURN_SEC = {J.TURN_SEC}초)")
    parser.add_argument("--delta", type=float, default=J.TURN_DELTA,
                        help=f"방향 표시 기준 (기본 judge.TURN_DELTA = {J.TURN_DELTA})")
    args = parser.parse_args()
    run(args.camera, args.seconds, args.delta)
