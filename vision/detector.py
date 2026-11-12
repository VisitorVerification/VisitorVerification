import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

from common.types import PersonBox
from config import (CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH, MIN_PERSON_HEIGHT, MIRROR,
                    YOLO_CONF, YOLO_IMGSZ, YOLO_MODEL)

PERSON, CAT, DOG = 0, 15, 16

_model = None

# 마지막 프레임에서 찾은 고양이·강아지 (디버그 표시용). (이름, 추적 번호, bbox, conf)
last_animals: list[tuple[str, int, tuple[int, int, int, int], float]] = []


def _get_model() -> YOLO:
    """모델은 처음 부를 때 한 번만 불러온다."""
    global _model
    if _model is None:
        _model = YOLO(YOLO_MODEL)
    return _model


def detect_people(frame: np.ndarray) -> list[PersonBox]:
    """프레임에서 사람 박스와 추적 번호를 찾는다. 고양이·강아지는 last_animals에만 기억한다."""
    model = _get_model()
    results = model.track(frame, persist=True, classes=[PERSON, CAT, DOG],
                          conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)
    boxes = results[0].boxes

    people: list[PersonBox] = []
    animals = []
    if boxes.id is not None:   # 추적 번호가 아직 없으면 건너뜀
        for xyxy, cls, conf, track_id in zip(boxes.xyxy.tolist(), boxes.cls.tolist(),
                                             boxes.conf.tolist(), boxes.id.tolist()):
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            if int(cls) == PERSON:
                if y2 - y1 >= MIN_PERSON_HEIGHT:   # 멀리 있는(작은) 사람은 뺌
                    people.append(PersonBox(int(track_id), (x1, y1, x2, y2), float(conf)))
            else:
                animals.append((model.names[int(cls)], int(track_id), (x1, y1, x2, y2), float(conf)))

    last_animals[:] = animals
    return people


def _demo() -> None:
    """웹캠 화면에 사람 박스와 추적 번호를 그린다. q 또는 Esc로 끝낸다."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print(f"카메라를 열 수 없습니다 (CAMERA_INDEX={CAMERA_INDEX})")
        return

    window = "detector demo"
    font = cv2.FONT_HERSHEY_SIMPLEX
    prev, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라에서 영상을 읽지 못했습니다")
                break

            people = detect_people(frame)   # 판별은 원본으로
            w = frame.shape[1]
            if MIRROR:                       # 보이는 화면만 거울 반전, 박스 좌표도 맞춰 뒤집음
                frame = cv2.flip(frame, 1)

            def place(bbox):
                x1, y1, x2, y2 = bbox
                return (w - 1 - x2, y1, w - 1 - x1, y2) if MIRROR else bbox

            for p in people:
                x1, y1, x2, y2 = place(p.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"ID {p.track_id} {p.conf:.2f}", (x1, max(y1 - 8, 16)),
                            font, 0.6, (0, 200, 0), 2)
            for name, _, bbox, _ in last_animals:
                x1, y1, x2, y2 = place(bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 2)
                cv2.putText(frame, f"{name} (ignored)", (x1, max(y1 - 8, 16)),
                            font, 0.6, (160, 160, 160), 2)

            now = time.time()
            current = 1.0 / max(now - prev, 1e-6)
            fps = current if fps == 0 else 0.9 * fps + 0.1 * current
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 25), font, 0.7, (0, 255, 255), 2)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # q 또는 Esc
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:   # 창 닫기 버튼
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 사람 판별")
    parser.add_argument("--demo", action="store_true", help="웹캠으로 사람 박스와 추적 번호 보기")
    args = parser.parse_args()
    if args.demo:
        _demo()
    else:
        parser.print_help()
