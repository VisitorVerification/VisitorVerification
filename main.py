import argparse
import itertools
import logging
import time

import cv2

from common.types import Decision
from config import (CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH, MESSAGE_SEC, MIRROR, MSG_COUNTER, MSG_WELCOME,
                    SERIAL_BAUD, SERIAL_PORT, USE_ARDUINO, WINDOW_NAME)
from output import door as door_module
from output.display import draw_debug, mirror, setup_window, show_message, toggle_fullscreen
from vision import detector

log = logging.getLogger("main")


# ── 가짜 함수 (--fake): A·C의 함수가 오기 전까지 쓴다. 4주차에 실제 함수로 바꾼다 ──

_fake_visit_ids = itertools.count(1)

# 키보드 가짜 판정: r 등록자 "테스트" / u 미등록자 / k 인식 불가
FAKE_KEYS = {ord("r"): "registered", ord("u"): "unregistered", ord("k"): "unknown"}
FAKE_NAME = "테스트"


def fake_save_visit(decision: Decision, message: str, frame, approval_status: str | None = None) -> int:
    """C의 save_visit() 대신: 콘솔에 출력하고 번호를 1씩 늘려 돌려준다."""
    visit_id = next(_fake_visit_ids)
    print(f"[FakeDB] save_visit #{visit_id} {decision.category} "
          f"(name={decision.name}, track={decision.track_id}) \"{message}\"")
    return visit_id


def fake_save_door_event(event: str, visit_id: int | None) -> None:
    """C의 save_door_event() 대신: 콘솔에 출력한다."""
    print(f"[FakeDB] save_door_event {event} (visit {visit_id})")


def fake_decision(key: int, people: list) -> Decision:
    """A의 vote() 대신: 키보드로 판정을 만든다. 화면에 사람이 있으면 첫 번째 사람의 추적 번호를 쓴다."""
    category = FAKE_KEYS[key]
    track_id = people[0].track_id if people else 0
    if category == "registered":
        return Decision(category, track_id, name=FAKE_NAME)
    return Decision(category, track_id)


# ── 메인 루프 ──

def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def make_door():
    """USE_ARDUINO면 Door, 아니면 FakeDoor."""
    if USE_ARDUINO:
        if hasattr(door_module, "Door"):
            return door_module.Door(SERIAL_PORT, SERIAL_BAUD)
        log.warning("Door가 아직 없어서 FakeDoor를 씁니다.")
    return door_module.FakeDoor()


def handle(decision: Decision, frame, now: float, door, save_visit) -> tuple[str, str, float]:
    """확정된 판정으로 기록·문 명령을 하고, 띄울 배너 (문구, 종류, 끝나는 시각)를 돌려준다."""
    if decision.category == "registered":
        message = MSG_WELCOME.format(name=decision.name)
        visit_id = save_visit(decision, message, frame)
        door.open(visit_id)
        return (message, "welcome", now + MESSAGE_SEC)
    # "unregistered" 또는 "unknown"
    message = MSG_COUNTER
    save_visit(decision, message, frame)
    door.deny()
    return (message, "counter", now + MESSAGE_SEC)


def run(fake: bool) -> None:
    if not fake:
        print("A·C의 함수(얼굴 인식, DB)가 아직 연결되지 않았습니다 (4주차 예정).")
        print("지금은 python main.py --fake 로 실행하세요.")
        return

    save_visit, save_door_event = fake_save_visit, fake_save_door_event   # 4주차: C의 함수로 교체

    door = make_door()
    door.on_event(lambda event, visit_id: save_door_event(event, visit_id))
    cap = open_camera()
    if not cap.isOpened():
        log.error("카메라를 열 수 없습니다 (CAMERA_INDEX=%s)", CAMERA_INDEX)
        door.close()
        return

    setup_window(fullscreen=not fake)   # --fake는 창 모드로 시작, f로 전환
    print("키: r 등록자 / u 미등록자 / k 인식 불가 / f 전체 화면 / d 디버그 / q·Esc 종료")

    banner = None          # (문구, 종류, 끝나는 시각)
    debug = False
    people = []
    prev, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.warning("카메라 영상을 읽지 못했습니다. 다시 시도합니다.")
                if cv2.waitKey(100) & 0xFF in (ord("q"), 27):
                    break
                continue
            now = time.time()

            try:
                people = detector.detect_people(frame)
            except Exception:
                log.exception("사람 판별 중 오류")
                people = []
            # 4주차: 여기서 FACE_INTERVAL마다 recognize() → vote() → handle() (A)

            if banner and now > banner[2]:
                banner = None
            # 화면에 보일 것만 거울 반전 (인식·기록은 원본 frame). 배너 글자는 뒤집은 뒤에 그림
            if debug:
                shown = draw_debug(frame, people, None, detector.last_animals, fps, mirrored=MIRROR)
            else:
                shown = mirror(frame) if MIRROR else frame
            if banner:
                shown = show_message(shown, banner[0], banner[1])
            cv2.imshow(WINDOW_NAME, shown)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                toggle_fullscreen()
            elif key == ord("d"):
                debug = not debug
            elif key in FAKE_KEYS:
                try:
                    banner = handle(fake_decision(key, people), frame, now, door, save_visit)
                except Exception:
                    log.exception("판정 처리 중 오류")
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:   # 창 닫기 버튼
                break

            current = 1.0 / max(now - prev, 1e-6)
            fps = current if fps == 0 else 0.9 * fps + 0.1 * current
            prev = now
    finally:
        door.close()
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="방문자 확인 시스템 메인 프로그램")
    parser.add_argument("--fake", action="store_true",
                        help="A·C 함수 대신 가짜 함수 사용 (키보드 r/u/k로 판정)")
    args = parser.parse_args()
    run(fake=args.fake)


if __name__ == "__main__":
    main()
