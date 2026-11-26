import argparse
import functools
import itertools
import logging
import time
from pathlib import Path

import cv2

from common.types import Decision
from config import (CAMERA_INDEX, DB_PATH, FACE_INTERVAL, FRAME_HEIGHT, FRAME_WIDTH, MESSAGE_SEC, MIRROR,
                    MSG_COUNTER, MSG_FACE_HINT, MSG_WELCOME, SERIAL_BAUD, SERIAL_PORT, USE_ARDUINO,
                    WINDOW_NAME)
from output import door as door_module
from output.display import draw_debug, mirror, setup_window, show_message, toggle_fullscreen
from vision import detector

HINT_SEC = 1           # 얼굴 안내 배너는 안내가 필요한 동안 1초씩 늘린다 (계획서 5.5절)

log = logging.getLogger("main")


# ── 가짜 함수: --fake에서만 쓴다 (판정은 키보드, 기록은 콘솔 출력. 진짜 DB에는 저장하지 않음) ──

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


def recognize_people(frame, people: list, now: float, results: dict, face, judge, announce) -> list:
    """사람마다 A의 recognize() → vote(), 확정되면 should_announce()를 거친 판정만 돌려준다."""
    decisions = []
    for person in people:
        try:
            result = face.recognize(frame, person)
            results[person.track_id] = result
            decision = judge.vote(person.track_id, result, now)
            if decision and announce.should_announce(decision, now):
                decisions.append(decision)
        except Exception:
            log.exception("얼굴 인식 중 오류 (추적 번호 %s)", person.track_id)
    return decisions


def needs_hint(people: list, now: float, judge) -> bool:
    """화면의 누군가가 얼굴 안내가 필요하면 True."""
    try:
        return any(judge.needs_face_hint(p.track_id, now) for p in people)
    except Exception:
        log.exception("얼굴 안내 확인 중 오류")
        return False


def turn_prompt(people: list, now: float, judge) -> str | None:
    """A의 고개 돌리기 확인(사진 대기 대책): 화면의 누군가가 대기 중이면 띄울 문구, 아니면 None."""
    try:
        for p in people:
            message = judge.turn_prompt(p.track_id, now)
            if message:
                return message
    except Exception:
        log.exception("고개 돌리기 안내 확인 중 오류")
    return None


def reload_faces(face, announce) -> None:
    """l 키: 등록 얼굴을 다시 불러오고 쿨다운 기억을 지운다. 실패해도 이전 얼굴로 계속."""
    try:
        face.reload_known_faces()
        announce.reset()
        print("등록 얼굴을 다시 불러오고 쿨다운을 초기화했습니다.")
    except Exception:
        log.exception("등록 얼굴을 다시 불러오지 못했습니다 (이전 등록 얼굴로 계속)")


def open_db():
    """C의 DB를 준비하고 (save_visit, save_door_event, recent_unknown_embeddings)를 돌려준다.
    DB 경로는 A의 등록 얼굴과 같은 config.DB_PATH. 실패하면 예외를 그대로 올린다."""
    from db import database as db
    db.init_db(DB_PATH)
    return (functools.partial(db.save_visit, db_path=DB_PATH),
            functools.partial(db.save_door_event, db_path=DB_PATH),
            functools.partial(db.recent_unknown_embeddings, db_path=DB_PATH))


def door_event_saver(save_door_event):
    """문 이벤트 콜백. 다른 스레드(타이머·시리얼 수신)에서도 불리므로 오류는 여기서 로그로 남긴다."""
    def on_event(event: str, visit_id: int | None) -> None:
        try:
            save_door_event(event, visit_id)
        except Exception:
            log.exception("문 이벤트 기록 중 오류: %s (visit %s)", event, visit_id)
    return on_event


def run(fake: bool) -> None:
    face = judge = announce = None
    if fake:
        save_visit, save_door_event = fake_save_visit, fake_save_door_event
        print("※ --fake: 기록은 콘솔에만 출력하고 DB에는 저장하지 않습니다.")
    else:
        from output import announce
        from vision import face, judge
        try:
            face.init_recognizer()
        except Exception:
            log.exception("얼굴 인식 모델을 불러오지 못했습니다 (models/ 폴더 확인)")
            return
        try:
            save_visit, save_door_event, announce.recent_unknown_embeddings = open_db()
        except Exception:
            log.exception("DB를 열지 못했습니다 (data/ 폴더 확인)")
            return
        print(f"기록 저장 위치: {Path(DB_PATH).resolve()}")

    door = make_door()
    door.on_event(door_event_saver(save_door_event))
    cap = open_camera()
    if not cap.isOpened():
        log.error("카메라를 열 수 없습니다 (CAMERA_INDEX=%s)", CAMERA_INDEX)
        door.close()
        return

    setup_window(fullscreen=not fake)   # --fake는 창 모드로 시작, f로 전환
    if fake:
        print("키: r 등록자 / u 미등록자 / k 인식 불가 / f 전체 화면 / d 디버그 / q·Esc 종료")
    else:
        print("키: l 등록 얼굴 다시 불러오기 / f 전체 화면 / d 디버그 / q·Esc 종료")

    banner = None          # (문구, 종류, 끝나는 시각)
    debug = False
    people = []
    results = {}           # 추적 번호 → 최근 인식 결과 (디버그 표시용)
    frame_no = 0
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
            frame_no += 1

            try:
                people = detector.detect_people(frame)
            except Exception:
                log.exception("사람 판별 중 오류")
                people = []

            if not fake:
                if frame_no % FACE_INTERVAL == 0:
                    for decision in recognize_people(frame, people, now, results, face, judge, announce):
                        try:
                            banner = handle(decision, frame, now, door, save_visit)
                        except Exception:
                            log.exception("판정 처리 중 오류")
                    visible = {p.track_id for p in people}
                    for tid in [t for t in results if t not in visible]:   # 화면을 떠난 사람은 지움
                        del results[tid]
                # 고개 돌리기 안내(A)가 얼굴 안내보다 먼저. 환영·카운터 안내 중에는 덮지 않음
                prompt = turn_prompt(people, now, judge)
                if prompt and (banner is None or banner[1] in ("hint", "turn")):
                    banner = (prompt, "turn", now + 0.3)
                # 다른 안내가 없을 때만 얼굴 안내 (이미 얼굴 안내 중이면 1초 연장)
                elif (banner is None or banner[1] == "hint") and needs_hint(people, now, judge):
                    banner = (MSG_FACE_HINT, "hint", now + HINT_SEC)

            if banner and now > banner[2]:
                banner = None
            # 화면에 보일 것만 거울 반전 (인식·기록은 원본 frame). 배너 글자는 뒤집은 뒤에 그림
            if debug:
                shown = draw_debug(frame, people, results, detector.last_animals, fps, mirrored=MIRROR)
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
            elif key == ord("l") and not fake:
                reload_faces(face, announce)
            elif key in FAKE_KEYS and fake:
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
                        help="A의 얼굴 인식 대신 키보드 r/u/k로 판정 (기록은 콘솔에만, DB 저장 안 함)")
    args = parser.parse_args()
    run(fake=args.fake)


if __name__ == "__main__":
    main()
