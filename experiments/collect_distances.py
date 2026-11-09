"""웹캠으로 20프레임을 인식해 거리를 CSV에 추가한다 (5.4).

python experiments/collect_distances.py --who 홍길동 --condition front_bright
python experiments/collect_distances.py --who unknown --condition front_dark   # 미등록자 역할

CSV 컬럼: time, who, condition, nearest_name, nearest_distance, self_distance, min_other_distance
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

import config
from common.types import PersonBox
from vision import face

CONDITIONS = ["front_bright", "side_bright", "front_dark", "front_dark_clahe", "far"]
COLUMNS = ["time", "who", "condition", "nearest_name", "nearest_distance",
           "self_distance", "min_other_distance"]
DEFAULT_CSV = os.path.join("experiments", "distances.csv")


def _fmt(v):
    return "" if v is None else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", required=True, help="카메라 앞에 선 사람 이름. 미등록자 역할이면 unknown")
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = ap.parse_args()

    face.init_recognizer()
    clahe = args.condition.endswith("_clahe") or config.USE_CLAHE
    face.set_use_clahe(clahe)
    registered = sorted(set(face._known_names))
    if not registered:
        print("등록자가 없습니다. 먼저 python -m vision.register 로 등록하세요.")
        return 1
    if args.who != "unknown" and args.who not in registered:
        print(f"'{args.who}'은(는) 등록자가 아닙니다. 등록자: {registered}  (미등록자 역할이면 --who unknown)")
        return 1
    print(f"대상 {args.who}, 조건 {args.condition}, CLAHE {'켬' if clahe else '끔'}, 등록자 {registered}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"카메라 {args.camera}번을 열 수 없습니다.")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    print("자세를 잡고 스페이스바를 누르면 측정을 시작합니다. ESC로 취소.")
    started = False
    rows = []
    while len(rows) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        view = frame.copy()
        if started:
            r = face.recognize(frame, PersonBox(track_id=1, bbox=(0, 0, w, h), conf=1.0))
            if r.face_found:
                per = face.nearest_per_person(r.embedding)
                nearest_name, (_, nearest_d) = min(per.items(), key=lambda kv: kv[1][1])
                self_d = per[args.who][1] if args.who in per else None
                others = [d for n, (_, d) in per.items() if n != args.who]
                rows.append({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "who": args.who,
                    "condition": args.condition,
                    "nearest_name": nearest_name,
                    "nearest_distance": _fmt(nearest_d),
                    "self_distance": _fmt(self_d),
                    "min_other_distance": _fmt(min(others) if others else None),
                })
                print(f"  {len(rows):>2}/{args.frames}  가장 가까운 {nearest_name} {nearest_d:.3f}"
                      f"  본인 {_fmt(self_d) or '-'}  타인 최소 {rows[-1]['min_other_distance'] or '-'}")
            msg = f"{len(rows)}/{args.frames}" + ("" if r.face_found else "  NO FACE")
        else:
            msg = "SPACE = start"
        cv2.putText(view, f"{args.condition}  {msg}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("collect", view)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("취소했습니다. 저장하지 않습니다.")
            rows = []
            break
        if key == ord(" "):
            started = True
    cap.release()
    cv2.destroyAllWindows()

    if not rows:
        return 1
    new_file = not os.path.exists(args.csv)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)}줄을 {args.csv}에 추가했습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
