"""판정 기준값 실험: 거리 수집 (A 담당, 인수인계서 5.4).

웹캠으로 얼굴을 찾은 프레임 20개를 인식해 거리를 CSV에 추가한다.

실행 (프로젝트 루트에서):
    python experiments/collect_distances.py --who 홍길동 --condition front_bright
    python experiments/collect_distances.py --who unknown --condition front_bright   # 미등록자 역할

조작: 스페이스바 = 측정 시작, q / Esc = 취소 (취소하면 아무것도 저장하지 않음)

front_dark_clahe 조건에서는 등록 벡터를 data/faces/의 등록 사진에서 CLAHE를 켜고 다시 뽑아
"CLAHE 켠 얼굴 vs CLAHE 켠 등록 얼굴"로 공정하게 비교한다. 저장된 등록은 바꾸지 않는다.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 프로젝트 루트

import cv2
import numpy as np

from common.imageio import imread_unicode
from config import CAMERA_INDEX, FACE_INTERVAL, FACES_DIR, USE_CLAHE
from vision import face as F

CONDITIONS = ["front_bright", "side_bright", "front_dark", "front_dark_clahe", "far"]
CLAHE_CONDITION = "front_dark_clahe"
UNKNOWN_WHO = "unknown"
CSV_PATH = os.path.join("experiments", "distances.csv")
CSV_COLUMNS = ["time", "who", "condition", "nearest_name", "nearest_distance",
               "self_distance", "min_other_distance"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ──────────────────────────────────────────────
# 특징 벡터 (CLAHE 켜고 끄기를 직접 정함)
# ──────────────────────────────────────────────
def _embed(image: np.ndarray, face_row: np.ndarray, clahe: bool) -> np.ndarray:
    """face.py와 같은 방법(alignCrop → feature)으로 벡터를 뽑되, CLAHE 여부만 직접 정한다."""
    aligned = F._recognizer.alignCrop(image, face_row)
    if clahe:
        aligned = F._apply_clahe(aligned)
    return np.asarray(F._recognizer.feature(aligned), dtype=np.float32).ravel().copy()


def _known_from_photos(clahe: bool) -> list[tuple[int, str, np.ndarray]]:
    """data/faces/<이름>/ 의 등록 사진에서 등록 벡터를 다시 뽑는다."""
    ids = {p["name"]: p["id"] for p in F._get_store().list_persons()}
    known = []
    for name, pid in ids.items():
        folder = os.path.join(FACES_DIR, name)
        if not os.path.isdir(folder):
            print(f"  ⚠️ {name}: 등록 사진 폴더가 없어 이 조건에서는 빠집니다 ({folder})")
            continue
        count = 0
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            image = imread_unicode(os.path.join(folder, fname))
            if image is None:
                continue
            row = F._largest(F._detect_faces(image))
            if row is None:
                continue
            known.append((pid, name, _embed(image, row, clahe)))
            count += 1
        print(f"  {name}: 사진 {count}장에서 벡터를 다시 뽑았습니다 (CLAHE {'켬' if clahe else '끔'})")
    return known


# ──────────────────────────────────────────────
# 거리 계산
# ──────────────────────────────────────────────
def _per_person_min(embedding: np.ndarray, known) -> dict[str, float]:
    """등록자별 가장 가까운 거리 {이름: 거리}."""
    out: dict[str, float] = {}
    for _, name, vec in known:
        d = F.embedding_distance(embedding, vec)
        if d < out.get(name, float("inf")):
            out[name] = d
    return out


def _row(who: str, condition: str, dists: dict[str, float]) -> dict:
    fmt = lambda v: f"{v:.4f}" if v is not None and np.isfinite(v) else ""
    nearest_name, nearest = min(dists.items(), key=lambda kv: kv[1]) if dists else ("", None)
    self_d = dists.get(who) if who != UNKNOWN_WHO else None
    others = [d for n, d in dists.items() if n != who]
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "who": who, "condition": condition,
            "nearest_name": nearest_name, "nearest_distance": fmt(nearest),
            "self_distance": fmt(self_d), "min_other_distance": fmt(min(others) if others else None)}


def _append_csv(rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    new_file = not os.path.isfile(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:   # 엑셀에서 한글이 안 깨지게
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────
# 측정
# ──────────────────────────────────────────────
def _measure(who: str, condition: str, frames: int, known, clahe: bool,
             camera_index: int = CAMERA_INDEX) -> list[dict] | None:
    cap = F.open_camera(camera_index)
    if cap is None:
        return None

    window = "collect distances"
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows: list[dict] = []
    measuring, frame_no, last_d = False, 0, None
    print("\n자세·조명을 맞춘 뒤 스페이스바를 누르면 측정을 시작합니다 (q 또는 Esc: 취소)")
    try:
        while len(rows) < frames:
            ok, frame = cap.read()
            if not ok:
                print("카메라에서 영상을 읽지 못했습니다")
                return None
            frame_no += 1

            faces = F._detect_faces(frame)
            view = frame.copy()
            for f in faces:
                x, y, w, h = (int(v) for v in f[:4])
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 0) if len(faces) == 1 else (0, 0, 255), 2)

            # 측정: FACE_INTERVAL 프레임마다, 얼굴이 정확히 1개일 때만
            if measuring and frame_no % FACE_INTERVAL == 0 and len(faces) == 1:
                dists = _per_person_min(_embed(frame, faces[0], clahe), known)
                row = _row(who, condition, dists)
                rows.append(row)
                last_d = row["self_distance"] or row["nearest_distance"]

            status = f"{len(rows)}/{frames}" + (f"  d={last_d}" if last_d else "") if measuring else "SPACE: start"
            cv2.putText(view, f"{condition}  {status}", (10, 30), font, 0.7, (0, 255, 255), 2)
            cv2.putText(view, f"faces {len(faces)}   q: cancel", (10, view.shape[0] - 12),
                        font, 0.6, (0, 255, 255), 2)
            cv2.imshow(window, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("측정을 취소했습니다. 아무것도 저장하지 않았습니다.")
                return None
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                print("창을 닫아 측정을 취소했습니다. 아무것도 저장하지 않았습니다.")
                return None
            if key == ord(" ") and not measuring:
                measuring = True
                print("측정을 시작합니다...")
        return rows
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _print_summary(rows: list[dict], who: str) -> None:
    def nums(col):
        return [float(r[col]) for r in rows if r[col] != ""]
    self_d, other_d, nearest = nums("self_distance"), nums("min_other_distance"), nums("nearest_distance")
    print(f"\n📊 요약 ({len(rows)}프레임)")
    if who != UNKNOWN_WHO and self_d:
        print(f"  본인 거리       평균 {np.mean(self_d):.4f}   최소 {min(self_d):.4f}   최대 {max(self_d):.4f}")
    if who == UNKNOWN_WHO and nearest:
        print(f"  가장 가까운 등록자 거리  평균 {np.mean(nearest):.4f}   최소 {min(nearest):.4f}")
    if other_d:
        print(f"  타인 거리(최소) 평균 {np.mean(other_d):.4f}   최소 {min(other_d):.4f}")
    names = [r["nearest_name"] for r in rows if r["nearest_name"]]
    if names:
        counts = {n: names.count(n) for n in sorted(set(names))}
        print("  가장 가까운 사람: " + ", ".join(f"{n} {c}회" for n, c in counts.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="기준값 실험: 웹캠으로 거리를 모아 CSV에 추가")
    parser.add_argument("--who", required=True, help=f"카메라 앞에 선 사람 이름. 미등록자 역할이면 {UNKNOWN_WHO}")
    parser.add_argument("--condition", required=True, choices=CONDITIONS, help="촬영 조건")
    parser.add_argument("--frames", type=int, default=20, help="모을 프레임 수 (기본 20)")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="카메라 번호")
    args = parser.parse_args(argv)
    who = args.who.strip()

    F.init_recognizer()
    clahe = args.condition == CLAHE_CONDITION or USE_CLAHE
    if args.condition == CLAHE_CONDITION and not USE_CLAHE:
        print("CLAHE 조건: 등록 사진에서 CLAHE를 켜고 등록 벡터를 다시 뽑습니다.")
        known = _known_from_photos(clahe=True)
    else:
        known = list(F._known)

    names = sorted({n for _, n, _ in known})
    if not names:
        print("등록된 얼굴이 없습니다. 먼저 python -m vision.register 이름 으로 등록해 주세요.")
        return 1
    if who != UNKNOWN_WHO and who not in names:
        print(f"'{who}'님은 등록자가 아닙니다 (등록자: {', '.join(names)}).")
        print(f"미등록자 역할이면 --who {UNKNOWN_WHO} 를 쓰세요. 오타인지도 확인해 주세요.")
        return 1

    print(f"사람: {who}   조건: {args.condition}   CLAHE: {'켬' if clahe else '끔'}   등록자: {', '.join(names)}")
    rows = _measure(who, args.condition, args.frames, known, clahe, args.camera)
    if not rows:
        return 1
    _append_csv(rows)
    _print_summary(rows, who)
    print(f"\n✅ {len(rows)}줄을 {CSV_PATH}에 추가했습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
