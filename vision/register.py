"""얼굴 등록 명령줄 도구 (A 담당, 5.3).

python -m vision.register 홍길동                  웹캠으로 5장을 찍어 등록
python -m vision.register 홍길동 --folder 경로    폴더 안의 사진으로 등록
python -m vision.register 홍길동 --add            이미 등록된 사람에게 사진 추가
python -m vision.register --list                  등록자 목록
python -m vision.register --delete 홍길동         등록자 삭제 (DB + data/faces/홍길동/)

C의 database.py(db/database.py 또는 루트)가 있으면 그 함수만 호출한다 (vision/facedb.py).
아직 없으면 experiments/known_faces.npz에 임시로 저장한다 (vision.face가 이 파일을 읽음).
"""
import argparse
import os
import shutil
import sys
import time

import cv2
import numpy as np

import config
from common.imageio import imread_unicode, imwrite_unicode
from vision import face, facedb

SHOTS = [
    ("FRONT",            "정면을 보고, 평소 조명"),
    ("LEFT 15-30 DEG",   "고개를 왼쪽으로 15~30도 돌리기"),
    ("RIGHT 15-30 DEG",  "고개를 오른쪽으로 15~30도 돌리기"),
    ("FRONT DARK",       "정면, 조명을 조금 어둡게"),
    ("FRONT FAR",        "정면, 카메라에서 조금 멀리"),
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ───────────────────────── 저장소 ─────────────────────────

def _get_store() -> str:
    """C의 DB를 쓸 수 있으면 "DB (...)", 아니면 "npz". 저장은 모두 vision/facedb.py가 한다."""
    kind = facedb.backend()
    if kind == "npz":
        print(f"[register] C의 database.py가 아직 없어 임시 파일({facedb.NPZ_PATH})에 저장합니다.")
    else:
        print(f"[register] {kind}, DB 파일 {config.DB_PATH}, model_key={facedb.model_key()}")
    return kind


# ───────────────────────── 사진 저장 ─────────────────────────

def _faces_dir(name: str) -> str:
    return os.path.join(config.FACES_DIR, name)


def _next_number(name: str) -> int:
    folder = _faces_dir(name)
    if not os.path.isdir(folder):
        return 1
    nums = [int(os.path.splitext(f)[0]) for f in os.listdir(folder)
            if os.path.splitext(f)[0].isdigit()]
    return max(nums) + 1 if nums else 1


def _save_photo(name: str, number: int, image: np.ndarray) -> str:
    """data/faces/<이름>/<번호>.jpg에 저장하고 DATA_DIR 기준 상대 경로를 돌려준다."""
    abs_path = os.path.join(_faces_dir(name), f"{number}.jpg")
    if not imwrite_unicode(abs_path, image):
        raise IOError(f"사진 저장 실패: {abs_path}")
    return os.path.relpath(abs_path, config.DATA_DIR).replace("\\", "/")


# ───────────────────────── 사진 모으기 ─────────────────────────

def _capture_webcam(camera_index: int) -> list[np.ndarray] | None:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"카메라 {camera_index}번을 열 수 없습니다.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    shots: list[np.ndarray] = []
    flash_until, flash_text, flash_color = 0.0, "", (0, 0, 0)
    print("스페이스바: 촬영,  ESC/q: 취소")
    print(f"[1/{len(SHOTS)}] {SHOTS[0][1]}")
    try:
        while len(shots) < len(SHOTS):
            ok, frame = cap.read()
            if not ok:
                print("프레임을 읽지 못했습니다.")
                return None
            view = frame.copy()
            faces = face.detect_faces(frame)
            for f in faces:
                x, y, w, h = [int(v) for v in f[:4]]
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 0) if len(faces) == 1 else (0, 0, 255), 2)
            i = len(shots)
            cv2.putText(view, f"{i + 1}/{len(SHOTS)} {SHOTS[i][0]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(view, f"faces: {len(faces)}   SPACE=shoot  ESC=cancel", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            if time.time() < flash_until:
                cv2.putText(view, flash_text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, flash_color, 2)
            cv2.imshow("register", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("취소했습니다.")
                return None
            if key == ord(" "):
                n = len(face.detect_faces(frame))
                if n != 1:
                    print(f"  얼굴이 {n}개 보입니다. 얼굴이 정확히 1개일 때 다시 찍어 주세요.")
                    flash_until, flash_text, flash_color = time.time() + 1.5, f"REJECTED ({n} faces)", (0, 0, 255)
                    continue
                shots.append(frame.copy())
                flash_until, flash_text, flash_color = time.time() + 1.0, "OK", (0, 255, 0)
                print(f"  {len(shots)}번째 사진 OK")
                if len(shots) < len(SHOTS):
                    print(f"[{len(shots) + 1}/{len(SHOTS)}] {SHOTS[len(shots)][1]}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return shots


def _load_folder(folder: str) -> list[np.ndarray]:
    if not os.path.isdir(folder):
        print(f"폴더가 없습니다: {folder}")
        return []
    images = []
    for fn in sorted(os.listdir(folder)):
        if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
            continue
        img = imread_unicode(os.path.join(folder, fn))
        if img is None:
            print(f"  건너뜀 (읽을 수 없음): {fn}")
            continue
        n = len(face.detect_faces(img))
        if n != 1:
            print(f"  건너뜀 (얼굴 {n}개): {fn}")
            continue
        print(f"  사용: {fn}")
        images.append(img)
    return images


# ───────────────────────── 명령 ─────────────────────────

def _ask_yes(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() == "y"
    except EOFError:
        return False


def cmd_register(name: str, folder: str | None, add: bool, camera: int) -> int:
    kind = _get_store()
    exists = bool(facedb.find_ids(name))
    if exists and not add:
        if not _ask_yes(f"'{name}'은(는) 이미 등록되어 있습니다. 사진을 추가할까요? (y/n): "):
            print("등록하지 않았습니다.")
            return 1

    face._load_models()
    face.set_use_clahe(config.USE_CLAHE)       # 인식과 같은 전처리
    images = _load_folder(folder) if folder else _capture_webcam(camera)
    if not images:
        print("등록할 사진이 없습니다.")
        return 1

    start = _next_number(name)
    photo_paths, embeddings = [], []
    for k, img in enumerate(images):
        emb = face.extract_embedding(img)
        if emb is None:
            print(f"  {k + 1}번째 사진에서 특징을 뽑지 못해 건너뜁니다.")
            continue
        photo_paths.append(_save_photo(name, start + len(photo_paths), img))
        embeddings.append(emb)
    if not embeddings:
        print("특징 벡터를 하나도 만들지 못했습니다.")
        return 1

    pid = facedb.register(name, photo_paths, embeddings)
    print(f"\n등록 완료 ({kind}): 등록자 번호 {pid}, '{name}', 이번에 추가한 벡터 {len(embeddings)}개")
    print("메인 프로그램을 다시 시작하거나 reload_known_faces()를 호출해야 반영됩니다.")
    return 0


def cmd_list() -> int:
    kind = _get_store()
    persons = facedb.list_persons()
    if not persons:
        print(f"등록자가 없습니다 ({kind}).")
        return 0
    print(f"등록자 {len(persons)}명 ({kind})")
    print(f"{'번호':>4}  {'이름':<10} {'벡터':>4}  {'등록일':<19}  대표 사진")
    for p in persons:
        print(f"{p['id']:>4}  {p['name']:<10} {p.get('embedding_count', 0):>4}  "
              f"{str(p.get('created_at', '-')):<19}  {p.get('photo_path') or ''}")
    return 0


def cmd_delete(name: str) -> int:
    kind = _get_store()
    ids = facedb.find_ids(name)
    if not ids:
        print(f"'{name}'은(는) 등록되어 있지 않습니다.")
        return 1
    if not _ask_yes(f"'{name}'의 등록 정보와 사진 폴더를 모두 삭제합니다. 계속하려면 y: "):
        print("삭제하지 않았습니다.")
        return 1
    if len(ids) > 1:
        print(f"  같은 이름의 등록자가 {len(ids)}명({ids}) 있어 모두 삭제합니다.")
    facedb.delete(name)
    folder = _faces_dir(name)
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    print(f"'{name}' 삭제 완료 ({kind}, {folder} 폴더 포함).")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="얼굴 등록 도구 (A)")
    ap.add_argument("name", nargs="?", help="등록할 사람 이름")
    ap.add_argument("--folder", help="이 폴더 안의 사진으로 등록")
    ap.add_argument("--add", action="store_true", help="이미 등록된 사람에게 사진 추가")
    ap.add_argument("--list", action="store_true", help="등록자 목록")
    ap.add_argument("--delete", metavar="NAME", help="등록자 삭제")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = ap.parse_args(argv)

    if args.list:
        return cmd_list()
    if args.delete:
        return cmd_delete(args.delete)
    if not args.name:
        ap.print_help()
        return 1
    return cmd_register(args.name, args.folder, args.add, args.camera)


if __name__ == "__main__":
    sys.exit(main())
