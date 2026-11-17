"""얼굴 등록 명령줄 도구 (A 담당, 인수인계서 5.3).

python -m vision.register 홍길동                 웹캠으로 5장을 찍어 등록
python -m vision.register 홍길동 --folder 경로   폴더 안의 사진으로 등록
python -m vision.register 홍길동 --add           이미 등록된 사람에게 사진 추가
python -m vision.register --list                 등록자 목록 보기
python -m vision.register --delete 홍길동        등록자 삭제 (DB와 data/faces/홍길동/ 폴더 모두)

옵션: --camera 번호 (기본 config.CAMERA_INDEX)

저장소: C의 database.py(db/database.py 또는 루트 database.py)가 있으면 C의 함수,
        없으면 experiments/known_faces.npz (임시). 연결은 vision/face.py가 맡는다.
등록 후에는 메인 프로그램을 다시 시작하거나 reload_known_faces()를 호출해야 반영된다.
"""
import argparse
import os
import shutil
import sys

import cv2
import numpy as np

from common.imageio import imread_unicode, imwrite_unicode
from config import CAMERA_INDEX, DATA_DIR, FACES_DIR, USE_CLAHE
from vision.face import (_detect_faces, _get_store, _store_label, extract_embedding,
                         model_key, open_camera)

# (화면에 쓸 영문, 콘솔에 쓸 한글 안내) — 인수인계서 5.3 촬영 조건
POSES = [
    ("FRONT", "정면을 보고, 평소 조명에서 찍어요"),
    ("LEFT 15-30", "얼굴을 왼쪽으로 15~30° 돌려 주세요"),
    ("RIGHT 15-30", "얼굴을 오른쪽으로 15~30° 돌려 주세요"),
    ("FRONT DARK", "정면, 조명을 조금 어둡게 해 주세요"),
    ("FRONT FAR", "정면, 카메라에서 조금 멀리 서 주세요"),
]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
BAD_NAME_CHARS = set('\\/:*?"<>|')


# ──────────────────────────────────────────────
# 보조 함수
# ──────────────────────────────────────────────
def _ask_yes(question: str) -> bool:
    return input(f"{question} (y/N): ").strip().lower() == "y"


def _person_dir(name: str) -> str:
    return os.path.join(FACES_DIR, name)


def _rel_to_data(path: str) -> str:
    """DATA_DIR 기준 상대 경로, 슬래시 사용 (3.8). 예: faces/홍길동/1.jpg"""
    return os.path.relpath(path, DATA_DIR).replace("\\", "/")


def _next_number(name: str) -> int:
    """data/faces/<이름>/ 안의 <번호>.jpg 중 가장 큰 번호 + 1 (사진 추가 시 덮어쓰지 않게)."""
    folder = _person_dir(name)
    if not os.path.isdir(folder):
        return 1
    numbers = [int(os.path.splitext(f)[0]) for f in os.listdir(folder)
               if os.path.splitext(f)[0].isdigit()]
    return max(numbers, default=0) + 1


def _find_person(name: str) -> dict | None:
    for p in _get_store().list_persons():
        if p["name"] == name:
            return p
    return None


# ──────────────────────────────────────────────
# 사진 모으기: 웹캠 / 폴더
# ──────────────────────────────────────────────
def _capture_webcam(camera_index: int) -> list[np.ndarray] | None:
    """웹캠으로 POSES 순서대로 찍는다. 스페이스바: 찍기, q/Esc: 취소. 취소하면 None."""
    cap = open_camera(camera_index)
    if cap is None:
        return None

    window = "register"
    font = cv2.FONT_HERSHEY_SIMPLEX
    shots: list[np.ndarray] = []
    flash_text, flash_color, flash_left = "", (0, 0, 0), 0   # 찍은 직후 잠깐 띄우는 문구
    total = len(POSES)
    print(f"\n[1/{total}] {POSES[0][1]} → 스페이스바를 누르세요 (q 또는 Esc: 취소)")
    try:
        while len(shots) < total:
            ok, frame = cap.read()
            if not ok:
                print("카메라에서 영상을 읽지 못했습니다")
                return None

            faces = _detect_faces(frame)
            view = frame.copy()                     # 저장할 원본에는 그리지 않는다
            for f in faces:
                x, y, w, h = (int(v) for v in f[:4])
                color = (0, 200, 0) if len(faces) == 1 else (0, 0, 255)
                cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)

            step = len(shots)
            cv2.putText(view, f"{step + 1}/{total} {POSES[step][0]}", (10, 30), font, 0.9, (0, 255, 255), 2)
            cv2.putText(view, f"faces {len(faces)}   SPACE: capture   q: cancel",
                        (10, view.shape[0] - 12), font, 0.6, (0, 255, 255), 2)
            if flash_left > 0:
                cv2.putText(view, flash_text, (10, 70), font, 0.9, flash_color, 2)
                flash_left -= 1

            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("등록을 취소했습니다. 아무것도 저장하지 않았습니다.")
                return None
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                print("창을 닫아 등록을 취소했습니다. 아무것도 저장하지 않았습니다.")
                return None
            if key == ord(" "):
                if len(faces) != 1:
                    print(f"  ✗ 얼굴이 정확히 1개여야 합니다 (지금 {len(faces)}개). 다시 찍어 주세요.")
                    flash_text, flash_color, flash_left = f"RETRY: {len(faces)} faces", (0, 0, 255), 30
                    continue
                shots.append(frame.copy())
                print(f"  ✓ {len(shots)}번째 사진을 찍었습니다")
                flash_text, flash_color, flash_left = "OK", (0, 200, 0), 15
                if len(shots) < total:
                    print(f"\n[{len(shots) + 1}/{total}] {POSES[len(shots)][1]} → 스페이스바를 누르세요")
        return shots
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _load_folder(folder: str) -> list[np.ndarray]:
    """폴더 안의 사진 중 얼굴이 정확히 1개인 것만 돌려준다."""
    if not os.path.isdir(folder):
        print(f"폴더가 없습니다: {folder}")
        return []
    images = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        image = imread_unicode(os.path.join(folder, fname))
        if image is None:
            print(f"  ✗ {fname}: 읽을 수 없어 건너뜁니다")
            continue
        n = len(_detect_faces(image))
        if n != 1:
            print(f"  ✗ {fname}: 얼굴이 {n}개라 건너뜁니다 (정확히 1개여야 함)")
            continue
        print(f"  ✓ {fname}")
        images.append(image)
    return images


# ──────────────────────────────────────────────
# 명령
# ──────────────────────────────────────────────
def cmd_register(name: str, folder: str | None, add: bool, camera_index: int = CAMERA_INDEX) -> int:
    if not name or any(c in BAD_NAME_CHARS for c in name):
        print(f"이름에 쓸 수 없는 글자가 있습니다: {name}  (\\ / : * ? \" < > | 는 안 됩니다)")
        return 1

    existing = _find_person(name)
    if existing and not add:
        print(f"'{name}'님은 이미 등록되어 있습니다 (id {existing['id']}, 벡터 {existing['embedding_count']}개).")
        if not _ask_yes("사진을 추가할까요?"):
            print("취소했습니다.")
            return 0
    if add and not existing:
        print(f"'{name}'님은 등록되어 있지 않아 새로 등록합니다.")

    print(f"저장소: {_store_label()}   model_key = {model_key()}   USE_CLAHE = {USE_CLAHE}")
    if folder:
        print(f"폴더에서 사진을 읽습니다: {folder}")
        images = _load_folder(folder)
    else:
        print("웹캠으로 사진을 찍습니다. 본인 동의를 받은 팀원만 등록해 주세요.")
        images = _capture_webcam(camera_index)
    if not images:
        print("등록할 사진이 없습니다.")
        return 1

    # 사진 저장 → 벡터 추출
    number = _next_number(name)
    photo_paths, embeddings = [], []
    for image in images:
        embedding = extract_embedding(image)
        if embedding is None:
            print("  ✗ 벡터를 뽑지 못한 사진이 있어 건너뜁니다")
            continue
        path = os.path.join(_person_dir(name), f"{number}.jpg")
        if not imwrite_unicode(path, image):
            print(f"  ✗ 사진을 저장하지 못했습니다: {path}")
            continue
        photo_paths.append(_rel_to_data(path))
        embeddings.append(embedding)
        number += 1

    if not embeddings:
        print("벡터를 하나도 뽑지 못해 등록하지 않았습니다.")
        return 1

    person_id = _get_store().register(name, photo_paths, embeddings)
    total = next((p["embedding_count"] for p in _get_store().list_persons() if p["id"] == person_id), "?")
    print(f"\n✅ 등록 완료: {name} (id {person_id}), 이번에 추가한 벡터 {len(embeddings)}개, 전체 {total}개")
    for p in photo_paths:
        print(f"   - {p}")
    print("※ 메인 프로그램을 다시 시작하거나 reload_known_faces()를 호출해야 반영됩니다.")
    return 0


def cmd_list() -> int:
    persons = _get_store().list_persons()
    print(f"저장소: {_store_label()}")
    if not persons:
        print("등록자가 없습니다.")
        return 0
    print(f"{'id':>4}  {'이름':<10} {'벡터':>4}  {'등록일':<19}  대표 사진")
    for p in persons:
        print(f"{p['id']:>4}  {p['name']:<10} {p['embedding_count']:>4}  "
              f"{p.get('created_at') or '-':<19}  {p.get('photo_path') or '-'}")
    print(f"총 {len(persons)}명")
    return 0


def cmd_delete(name: str) -> int:
    existing = _find_person(name)
    folder = _person_dir(name)
    if not existing and not os.path.isdir(folder):
        print(f"'{name}'님은 등록되어 있지 않습니다.")
        return 1
    if existing:
        print(f"'{name}'님 (id {existing['id']}, 벡터 {existing['embedding_count']}개)을 삭제합니다.")
    if os.path.isdir(folder):
        print(f"사진 폴더도 함께 지웁니다: {folder}")
    if not _ask_yes("정말 삭제할까요? 되돌릴 수 없습니다."):
        print("취소했습니다.")
        return 0

    if existing:
        same_name = [p["id"] for p in _get_store().list_persons() if p["name"] == name]
        if len(same_name) > 1:
            print(f"같은 이름의 등록자가 {len(same_name)}명 있어 모두 삭제합니다 (id {same_name}).")
        count = _get_store().delete(name)
        print(f"저장소에서 {count}명을 삭제했습니다." if count else "저장소에서 찾지 못했습니다.")
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        print("사진 폴더를 지웠습니다.")
    print("※ 메인 프로그램을 다시 시작하거나 reload_known_faces()를 호출해야 반영됩니다.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="얼굴 등록 도구")
    parser.add_argument("name", nargs="?", help="등록할 사람 이름")
    parser.add_argument("--folder", help="이 폴더 안의 사진으로 등록")
    parser.add_argument("--add", action="store_true", help="이미 등록된 사람에게 사진 추가")
    parser.add_argument("--list", action="store_true", help="등록자 목록 보기")
    parser.add_argument("--delete", metavar="NAME", help="등록자 삭제")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="카메라 번호")
    args = parser.parse_args(argv)

    if args.list:
        return cmd_list()
    if args.delete:
        return cmd_delete(args.delete)
    if args.name:
        return cmd_register(args.name.strip(), args.folder, args.add, args.camera)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
