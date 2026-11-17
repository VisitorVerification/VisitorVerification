"""얼굴 검출·특징 추출·비교 (A 담당).

OpenCV 내장 YuNet(얼굴 검출) + SFace(특징 추출)를 쓴다.

공통 명세 함수 (3.5)
- init_recognizer(), reload_known_faces()
- extract_embedding(), compare(), embedding_distance(), recognize()

A가 더 제공하는 함수
- model_key()  : C의 DB에 넘기는 모델·전처리 이름 (B가 save_visit, recent_unknown_embeddings에 사용)
- open_camera(): 카메라 열기 (Windows CAP_DSHOW 먼저 시도)

등록 벡터 저장소
- C의 database.py(sqlite_1 브랜치)가 db/database.py 또는 루트 database.py에 있으면 C의 함수를 쓴다.
- 없으면 experiments/known_faces.npz를 임시로 쓴다 (_NpzStore).

python -m vision.face --demo : 웹캠으로 확인 (프레임 전체를 사람 1명으로 보고 recognize() 시험)
"""
import argparse
import importlib
import os
import shutil
import tempfile
import time

import cv2
import numpy as np

from common.types import PersonBox, RecognitionResult
from config import (CAMERA_INDEX, DB_PATH, FRAME_HEIGHT, FRAME_WIDTH, MATCH_THRESHOLD,
                    MODELS_DIR, USE_CLAHE)

# ── 모델 파일 (models/ 폴더에 직접 받아 둔다) ──
DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"
FACE_MODEL_KEY = "sface_2021dec"   # C의 DB에 저장할 때 쓰는 모델 이름 (CLAHE면 "_clahe"가 붙음)

# ── YuNet 검출 설정 (OpenCV 예제 기본값) ──
DETECT_SCORE = 0.9       # 이 신뢰도 이상만 얼굴로 본다
DETECT_NMS = 0.3
DETECT_TOP_K = 5000

# ── recognize() 설정 (인수인계서 5.2) ──
BBOX_MARGIN = 0.10       # 사람 박스를 이 비율만큼 넓혀 자른다
MIN_FACE_WIDTH = 40      # 얼굴 폭이 이보다 작으면 얼굴 못 찾음으로 본다 (픽셀)

# ── CLAHE 설정 (인수인계서 5.4) ──
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# ── 임시 등록 벡터 파일 (C의 DB가 오기 전까지) ──
KNOWN_NPZ = os.path.join("experiments", "known_faces.npz")
EMBEDDING_DIM = 128      # SFace 벡터 길이 (빈 npz를 만들 때만 씀)

_detector = None
_recognizer = None
_detector_size = None                               # 마지막으로 setInputSize()한 (w, h)
_known: list[tuple[int, str, np.ndarray]] = []      # [(person_id, name, embedding), ...]
_known_loaded = False


# ──────────────────────────────────────────────
# 등록 벡터 저장소: C의 DB 또는 임시 npz
#
# 두 저장소는 아래 A 내부 함수 모양을 똑같이 가진다.
#   label                           화면·콘솔 표시용 이름
#   load() -> [(person_id, name, float32 벡터)]   지금 model_key()의 벡터만
#   list_persons() -> [{"id", "name", "photo_path", "created_at", "embedding_count"}]
#   register(name, photo_paths, embeddings) -> person_id   같은 이름이면 벡터 추가
#   delete(name) -> 지운 사람 수
# ──────────────────────────────────────────────
def model_key() -> str:
    """등록·인식에 쓰는 모델·전처리 이름. C의 DB 함수에 넘긴다 (B는 save_visit 등에 사용).
    CLAHE를 켜면 이름이 바뀌어 예전 벡터와 섞이지 않는다."""
    return FACE_MODEL_KEY + ("_clahe" if USE_CLAHE else "")


class _CStore:
    """C의 database.py (sqlite_1 브랜치) 함수를 부른다. DB 경로는 항상 config.DB_PATH."""

    def __init__(self, module, module_name: str):
        self.db = module
        self.label = f"DB (C: {module_name})"
        self.db.init_db(DB_PATH)          # 없으면 만들고, 있으면 그대로 둔다

    def load(self) -> list[tuple[int, str, np.ndarray]]:
        rows = self.db.load_embeddings(model_key=model_key(), db_path=DB_PATH)
        return [(int(r["person_id"]), str(r["person_name"]), np.asarray(r["embedding"], dtype=np.float32))
                for r in rows]

    def list_persons(self) -> list[dict]:
        return self.db.list_persons(db_path=DB_PATH)

    def _ids(self, name: str) -> list[int]:
        """이름이 같은 등록자 번호 (C의 DB는 이름 중복을 막지 않아 여러 개일 수 있다)."""
        return [int(p["id"]) for p in self.list_persons() if p["name"] == name]

    def register(self, name: str, photo_paths: list[str], embeddings: list[np.ndarray]) -> int:
        vectors = [np.asarray(e, dtype=np.float32).ravel().tolist() for e in embeddings]
        ids = self._ids(name)
        if ids:                            # 이미 있는 사람 → 벡터만 추가
            self.db.add_embeddings(ids[0], vectors, model_key=model_key(), db_path=DB_PATH)
            return ids[0]
        return int(self.db.register_person(name, vectors, model_key=model_key(),
                                           photo_path=photo_paths[0] if photo_paths else None,
                                           db_path=DB_PATH))

    def delete(self, name: str) -> int:
        return sum(1 for pid in self._ids(name) if self.db.delete_person(pid, db_path=DB_PATH))


class _NpzStore:
    """C의 database.py가 없을 때 쓰는 임시 저장소 (experiments/known_faces.npz)."""
    label = "npz (임시)"
    path = KNOWN_NPZ

    def _read(self) -> list[dict]:
        """벡터 한 개당 한 줄: {"person_id", "name", "embedding", "photo_path", "created_at", "model_key"}"""
        if not os.path.isfile(self.path):
            return []
        with np.load(self.path, allow_pickle=False) as data:
            n = len(data["person_ids"])
            photos = data["photo_paths"] if "photo_paths" in data else np.array([""] * n)
            created = data["created_at"] if "created_at" in data else np.array([""] * n)
            keys = data["model_keys"] if "model_keys" in data else np.array([FACE_MODEL_KEY] * n)
            return [{"person_id": int(data["person_ids"][i]), "name": str(data["names"][i]),
                     "embedding": np.asarray(data["embeddings"][i], dtype=np.float32),
                     "photo_path": str(photos[i]), "created_at": str(created[i]),
                     "model_key": str(keys[i])}
                    for i in range(n)]

    def _write(self, rows: list[dict]) -> None:
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        if rows:
            embeddings = np.stack([r["embedding"] for r in rows]).astype(np.float32)
        else:
            embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        np.savez(self.path,
                 person_ids=np.array([r["person_id"] for r in rows], dtype=np.int64),
                 names=np.array([r["name"] for r in rows], dtype=str),
                 embeddings=embeddings,
                 photo_paths=np.array([r["photo_path"] for r in rows], dtype=str),
                 created_at=np.array([r["created_at"] for r in rows], dtype=str),
                 model_keys=np.array([r["model_key"] for r in rows], dtype=str))

    def load(self) -> list[tuple[int, str, np.ndarray]]:
        key = model_key()
        return [(r["person_id"], r["name"], r["embedding"]) for r in self._read() if r["model_key"] == key]

    def list_persons(self) -> list[dict]:
        persons: dict[int, dict] = {}
        for r in self._read():
            p = persons.setdefault(r["person_id"], {
                "id": r["person_id"], "name": r["name"], "photo_path": r["photo_path"],
                "created_at": r["created_at"], "embedding_count": 0})
            p["embedding_count"] += 1
        return sorted(persons.values(), key=lambda p: p["id"])

    def register(self, name: str, photo_paths: list[str], embeddings: list[np.ndarray]) -> int:
        rows = self._read()
        ids = {r["name"]: r["person_id"] for r in rows}
        pid = ids.get(name, max(ids.values(), default=0) + 1)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for i, emb in enumerate(embeddings):
            rows.append({"person_id": pid, "name": name,
                         "embedding": np.asarray(emb, dtype=np.float32).ravel(),
                         "photo_path": photo_paths[i] if i < len(photo_paths) else "",
                         "created_at": now, "model_key": model_key()})
        self._write(rows)
        return pid

    def delete(self, name: str) -> int:
        rows = self._read()
        kept = [r for r in rows if r["name"] != name]
        if len(kept) == len(rows):
            return 0
        self._write(kept)
        return 1


_store = None


def _find_c_module():
    """C의 database.py를 db/database.py → 루트 database.py 순서로 찾는다. 없으면 (None, None)."""
    for name in ("db.database", "database"):
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name not in (name, name.split(".")[0]):
                raise                      # C의 파일 안에서 난 다른 import 오류는 그대로 보여 준다
            continue
        if all(hasattr(module, f) for f in ("init_db", "register_person", "add_embeddings",
                                            "list_persons", "load_embeddings", "delete_person")):
            return module, name
    return None, None


def _get_store():
    """C의 database.py가 있으면 _CStore, 없으면 _NpzStore."""
    global _store
    if _store is None:
        module, name = _find_c_module()
        _store = _CStore(module, name) if module is not None else _NpzStore()
    return _store


def _store_label() -> str:
    """지금 쓰는 저장소 이름 (화면·콘솔 표시용)."""
    return _get_store().label


def open_camera(index: int = CAMERA_INDEX):
    """카메라를 연다. Windows에서는 CAP_DSHOW가 더 빨리 열려서 먼저 시도한다. 못 열면 None."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"카메라를 열 수 없습니다 (카메라 번호 {index})")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


# ──────────────────────────────────────────────
# 모델 불러오기
# ──────────────────────────────────────────────
def _model_path(filename: str) -> str:
    path = os.path.abspath(os.path.join(MODELS_DIR, filename))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {path}\n"
            f"  OpenCV Zoo(github.com/opencv/opencv_zoo)에서 {filename}을 받아 "
            f"{MODELS_DIR}/ 폴더에 넣어 주세요.")
    return path


def _ascii_copy(path: str) -> str:
    """한글 경로 대비: 모델을 영문 경로(임시 폴더)에 복사하고 그 경로를 돌려준다."""
    folder = os.path.join(tempfile.gettempdir(), "visitor_models")
    os.makedirs(folder, exist_ok=True)
    dst = os.path.join(folder, os.path.basename(path))
    if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(path):
        shutil.copyfile(path, dst)
    return dst


def _create_detector(path: str):
    size = (320, 320)   # 임시 크기. 실제 크기는 _detect_faces()에서 맞춘다
    if path.isascii():
        return cv2.FaceDetectorYN.create(path, "", size, DETECT_SCORE, DETECT_NMS, DETECT_TOP_K)
    # 한글 경로: 1) 파일을 바이트로 읽어 넘기기 → 2) 안 되면 영문 경로로 복사
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.FaceDetectorYN.create("onnx", buf, np.empty(0, np.uint8), size,
                                         DETECT_SCORE, DETECT_NMS, DETECT_TOP_K)
    except (cv2.error, TypeError):
        return cv2.FaceDetectorYN.create(_ascii_copy(path), "", size,
                                         DETECT_SCORE, DETECT_NMS, DETECT_TOP_K)


def _create_recognizer(path: str):
    if path.isascii():
        return cv2.FaceRecognizerSF.create(path, "")
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.FaceRecognizerSF.create("onnx", buf, np.empty(0, np.uint8))
    except (cv2.error, TypeError):
        return cv2.FaceRecognizerSF.create(_ascii_copy(path), "")


def _load_models() -> None:
    """모델은 처음 한 번만 불러온다."""
    global _detector, _recognizer
    if _detector is not None and _recognizer is not None:
        return
    if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
        raise RuntimeError(f"현재 OpenCV({cv2.__version__})에 FaceDetectorYN/FaceRecognizerSF가 없습니다.")
    _detector = _create_detector(_model_path(DETECTOR_MODEL))
    _recognizer = _create_recognizer(_model_path(RECOGNIZER_MODEL))


def _load_known(force: bool = False) -> None:
    global _known, _known_loaded
    if _known_loaded and not force:
        return
    _known = [(int(pid), str(name), np.asarray(emb, dtype=np.float32).ravel())
              for pid, name, emb in _get_store().load()]
    _known_loaded = True


# ──────────────────────────────────────────────
# 검출·특징 추출
# ──────────────────────────────────────────────
def _detect_faces(image: np.ndarray) -> np.ndarray:
    """얼굴을 모두 찾는다. 각 행: x, y, w, h, 눈·코·입 5점(10개), 신뢰도 → 15개 값."""
    global _detector_size
    _load_models()
    h, w = image.shape[:2]
    if _detector_size != (w, h):           # 입력 크기가 바뀌면 다시 설정해야 한다
        _detector.setInputSize((w, h))
        _detector_size = (w, h)
    _, faces = _detector.detect(image)
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    return faces


def _largest(faces: np.ndarray) -> np.ndarray | None:
    if len(faces) == 0:
        return None
    return faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    """LAB 색공간의 L 채널에 CLAHE를 적용한다."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _embedding_from_face(image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
    """검출된 얼굴 한 개를 정렬(alignCrop)하고 특징 벡터를 뽑는다. 등록·인식 모두 이 함수를 쓴다."""
    aligned = _recognizer.alignCrop(image, face_row)
    if USE_CLAHE:
        aligned = _apply_clahe(aligned)
    feature = _recognizer.feature(aligned)
    return np.asarray(feature, dtype=np.float32).ravel().copy()


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


# ──────────────────────────────────────────────
# 공통 명세 함수 (3.5)
# ──────────────────────────────────────────────
def init_recognizer() -> None:
    """프로그램 시작 시 1번 호출. 모델과 등록 얼굴을 불러온다."""
    _load_models()
    reload_known_faces()


def reload_known_faces() -> None:
    """등록 얼굴만 다시 불러온다 (등록자가 바뀌었을 때)."""
    _load_known(force=True)
    persons = len({pid for pid, _, _ in _known})
    print(f"[face] 등록자 {persons}명, 벡터 {len(_known)}개 불러옴 ({_store_label()}, model_key={model_key()})")


def extract_embedding(face_bgr: np.ndarray) -> np.ndarray | None:
    """이미지 안의 가장 큰 얼굴 1개 → float32 1차원 특징 벡터. 얼굴이 없으면 None."""
    if face_bgr is None or face_bgr.size == 0:
        return None
    face = _largest(_detect_faces(face_bgr))
    if face is None:
        return None
    return _embedding_from_face(face_bgr, face)


def embedding_distance(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터 사이의 거리 = 1 − 코사인 유사도. 작을수록 비슷하다."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("inf")
    return float(1.0 - np.dot(a, b) / (na * nb))


def _nearest(embedding: np.ndarray) -> tuple[int | None, str | None, float]:
    """기준값과 상관없이 가장 가까운 등록자를 찾는다. 등록자가 없으면 (None, None, inf)."""
    _load_known()
    best_id, best_name, best = None, None, float("inf")
    for pid, name, known in _known:   # 한 사람에게 벡터가 여러 개면 가장 가까운 것이 그 사람의 거리
        d = embedding_distance(embedding, known)
        if d < best:
            best_id, best_name, best = pid, name, d
    return best_id, best_name, best


def compare(embedding: np.ndarray) -> tuple[int | None, str | None, float]:
    """(기준값 이내 등록자 id, 이름, 가장 가까운 거리). 기준값을 넘으면 id·이름은 None."""
    pid, name, dist = _nearest(embedding)
    if dist <= MATCH_THRESHOLD:
        return pid, name, dist
    return None, None, dist


def recognize(frame: np.ndarray, person: PersonBox) -> RecognitionResult:
    """사람 박스 안의 얼굴을 인식한다 (인수인계서 5.2 처리 순서)."""
    if frame is None or frame.size == 0:
        return RecognitionResult(face_found=False)
    fh, fw = frame.shape[:2]

    # 1. 사람 박스를 10% 넓혀 자른다 (프레임 밖으로 나가지 않게)
    x1, y1, x2, y2 = person.bbox
    mx, my = int((x2 - x1) * BBOX_MARGIN), int((y2 - y1) * BBOX_MARGIN)
    cx1, cy1, cx2, cy2 = _clip_box(x1 - mx, y1 - my, x2 + mx, y2 + my, fw, fh)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return RecognitionResult(face_found=False)

    # 2~3. 가장 큰 얼굴 하나. 없거나 너무 작으면 못 찾음
    face = _largest(_detect_faces(crop))
    if face is None or face[2] < MIN_FACE_WIDTH:
        return RecognitionResult(face_found=False)

    # 4~5. (CLAHE는 _embedding_from_face 안에서) 특징 벡터 → 비교
    embedding = _embedding_from_face(crop, face)
    pid, name, dist = compare(embedding)

    # 6. 얼굴 부분 이미지
    ch, cw = crop.shape[:2]
    ax, ay, bx, by = _clip_box(int(face[0]), int(face[1]),
                               int(face[0] + face[2]), int(face[1] + face[3]), cw, ch)
    face_image = crop[ay:by, ax:bx].copy()

    return RecognitionResult(face_found=True, person_id=pid, name=name, distance=dist,
                             face_image=face_image, embedding=embedding)


# ──────────────────────────────────────────────
# 데모: python -m vision.face --demo
# ──────────────────────────────────────────────
def _demo(camera_index: int = CAMERA_INDEX) -> None:
    """프레임 전체를 사람 1명으로 보고 recognize()를 시험한다.
    화면에는 영문·숫자만 쓰고 이름은 콘솔에 출력한다.
    l: 등록 얼굴 다시 불러오기   q / Esc: 끝내기
    """
    init_recognizer()
    print(f"기준값 MATCH_THRESHOLD = {MATCH_THRESHOLD}, USE_CLAHE = {USE_CLAHE}")
    print("등록은 python -m vision.register 이름  /  l: 등록 얼굴 다시 불러오기  /  q 또는 Esc: 끝내기")

    cap = open_camera(camera_index)
    if cap is None:
        return

    window = "face demo"
    font = cv2.FONT_HERSHEY_SIMPLEX
    prev, fps = time.time(), 0.0
    last_state = None   # 콘솔 출력을 상태가 바뀔 때만 하기 위해
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라에서 영상을 읽지 못했습니다")
                break
            h, w = frame.shape[:2]

            # B의 YOLO 대신 프레임 전체를 사람 한 명으로 본다 (인수인계서 8장)
            result = recognize(frame, PersonBox(track_id=1, bbox=(0, 0, w, h), conf=1.0))

            # 그리기용: 얼굴 박스와 눈·코·입 5점
            faces = _detect_faces(frame)
            main = _largest(faces)
            for f in faces:
                x, y, fw_, fh_ = (int(v) for v in f[:4])
                cv2.rectangle(frame, (x, y), (x + fw_, y + fh_), (160, 160, 160), 1)

            if not result.face_found:
                state = ("noface",)
                label, color = "NO FACE (or too small)", (0, 200, 255)
            elif result.person_id is not None:
                state = ("match", result.person_id)
                label, color = f"ID {result.person_id}  d={result.distance:.3f}  MATCH", (0, 200, 0)
            elif result.distance == float("inf"):
                state = ("empty",)
                label, color = "no known faces", (0, 200, 255)
            else:
                state = ("unreg",)
                label, color = f"UNREG  d={result.distance:.3f}", (0, 0, 255)

            if main is not None:
                x, y, fw_, fh_ = (int(v) for v in main[:4])
                cv2.rectangle(frame, (x, y), (x + fw_, y + fh_), color, 2)
                cv2.putText(frame, label, (x, max(y - 8, 16)), font, 0.6, color, 2)
                for i in range(5):
                    px, py = int(main[4 + 2 * i]), int(main[5 + 2 * i])
                    cv2.circle(frame, (px, py), 2, (255, 0, 255), -1)
            else:
                cv2.putText(frame, label, (10, 55), font, 0.6, color, 2)

            if state != last_state:
                if state[0] == "match":
                    print(f"[인식] 등록자: {result.name} (id {result.person_id}), 거리 {result.distance:.3f}")
                elif state[0] == "unreg":
                    print(f"[인식] 미등록자 (가장 가까운 거리 {result.distance:.3f})")
                elif state[0] == "empty":
                    print("[인식] 등록된 얼굴이 없습니다")
                else:
                    print("[인식] 얼굴 없음")
                last_state = state

            now = time.time()
            current = 1.0 / max(now - prev, 1e-6)
            fps = current if fps == 0 else 0.9 * fps + 0.1 * current
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  faces {len(faces)}  known {len(_known)}  thr {MATCH_THRESHOLD:.3f}",
                        (10, 25), font, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "l: reload   q: quit", (10, h - 12), font, 0.6, (0, 255, 255), 2)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # q 또는 Esc
                break
            if key == ord("l"):
                reload_known_faces()
                last_state = None
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:   # 창 닫기 버튼
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="얼굴 검출·특징 추출·비교 (YuNet + SFace)")
    parser.add_argument("--demo", action="store_true", help="웹캠으로 recognize() 결과 보기")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="카메라 번호")
    args = parser.parse_args()
    if args.demo:
        _demo(args.camera)
    else:
        parser.print_help()
