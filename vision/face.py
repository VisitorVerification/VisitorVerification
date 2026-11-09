"""얼굴 검출·특징 추출·비교 (A 담당).

OpenCV 내장 YuNet(얼굴 검출) + SFace(특징 추출)를 쓴다.
거리 = 1 - 코사인 유사도 (작을수록 비슷, 공통 규칙 3.8).

혼자 시험:  python -m vision.face --demo
"""
import argparse
import os
import time

import cv2
import numpy as np

import config
from common.types import PersonBox, RecognitionResult
from vision import facedb

_detector = None
_recognizer = None
_det_size = None                    # 검출기에 마지막으로 설정한 입력 크기
_use_clahe = config.USE_CLAHE       # 실험 스크립트가 set_use_clahe()로 바꿀 수 있음

# 등록 얼굴 (메모리)
_known_ids: list[int] = []
_known_names: list[str] = []
_known_matrix = np.zeros((0, 128), dtype=np.float32)   # 행마다 정규화된 벡터
_known_source = "none"


# ───────────────────────── 모델 불러오기 ─────────────────────────

def _read_model_bytes(filename: str) -> np.ndarray:
    """한글 경로에서도 되도록 모델을 바이트로 읽는다."""
    path = os.path.join(config.MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {path}\n"
            "OpenCV Zoo(github.com/opencv/opencv_zoo)에서 받아 models/ 폴더에 두세요.")
    return np.fromfile(path, dtype=np.uint8)


def _load_models() -> None:
    global _detector, _recognizer, _det_size
    if _detector is not None and _recognizer is not None:
        return
    det_buf = _read_model_bytes(config.FACE_DET_MODEL)
    rec_buf = _read_model_bytes(config.FACE_REC_MODEL)
    empty = np.zeros(0, dtype=np.uint8)
    _detector = cv2.FaceDetectorYN.create(
        "onnx", det_buf, empty, (320, 320), config.FACE_DET_SCORE, 0.3, 5000)
    _recognizer = cv2.FaceRecognizerSF.create("onnx", rec_buf, empty)
    _det_size = (320, 320)


def _set_known(rows: list[tuple[int, str, np.ndarray]], source: str) -> None:
    global _known_ids, _known_names, _known_matrix, _known_source
    _known_ids = [int(r[0]) for r in rows]
    _known_names = [str(r[1]) for r in rows]
    if rows:
        mat = np.stack([np.asarray(r[2], dtype=np.float32).ravel() for r in rows])
        _known_matrix = _normalize_rows(mat)
    else:
        _known_matrix = np.zeros((0, 128), dtype=np.float32)
    _known_source = source


def reload_known_faces() -> None:
    """등록 얼굴만 다시 불러온다 (등록자가 바뀌었을 때).
    C의 database.py가 있으면 DB에서, 없으면 임시 파일 experiments/known_faces.npz에서 읽는다."""
    source = facedb.backend()
    _set_known(facedb.load_known(), source)
    where = f"model_key={facedb.model_key()}" if source != "npz" else facedb.NPZ_PATH
    print(f"[face] {source}에서 등록 벡터 {len(_known_ids)}개 ({len(set(_known_names))}명)를 불러왔습니다. ({where})")


def init_recognizer() -> None:
    """프로그램 시작 시 1번 호출. 모델과 등록 얼굴을 불러온다."""
    _load_models()
    reload_known_faces()


def known_source() -> str:
    return _known_source


def set_use_clahe(flag: bool) -> None:
    """실험용: CLAHE 적용 여부를 실행 중에 바꾼다. 평소에는 config.USE_CLAHE를 쓴다."""
    global _use_clahe
    _use_clahe = bool(flag)


# ───────────────────────── 검출·전처리 ─────────────────────────

def detect_faces(image: np.ndarray) -> np.ndarray:
    """이미지 안의 얼굴을 모두 찾는다. 반환: (N, 15) float32 (x, y, w, h, 랜드마크 10개, 점수)."""
    global _det_size
    _load_models()
    if image is None or image.size == 0:
        return np.zeros((0, 15), dtype=np.float32)
    h, w = image.shape[:2]
    if _det_size != (w, h):                 # 입력 크기가 바뀌면 다시 설정해야 함
        _detector.setInputSize((w, h))
        _det_size = (w, h)
    _, faces = _detector.detect(image)
    if faces is None:
        return np.zeros((0, 15), dtype=np.float32)
    return faces


def _largest(faces: np.ndarray) -> np.ndarray | None:
    if len(faces) == 0:
        return None
    return faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """LAB 색공간의 L 채널에 CLAHE를 적용한다."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _face_rect(image: np.ndarray, face: np.ndarray, pad: float = 0.0) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    x, y, fw, fh = face[:4]
    px, py = fw * pad, fh * pad
    x1 = int(max(0, np.floor(x - px)))
    y1 = int(max(0, np.floor(y - py)))
    x2 = int(min(w, np.ceil(x + fw + px)))
    y2 = int(min(h, np.ceil(y + fh + py)))
    return x1, y1, x2, y2


def _embed(image: np.ndarray, face: np.ndarray) -> np.ndarray:
    """검출된 얼굴 하나 → 특징 벡터. 등록·인식이 모두 이 함수를 거친다(같은 전처리)."""
    if _use_clahe:
        image = image.copy()
        x1, y1, x2, y2 = _face_rect(image, face, pad=0.3)
        if x2 > x1 and y2 > y1:
            image[y1:y2, x1:x2] = apply_clahe(image[y1:y2, x1:x2])
    aligned = _recognizer.alignCrop(image, face)
    feat = _recognizer.feature(aligned)
    return np.asarray(feat, dtype=np.float32).ravel().copy()


# ───────────────────────── 공개 함수 (3.5) ─────────────────────────

def extract_embedding(face_bgr: np.ndarray) -> np.ndarray | None:
    """이미지 안에서 가장 큰 얼굴 1개의 특징 벡터(float32 1차원). 얼굴이 없으면 None."""
    face = _largest(detect_faces(face_bgr))
    if face is None:
        return None
    return _embed(face_bgr, face)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def embedding_distance(a: np.ndarray, b: np.ndarray) -> float:
    """벡터 두 개 사이의 거리 = 1 - 코사인 유사도."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("inf")
    return float(max(0.0, 1.0 - np.dot(a, b) / (na * nb)))


def _all_distances(embedding: np.ndarray) -> np.ndarray:
    e = np.asarray(embedding, dtype=np.float32).ravel()
    n = np.linalg.norm(e)
    if n == 0 or len(_known_ids) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.maximum(0.0, 1.0 - _known_matrix @ (e / n))


def nearest_per_person(embedding: np.ndarray) -> dict[str, tuple[int, float]]:
    """등록자별 가장 가까운 거리. {이름: (person_id, 거리)} (실험 스크립트용)"""
    out: dict[str, tuple[int, float]] = {}
    for pid, name, d in zip(_known_ids, _known_names, _all_distances(embedding)):
        if name not in out or d < out[name][1]:
            out[name] = (pid, float(d))
    return out


def compare(embedding: np.ndarray) -> tuple[int | None, str | None, float]:
    """(기준값 이내 등록자 id, 이름, 가장 가까운 거리). 기준값을 넘으면 id·이름은 None.
    한 사람에게 벡터가 여러 개면 그중 가장 가까운 거리를 쓴다."""
    dists = _all_distances(embedding)
    if dists.size == 0:
        return None, None, float("inf")
    i = int(np.argmin(dists))
    d = float(dists[i])
    if d <= config.MATCH_THRESHOLD:
        return _known_ids[i], _known_names[i], d
    return None, None, d


def recognize(frame: np.ndarray, person: PersonBox) -> RecognitionResult:
    """사람 박스 안의 얼굴을 인식한다 (5.2 처리 순서)."""
    if _recognizer is None:
        init_recognizer()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = person.bbox
    mx = (x2 - x1) * config.PERSON_BOX_MARGIN
    my = (y2 - y1) * config.PERSON_BOX_MARGIN
    cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    cx2, cy2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return RecognitionResult(face_found=False)
    crop = frame[cy1:cy2, cx1:cx2]

    face = _largest(detect_faces(crop))
    if face is None or face[2] < config.MIN_FACE_WIDTH:
        return RecognitionResult(face_found=False)

    embedding = _embed(crop, face)
    person_id, name, distance = compare(embedding)
    fx1, fy1, fx2, fy2 = _face_rect(crop, face)
    return RecognitionResult(
        face_found=True,
        person_id=person_id,
        name=name,
        distance=distance,
        face_image=crop[fy1:fy2, fx1:fx2].copy(),
        embedding=embedding,
    )


# ───────────────────────── 데모 ─────────────────────────

def _demo(camera_index: int) -> None:
    init_recognizer()
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"카메라 {camera_index}번을 열 수 없습니다.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    print(f"[demo] 등록자 {len(set(_known_names))}명, 기준값 {config.MATCH_THRESHOLD}, "
          f"CLAHE {'켬' if _use_clahe else '끔'}. q 또는 ESC로 종료.")

    last_print = 0.0
    fps, t_prev = 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했습니다.")
            break
        h, w = frame.shape[:2]
        result = recognize(frame, PersonBox(track_id=1, bbox=(0, 0, w, h), conf=1.0))

        if result.face_found:
            face = _largest(detect_faces(frame))
            nearest = nearest_per_person(result.embedding)
            near_name, (near_id, near_d) = (min(nearest.items(), key=lambda kv: kv[1][1])
                                            if nearest else (None, (None, float("inf"))))
            matched = result.person_id is not None
            color = (0, 200, 0) if matched else (0, 0, 255)
            if face is not None:
                x1, y1, x2, y2 = _face_rect(frame, face)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                for k in range(5):
                    cv2.circle(frame, (int(face[4 + 2 * k]), int(face[5 + 2 * k])), 2, (255, 255, 0), -1)
                label = (f"ID {result.person_id}  d={result.distance:.3f}" if matched
                         else f"UNREG  nearest ID {near_id}  d={near_d:.3f}")
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            now = time.time()
            if now - last_print > 0.5:          # 이름은 콘솔에 (OpenCV 창은 한글이 깨짐)
                state = f"등록자 {result.name}" if matched else "미등록"
                print(f"  {state:<12} 가장 가까운: {near_name}  거리 {near_d:.3f}")
                last_print = now
        else:
            cv2.putText(frame, "NO FACE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        t_now = time.time()
        fps = 0.9 * fps + 0.1 / max(1e-6, t_now - t_prev)
        t_prev = t_now
        cv2.putText(frame, f"TH {config.MATCH_THRESHOLD:.3f}  FPS {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("face demo", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="얼굴 인식 모듈 (A)")
    ap.add_argument("--demo", action="store_true", help="웹캠으로 인식 결과를 확인")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = ap.parse_args()
    if args.demo:
        _demo(args.camera)
    else:
        ap.print_help()
