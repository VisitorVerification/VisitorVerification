# 방문자 확인 시스템 — A 파트 (얼굴 인식)

| 항목 | 내용 |
|---|---|
| 담당 | 정재열 (역할 A: 얼굴 인식) |
| 기준 문서 | `A작업.md` (A 작업 인수인계서), `방문자확인시스템_계획서.md` (3차 개정) |
| 작성일 | 2026-11-09 |
| 사용 기술 | Python, OpenCV 내장 **YuNet**(얼굴 검출) + **SFace**(특징 추출), NumPy, Matplotlib |
| 상태 | 1~3주차 A 코드 작성 완료. 샘플 사진으로 검증 완료, **실제 웹캠 시험은 아직** |

카메라에 잡힌 사람의 얼굴을 특징 벡터로 바꾸고, 등록된 얼굴과 비교해 **등록자 / 미등록자 / 인식 불가**를 판정하는 부분이다. 얼굴 등록 도구와 판정 기준값 실험 스크립트도 포함한다.

---

## 목차
1. [한눈에 보기](#1-한눈에-보기)
2. [폴더 구조](#2-폴더-구조)
3. [설치와 실행](#3-설치와-실행)
4. [명령어 모음](#4-명령어-모음)
5. [얼굴 인식 모듈 `vision/face.py`](#5-얼굴-인식-모듈-visionfacepy)
6. [판정 모듈 `vision/judge.py`](#6-판정-모듈-visionjudgepy)
7. [얼굴 등록 도구 `vision/register.py`](#7-얼굴-등록-도구-visionregisterpy)
8. [기준값 실험 `experiments/`](#8-기준값-실험-experiments)
9. [설정값 `config.py`](#9-설정값-configpy)
10. [검증 결과](#10-검증-결과)
11. [팀원에게 알릴 것](#11-팀원에게-알릴-것)
12. [남은 일](#12-남은-일)
13. [문제 해결](#13-문제-해결)

---

## 1. 한눈에 보기

### 1.1 처리 흐름에서 A가 맡은 부분

```
카메라 프레임 (B)
 → detect_people()          사람 박스 + 추적 번호            (B)
 → recognize(frame, person) 얼굴 검출 → 특징 벡터 → 등록자와 비교   (A) ← 이 저장소
 → vote(track_id, result)   최근 5번 중 4번 같으면 확정        (A) ← 이 저장소
 → needs_face_hint()        얼굴 안내 문구가 필요한지          (A) ← 이 저장소
 → 안내·문 열기·DB 저장       (B, C)
```

### 1.2 만든 것

| 파일 | 내용 | 인수인계서 |
|---|---|---|
| `vision/face.py` | 모델 불러오기, 얼굴 검출, 특징 추출, 비교, `recognize()`, `--demo` | 5.1, 5.2 |
| `vision/judge.py` | 다수결 판정 `vote()`, 얼굴 안내 필요 여부 `needs_face_hint()` | 5.5 |
| `vision/register.py` | 얼굴 등록 명령줄 도구 (웹캠·폴더·추가·목록·삭제) | 5.3 |
| `experiments/collect_distances.py` | 웹캠으로 거리 20프레임 측정 → CSV | 5.4 |
| `experiments/plot_distances.py` | CSV → 히스토그램 PNG, 보고서용 표, 기준값 후보, CLAHE 비교 | 5.4 |
| `experiments/test_judge.py` | 가짜 인식 결과로 판정 규칙 자동 검사 (9개) | 5.5 |
| `models/*.onnx` | YuNet·SFace 모델 파일 (OpenCV Zoo에서 받음) | 5.1 |

### 1.3 A가 먼저 만들어 둔 **다른 사람 담당** 파일
A 코드를 혼자 실행하고 시험하려고 공통 명세(3장)대로 먼저 만들었다. **원래 담당자의 파일로 바꿔도 된다.**

| 파일 | 원래 담당 | 내용 |
|---|---|---|
| `config.py` | B | 3.3 그대로 + A 값 수정·추가 ([9장](#9-설정값-configpy)) |
| `common/types.py` | B | 3.4 그대로 (`PersonBox`, `RecognitionResult`, `Decision`) |
| `common/imageio.py` | B | 한글 경로용 `imread_unicode()`, `imwrite_unicode()` |
| `requirements.txt`, `.gitignore` | B | 3.2 초안 그대로 + 얼굴 인식 라이브러리 메모 |
| `*/__init__.py` | — | 빈 파일 (패키지 표시) |

`database.py`(C)는 만들지 않았다. A는 **C의 `sqlite_1` 브랜치 `database.py` 실제 함수 형태에 맞춰** 호출하고(`vision/facedb.py`), 그 파일이 없을 때는 **임시 파일 `experiments/known_faces.npz`** 를 쓴다 ([5.4절](#54-등록-얼굴은-어디서-불러오나)).

---

## 2. 폴더 구조

```
visitor-system/
├── README.md                     이 문서
├── requirements.txt              (B) 패키지 목록
├── .gitignore                    (B) data/, .venv/, known_faces.npz 제외
├── config.py                     (B) 설정값. A 담당 값은 9장
├── common/
│   ├── __init__.py
│   ├── types.py                  (B) 공통 데이터 형식
│   └── imageio.py                (B) 한글 경로 이미지 읽기·쓰기
├── vision/
│   ├── __init__.py
│   ├── face.py                   (A) 얼굴 검출·특징 추출·비교
│   ├── judge.py                  (A) 다수결 판정
│   ├── facedb.py                 (A) C의 DB 함수 호출 (없으면 임시 npz)
│   └── register.py               (A) 얼굴 등록 도구
├── experiments/
│   ├── collect_distances.py      (A) 거리 측정
│   ├── plot_distances.py         (A) 히스토그램·결과표
│   ├── test_judge.py             (A) 판정 규칙 테스트
│   ├── known_faces.npz           (실행 중 생성) DB 대신 쓰는 임시 등록 벡터
│   ├── distances.csv             (실행 중 생성) 측정 결과
│   └── plots/                    (실행 중 생성) 그래프·표
├── models/
│   ├── face_detection_yunet_2023mar.onnx     약 230KB
│   └── face_recognition_sface_2021dec.onnx   약 37MB
├── db/        output/   dashboard/           (B·C가 채울 폴더, 지금은 __init__.py만)
└── data/                         (실행 중 생성, GitHub에 올리지 않음)
    └── faces/<이름>/<번호>.jpg    등록 사진
```

---

## 3. 설치와 실행

### 3.1 작업 위치
**코드는 한글과 공백이 없는 로컬 경로에 둔다.** 예: `C:\projects\visitor-system`
- 지금 작업한 폴더(`OneDrive\바탕 화면\디영처 a 수정 후`)는 이 규칙에 맞지 않는다. 팀 GitHub 저장소로 옮길 때 경로를 바꾼다.
- A 코드는 모델 파일을 바이트로 읽고 이미지는 `imageio`를 쓰므로 한글 경로에서도 동작은 확인했다. 하지만 OneDrive 동기화가 SQLite 파일(C의 DB)을 잠글 수 있다.

### 3.2 설치

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- 얼굴 인식에 **추가 라이브러리는 필요 없다.** `opencv-python`(4.8 이상)에 YuNet·SFace가 들어 있다.
- 모델 파일 2개는 `models/`에 이미 있다. 없으면 [OpenCV Zoo](https://github.com/opencv/opencv_zoo)에서 아래 파일을 받아 `models/`에 넣는다.
  - `models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
  - `models/face_recognition_sface/face_recognition_sface_2021dec.onnx`

### 3.3 모든 명령은 프로젝트 루트에서 실행
`config.py`의 경로(`data`, `models`)가 상대 경로라서 루트에서 실행해야 한다.

---

## 4. 명령어 모음

| 하고 싶은 일 | 명령 | 끄는 법 |
|---|---|---|
| 웹캠으로 얼굴 등록 (5장) | `python -m vision.register 홍길동` | `ESC` / `q` (취소) |
| 폴더 사진으로 등록 | `python -m vision.register 홍길동 --folder 사진폴더` | — |
| 이미 등록된 사람에게 사진 추가 | `python -m vision.register 홍길동 --add` | `ESC` / `q` |
| 등록자 목록 | `python -m vision.register --list` | — |
| 등록자 삭제 | `python -m vision.register --delete 홍길동` (`y` 입력 시 삭제) | — |
| 인식 데모 | `python -m vision.face --demo` | `q` / `ESC` |
| 판정 규칙 테스트 | `python experiments/test_judge.py` | — |
| 거리 측정 | `python experiments/collect_distances.py --who 홍길동 --condition front_bright` | `ESC` (저장 안 함) |
| 그래프·결과표 | `python experiments/plot_distances.py` | — |

- 카메라 번호를 바꾸려면 `--camera 1`처럼 붙인다 (기본값 `config.CAMERA_INDEX`).
- 키 입력은 **영상 창을 클릭해 선택한 상태**에서 해야 한다. 창이 멈추면 터미널에서 `Ctrl + C`.
- OpenCV 창은 한글이 깨지므로 화면에는 영문·숫자만 나오고, 이름과 한글 안내는 **콘솔(터미널)** 에 나온다.

---

## 5. 얼굴 인식 모듈 `vision/face.py`

### 5.1 라이브러리 선정 (1주차 결정 사항)
**OpenCV 내장 YuNet + SFace**를 쓴다.

| 이유 | 설명 |
|---|---|
| 설치가 쉬움 | `opencv-python`만 있으면 됨. dlib·C++ 빌드 도구 불필요 |
| 빠름 | 얼굴 한 명 인식(검출+특징 추출+비교)에 약 27ms (노트북 CPU) |
| 수업 주제와 맞음 | OpenCV 기반 영상처리 |
| 정확도 충분 | 같은 사람 거리 0.09 vs 다른 사람 0.83~0.86 (샘플 사진, [10장](#10-검증-결과)) |

- 얼굴 검출: `cv2.FaceDetectorYN` — 얼굴 박스와 눈·코·입 랜드마크 5점
- 특징 추출: `cv2.FaceRecognizerSF` — `alignCrop()`으로 얼굴을 정렬한 뒤 `feature()`로 **128차원 float32 벡터**

### 5.2 거리 단위
공통 규칙 3.8에 따라 **거리 = 1 − 코사인 유사도**. 작을수록 비슷하다.
- 범위는 대략 0(같은 얼굴) ~ 1 이상(전혀 다른 얼굴). 반올림 오차로 음수가 나오지 않도록 0 아래는 0으로 자른다.
- OpenCV 문서의 SFace 기준(유사도 0.363)은 거리로 **0.637**. 지금 `MATCH_THRESHOLD`는 이보다 엄격한 **0.55**(임시값)이다.

### 5.3 공개 함수 (공통 명세 3.5)

| 함수 | 하는 일 | 쓰는 사람 |
|---|---|---|
| `init_recognizer()` | 모델을 한 번 불러오고 등록 얼굴을 메모리에 담는다. 프로그램 시작 시 1번 | B (메인 루프) |
| `reload_known_faces()` | 등록 얼굴만 다시 불러온다 (등록자가 바뀌었을 때) | B |
| `extract_embedding(image)` | 이미지 안 **가장 큰 얼굴 1개** → 특징 벡터. 얼굴이 없으면 `None` | A (등록 도구) |
| `compare(embedding)` | 가장 가까운 등록자를 찾는다. 기준값 이하면 `(id, 이름, 거리)`, 넘으면 `(None, None, 거리)` | A |
| `embedding_distance(a, b)` | 벡터 두 개 사이 거리 | B (미등록자 중복 방지) |
| `recognize(frame, person)` | 사람 박스 안의 얼굴 인식 → `RecognitionResult` | B (메인 루프) |

명세에 없는 **추가 함수** (A 내부·실험용, 다른 사람이 쓸 필요 없음):

| 함수 | 하는 일 |
|---|---|
| `detect_faces(image)` | 이미지 속 얼굴 전부 찾기 → `(N, 15)` 배열. 등록 도구가 "얼굴이 정확히 1개인지" 검사할 때 씀 |
| `nearest_per_person(embedding)` | 등록자별 가장 가까운 거리 `{이름: (id, 거리)}`. 실험 스크립트용 |
| `apply_clahe(image)` | LAB의 L 채널에 CLAHE 적용 |
| `set_use_clahe(flag)` | 실행 중 CLAHE 켜기/끄기 (실험용. 평소에는 `config.USE_CLAHE`) |
| `known_source()` | 지금 등록 얼굴을 어디서 불러왔는지 (`DB (database)` / `npz`) |

### 5.4 등록 얼굴은 어디서 불러오나
`reload_known_faces(source="auto")`의 동작:
DB 호출은 모두 **`vision/facedb.py`** 한 곳에서 한다.
1. C의 `database.py`를 `db/database.py` → 루트 `database.py` 순서로 찾는다 (파일 위치가 어떻게 정해져도 동작).
2. 찾으면 `init_db(config.DB_PATH)` 후 `load_embeddings(model_key=..., db_path=config.DB_PATH)`로 불러온다. **DB 경로는 항상 `config.DB_PATH`** 를 넘긴다.
3. `database.py`가 없으면 **`experiments/known_faces.npz`** 를 읽는다. DB 파일이 있는데 오류가 나면 npz로 넘어가지 않고 **오류를 그대로 보여 준다** (모르고 임시 파일을 쓰는 일 방지).
3. 둘 다 없으면 등록자 0명 → 거리 `inf`, 모두 미등록자.

→ **C의 `database.py`가 저장소에 들어오면 코드 수정 없이 자동으로 DB를 쓴다.** DB로 넘어간 뒤에는 팀원을 다시 등록해야 한다 (npz 내용은 DB로 옮겨지지 않음).

**`model_key`**: C의 DB는 벡터마다 "어떤 모델로 만들었는지"를 저장한다. A는 `config.FACE_MODEL_KEY`(`"sface_2021dec"`)를 쓰고, `USE_CLAHE = True`이면 자동으로 `"sface_2021dec_clahe"`가 된다. 그래서 CLAHE 설정을 바꾸면 예전 벡터는 불러오지 않는다 (재등록 필요).

**C의 실제 함수와 A가 부르는 방식**

| C의 함수 (`sqlite_1`) | A가 하는 일 |
|---|---|
| `load_embeddings(*, model_key, db_path)` → dict 목록 | `(person_id, person_name, float32 배열)` 튜플로 바꿔 사용 |
| `register_person(name, embeddings, *, model_key, photo_path, db_path)` | 새 사람 등록. 대표 사진은 첫 번째 사진 |
| `add_embeddings(person_id, embeddings, *, model_key, db_path)` | 이미 있는 이름이면 이쪽으로 벡터 추가 |
| `list_persons(*, db_path)` | 목록, 이름 → `person_id` 찾기 |
| `delete_person(person_id, *, db_path)` | 이름으로 id를 찾아 삭제. C의 DB는 이름 중복을 막지 않아서 같은 이름이 여러 명이면 모두 삭제 |

`known_faces.npz` 형식: `ids`(int), `names`(str), `embeddings`(N×128 float32), `photos`(str)

### 5.5 `recognize()` 처리 순서 (인수인계서 5.2)
1. `person.bbox`를 **10%** (`PERSON_BOX_MARGIN`) 넓혀 자른다. 프레임 밖으로 나가는 부분은 잘라낸다.
2. 잘라낸 영역에서 얼굴을 검출하고 **가장 큰 얼굴 하나**를 고른다.
3. 얼굴이 없거나 얼굴 폭이 **40픽셀**(`MIN_FACE_WIDTH`) 미만이면 `RecognitionResult(face_found=False)`.
4. `USE_CLAHE`가 켜져 있으면 얼굴 주변 영역(얼굴 크기의 30% 여유)에 CLAHE를 먼저 적용한다.
5. 정렬(`alignCrop`) → 특징 벡터 → `compare()`.
6. `face_found=True`, `person_id`, `name`, `distance`, `face_image`(얼굴 부분 BGR), `embedding`을 채워 돌려준다.

**규칙**
- 한 등록자에게 벡터가 여러 개(등록 사진 5장)면 그중 **가장 가까운 거리**가 그 사람의 거리.
- **등록과 인식은 같은 내부 함수(`_embed`)** 를 거친다 → 전처리가 항상 같다. `USE_CLAHE`를 바꾸면 **전원 재등록**.
- 얼굴 검출기는 입력 크기가 바뀔 때마다 `setInputSize()`를 다시 호출한다 (사람 박스 크기가 매번 다르므로).
- 모델은 `np.fromfile()`로 바이트를 읽어 메모리에서 만든다 → 경로에 한글이 있어도 된다.

### 5.6 데모 `python -m vision.face --demo`
- 프레임 전체를 사람 한 명(`PersonBox(track_id=1, bbox=(0,0,w,h))`)으로 보고 `recognize()`를 매 프레임 실행한다. B의 `detect_people()` 없이 혼자 시험하는 용도.
- 화면: 얼굴 박스(초록=등록자, 빨강=미등록), 랜드마크 5점, `ID 1 d=0.312` 또는 `UNREG nearest ID 2 d=0.701`, 기준값, FPS. 얼굴이 없으면 `NO FACE`.
- 콘솔: 0.5초마다 `등록자 홍길동  가장 가까운: 홍길동  거리 0.312`

---

## 6. 판정 모듈 `vision/judge.py`

### 6.1 추적 번호별 상태
| 상태 | 내용 |
|---|---|
| 최근 결과 목록 | 길이 `VOTE_WINDOW`(5). **얼굴을 찾은 결과만** 넣는다. 등록자는 `person_id`, 미등록은 `"UNREG"` |
| 처음 본 시각 | 처음 `vote()`가 불린 시각 (판정 후에는 판정 시각으로 초기화) |
| 마지막으로 얼굴을 찾은 시각 | `face_found=True`였던 마지막 시각 |
| 마지막 갱신 시각 | `TRACK_EXPIRE_SEC`(10초) 넘게 갱신 안 된 추적 번호는 지운다 |
| 최근 얼굴 정보 | 마지막 `face_image`, `embedding`, `distance` |

### 6.2 `vote(track_id, result, now)` 규칙 (위에서부터 먼저 맞는 것)

| 판정 | 조건 | 돌려주는 `Decision` |
|---|---|---|
| 등록자 | 최근 결과 중 같은 `person_id`가 `VOTE_MIN`(4)개 이상 | `registered`, id·이름, 그 결과들 중 **가장 작은 거리**, 최근 얼굴·벡터 |
| 미등록자 | `"UNREG"`가 `VOTE_MIN`개 이상 | `unregistered`, 최근 얼굴·벡터·거리 |
| 인식 불가 | 마지막으로 얼굴 찾은 지(한 번도 못 찾았으면 처음 본 지) `UNKNOWN_TIMEOUT_SEC`(5초) 이상 | `unknown`, 얼굴·벡터 없음 |
| 보류 | 그 외 | `None` |

- 판정을 돌려준 뒤에는 그 추적 번호의 결과 목록과 시각 기준을 **초기화**한다. 같은 사람 재안내 방지(쿨다운)는 B의 몫.
- `needs_face_hint(track_id, now)`: 얼굴을 못 찾은 지 `FACE_HINT_SEC`(2초) 이상 `UNKNOWN_TIMEOUT_SEC`(5초) 미만이면 `True`. 모르는 추적 번호면 `False`.
- `reset_all()`: 모든 상태 삭제 (테스트용).

### 6.3 판정 속도
`vote()`는 얼굴 인식을 한 프레임에서만 불린다(`FACE_INTERVAL`=5 프레임마다). 그래서 판정까지 약 `5 × 4 = 20`프레임 ≈ 30fps 기준 0.7초, 실제 처리 속도에 따라 더 걸린다. **2초보다 느리면 `FACE_INTERVAL`을 3으로** 줄여 본다.

### 6.4 B의 메인 루프에서 쓰는 법 (인수인계서 4.4)

```python
from vision.face import init_recognizer, recognize
from vision.judge import vote, needs_face_hint

init_recognizer()                                      # 프로그램 시작 시 1번

while True:
    frame = camera.read()
    now = time.time()
    for person in detect_people(frame):                # B
        if frame_no % FACE_INTERVAL == 0:
            result = recognize(frame, person)          # A
            decision = vote(person.track_id, result, now)   # A
            if decision and should_announce(decision, now):  # B
                handle(decision, frame)                # B
        if needs_face_hint(person.track_id, now):      # A
            show_face_hint()                           # B
```

---

## 7. 얼굴 등록 도구 `vision/register.py`

### 7.1 웹캠 등록 순서
1. 웹캠 화면에 찍을 자세를 영문으로 표시한다 (`1/5 FRONT`). 한글 안내는 콘솔에 나온다.
2. 화면에 얼굴 박스가 보인다. 얼굴이 1개면 초록, 아니면 빨강.
3. **스페이스바**를 누르면 그 순간의 프레임을 검사한다. 얼굴이 **정확히 1개**가 아니면 `REJECTED (n faces)`를 띄우고 다시 찍게 한다.
4. 5장을 다 찍으면 사진마다 `extract_embedding()`으로 벡터를 뽑는다.
5. 사진을 `data/faces/<이름>/<번호>.jpg`에 저장한다 (`imwrite_unicode` 사용).
6. `vision/facedb.py`가 C의 `register_person()` (이미 있는 이름이면 `add_embeddings()`)로 저장한다. 사진 경로는 `DATA_DIR` 기준 상대 경로 (`faces/홍길동/1.jpg`).
7. 등록자 번호와 추가한 벡터 수를 출력한다.

| 번호 | 화면 표시 | 촬영 조건 |
|---|---|---|
| 1 | `FRONT` | 정면, 평소 조명 |
| 2 | `LEFT 15-30 DEG` | 왼쪽으로 15~30° |
| 3 | `RIGHT 15-30 DEG` | 오른쪽으로 15~30° |
| 4 | `FRONT DARK` | 정면, 조금 어두운 조명 |
| 5 | `FRONT FAR` | 정면, 카메라에서 조금 멀리 |

### 7.2 그 밖의 동작
- `--folder`: 폴더 안의 `.jpg .jpeg .png .bmp`를 읽어 **얼굴이 정확히 1개인 사진만** 쓰고 나머지는 이유와 함께 건너뛴다. 쓴 사진은 `data/faces/<이름>/`로 복사된다.
- 같은 이름이 이미 있는데 `--add`가 없으면 "사진을 추가할까요? (y/n)"를 묻는다.
- `--add`로 추가하면 사진 번호가 기존 번호 다음부터 이어진다 (덮어쓰지 않음).
- `--delete`는 `y`를 입력해야만 진행하고, 등록 정보와 `data/faces/<이름>/` 폴더를 함께 지운다.
- 등록 후에는 메인 프로그램을 다시 시작하거나 `reload_known_faces()`를 호출해야 반영된다.

### 7.3 저장소 선택
- C의 `database.py`가 있으면 → `vision/facedb.py`를 통해 C의 `init_db()`, `register_person()`, `add_embeddings()`, `list_persons()`, `delete_person()`만 호출한다 (SQL 직접 사용 안 함). 콘솔에 `[register] DB (database), DB 파일 data/visitor.db, model_key=sface_2021dec`가 나온다.
- 없으면 → 임시 파일 `experiments/known_faces.npz`에 저장한다. 콘솔에 `[register] C의 database.py가 아직 없어 임시 파일...`이 나온다.

### 7.4 주의
- 시연 장소와 **같은 카메라**로 찍는 것이 가장 좋다.
- 등록 전에 **본인 동의**를 받는다. `data/`는 GitHub에 올리지 않는다.

---

## 8. 기준값 실험 `experiments/`

기준값은 감으로 정하지 않고 실험으로 정한다. 결과는 그대로 **보고서의 핵심 자료**가 된다.

### 8.1 준비
팀원 3명 중 **2명만 등록**하고 1명은 미등록자 역할을 한다. 끝나면 역할을 바꿔 한 번 더 한다.

### 8.2 측정 `collect_distances.py`

```powershell
python experiments/collect_distances.py --who 홍길동 --condition front_bright
python experiments/collect_distances.py --who unknown --condition front_bright   # 미등록자 역할
```

- 자세를 잡고 **스페이스바**를 누르면 얼굴이 잡힌 프레임 20개(`--frames`로 변경)를 측정한다. `ESC`는 취소(저장 안 함).
- `--who`는 등록자 이름 또는 `unknown`. 등록자가 아닌 이름을 넣으면 거부한다.
- 결과는 `experiments/distances.csv`에 **추가**된다 (Excel에서 한글이 안 깨지도록 UTF-8 BOM).

| 조건 이름 | 내용 |
|---|---|
| `front_bright` | 정면, 밝은 조명 |
| `side_bright` | 좌우 30°, 밝은 조명 |
| `front_dark` | 정면, 어두운 조명 |
| `front_dark_clahe` | 정면, 어두운 조명, **CLAHE 적용** (이 조건만 자동으로 CLAHE 켬) |
| `far` | 정면, 평소보다 먼 거리 |

CSV 컬럼: `time, who, condition, nearest_name, nearest_distance, self_distance, min_other_distance`
- `self_distance`: 본인 등록 벡터와의 거리 (미등록자는 빈칸)
- `min_other_distance`: 본인이 아닌 등록자 중 가장 가까운 거리

### 8.3 분석 `plot_distances.py`

```powershell
python experiments/plot_distances.py                    # 기준값 = config.MATCH_THRESHOLD
python experiments/plot_distances.py --threshold 0.45   # 다른 기준값으로 표 계산
```

결과 (`experiments/plots/`):

| 파일 | 내용 |
|---|---|
| `hist_all.png` | 전체 조건의 본인 거리(파랑)·타인 거리(빨강) 히스토그램 + 기준값 선 + 타인 최솟값 선 |
| `hist_<조건>.png` | 조건별 히스토그램 |
| `summary.csv` | 보고서용 결과 정리 표 |

콘솔에는 아래 내용이 나온다.
- 조건별 표: 본인 거리 평균 / 타인 거리 최솟값 / 기준값에서 본인 인식 / 기준값에서 오인식
- **기준값 후보**: 타인 거리 최솟값보다 10% 아래 (`--margin`으로 변경), 그 값에서의 본인 인식률
- **CLAHE 비교**: `front_dark`와 `front_dark_clahe`의 본인 거리 평균 비교와 권장 사항

### 8.4 기준값 정하는 법
1. 히스토그램에서 **타인 거리 중 가장 작은 값**(미등록자 역할 포함)을 찾는다.
2. 그보다 여유 있게 작은 값(예: 10% 아래)을 후보로 잡는다 → 스크립트가 계산해 준다.
3. 그 후보에서 본인 인식이 충분한지 확인한다. 부족하면 등록 사진을 보강하고 다시 측정한다.
4. 최종값을 `config.py`의 `MATCH_THRESHOLD`에 넣고 팀에 알린다.

### 8.5 판정 규칙 테스트 `test_judge.py`
카메라 없이 **가짜 `RecognitionResult`와 가짜 시각**을 `vote()`에 넣어 규칙대로 판정되는지 자동으로 검사한다.

| 테스트 | 확인 내용 |
|---|---|
| `test_registered_after_vote_min` | 같은 등록자 4번이면 `registered`, 3번까지 보류. 거리는 그중 최솟값 |
| `test_mixed_stays_pending` | 결과가 섞이면 보류 |
| `test_window_slides` | 최근 5개 안에서만 셈 |
| `test_unregistered` | 미등록 4번이면 `unregistered` |
| `test_no_face_hint_then_unknown` | 얼굴 없으면 2초부터 안내, 5초에 `unknown` |
| `test_face_lost_after_found` | 얼굴을 놓친 경우 마지막으로 찾은 시각부터 5초 |
| `test_tracks_independent` | 추적 번호별 결과가 섞이지 않음 |
| `test_reset_after_decision` | 판정 후 처음부터 다시 모음 |
| `test_stale_track_removed` | 10초 넘게 안 보인 추적 번호는 삭제 |

- `config.py`의 판정 값이나 `judge.py`를 바꿨을 때 돌린다. 끝에 `9/9 통과`면 정상.
- 테스트 안의 시각은 기본값(`VOTE_WINDOW=5, VOTE_MIN=4, FACE_HINT_SEC=2, UNKNOWN_TIMEOUT_SEC=5`) 기준이라 값을 바꾸면 테스트도 고쳐야 할 수 있다.

---

## 9. 설정값 `config.py`

### 9.1 A 담당 값

| 이름 | 값 | 의미 | 확정 시점 |
|---|---|---|---|
| `FACE_INTERVAL` | 5 | 몇 프레임마다 얼굴 인식 | 5주 |
| `MATCH_THRESHOLD` | **0.55** (임시) | 이 거리 이하면 등록자. 문서 기준 0.637보다 엄격하게 잡음 | 3주 (6주 재확인) |
| `USE_CLAHE` | False | 등록·인식 모두 CLAHE 적용 여부 | 3주 |
| `VOTE_WINDOW` | 5 | 최근 몇 번으로 판정 | 5주 |
| `VOTE_MIN` | 4 | 그중 몇 번 같아야 확정 | 5주 |
| `FACE_HINT_SEC` | 2 | 얼굴 못 찾은 지 몇 초부터 안내 | 4주 |
| `UNKNOWN_TIMEOUT_SEC` | 5 | 몇 초면 "인식 불가" 확정 | 4주 |

### 9.2 A가 **추가**한 값 (팀 공유 필요)

| 이름 | 값 | 의미 |
|---|---|---|
| `FACE_MODEL_KEY` | `sface_2021dec` | C의 DB에 저장하는 모델 이름. `USE_CLAHE`면 `_clahe`가 붙음 |
| `FACE_DET_MODEL` | `face_detection_yunet_2023mar.onnx` | 얼굴 검출 모델 파일 이름 (`MODELS_DIR` 안) |
| `FACE_REC_MODEL` | `face_recognition_sface_2021dec.onnx` | 특징 추출 모델 파일 이름 |
| `FACE_DET_SCORE` | 0.8 | YuNet 얼굴 검출 신뢰도 기준 |
| `MIN_FACE_WIDTH` | 40 | 이보다 작은 얼굴(픽셀)은 "얼굴 못 찾음" |
| `PERSON_BOX_MARGIN` | 0.1 | 사람 박스를 넓혀 자르는 비율 |
| `TRACK_EXPIRE_SEC` | 10 | 이 시간 넘게 갱신 안 된 추적 번호 삭제 |

"모든 숫자·경로·문구는 `config.py`에만 둔다"는 규칙에 따라 코드에 직접 쓰지 않고 여기에 추가했다.

---

## 10. 검증 결과

실행 환경: Windows 11, Python 3.13, OpenCV 5.0 (팀 표준은 Python 3.11, OpenCV 4.8 이상).
웹캠이 없는 환경이라 **OpenCV 샘플 사진**(`lena.jpg`, `messi5.jpg`)을 변형해 시험했다. 시험용 사진과 등록 데이터는 끝난 뒤 지웠다.

### 10.1 얼굴 인식

| 시험 | 결과 |
|---|---|
| 모델 불러오기 (한글 경로) | 성공 |
| 등록: 원본·어둡게·8° 회전·축소·흐림 5장 | 5장 모두 등록, 벡터 5개 |
| 등록 거부: 얼굴 0개 사진, 얼굴 2개 사진 | 둘 다 건너뜀 (인수인계서 7장 테스트 항목) |
| 같은 사람(좌우 반전·밝게 한 사진) 인식 | 등록자로 인식, 거리 **0.091** |
| 다른 사람과의 거리 | **0.83 ~ 0.86** |
| 얼굴 없는 이미지 `extract_embedding` | `None` |
| 60×60으로 줄인 사진 (얼굴 폭 40px 미만) | `face_found=False` |
| 사람 박스가 프레임 밖으로 나간 경우 | 정상 인식 (잘라서 처리) |
| 등록자 0명일 때 `compare` | `(None, None, inf)` |
| CLAHE 켠 상태 인식 | 정상 (거리 0.130) |
| `recognize()` → `vote()` 4번 연결 | `registered`, 이름·거리 정상 |
| 처리 시간 | `recognize()` 1회 평균 **약 27ms** |

### 10.2 등록 도구
| 시험 | 결과 |
|---|---|
| 한글 이름(`레나`, `메시`) 등록, `data/faces/<한글>/` 저장 | 성공 |
| 같은 이름 다시 등록 → 추가 질문에 `n` | 등록하지 않음 |
| `--list` | 이름·벡터 수·대표 사진 표시 |
| `--delete` + `y` | 등록 정보와 사진 폴더 삭제 |

### 10.3 판정·실험
| 시험 | 결과 |
|---|---|
| `test_judge.py` | **9/9 통과** |
| `plot_distances.py` (가짜 CSV 300줄) | PNG 6개 + `summary.csv` 생성, 기준값 후보·CLAHE 비교 출력 정상 |

### 10.4 C의 DB와 연결 (`sqlite_1` 브랜치의 `database.py`, `schema.sql`을 시험용 폴더에 복사해 확인)
| 시험 | 결과 |
|---|---|
| 폴더 사진 5장 등록 | DB에 `persons` 1행, `face_embeddings` 5행 (`model_key=sface_2021dec`, 128차원) |
| 같은 이름 `--add` 2장 | 같은 등록자 번호에 벡터 추가 → 7개 |
| 같은 이름 `--add` 없이 → `n` | 등록하지 않음 |
| `--list` | 이름·벡터 7개·등록일·대표 사진 표시 |
| DB에서 불러와 인식 | 등록자 1, `정재열`로 인식 |
| `database.py`를 `db/`로 옮김 | 같은 DB(`data/visitor.db`)를 그대로 사용 |
| `USE_CLAHE = True` | 다른 `model_key`라 벡터 0개 (섞이지 않음) |
| `--delete` + `y` | DB와 사진 폴더 삭제 |

### 10.5 아직 시험하지 못한 것
- 실제 웹캠으로 `--demo`, 웹캠 등록, `collect_distances.py` 실행
- 실제 팀원 얼굴로 거리 측정 (위 숫자는 샘플 사진 결과라 **기준값 결정에 쓰면 안 됨**)
- B의 `detect_people()`와 연결
- C의 DB는 `sqlite_1` 브랜치 파일을 시험용 폴더에 가져와 등록·추가·목록·인식·삭제를 확인함 (10.4). 실제 브랜치 병합 후 재확인 필요

---

## 11. 팀원에게 알릴 것

| 대상 | 내용 |
|---|---|
| 팀 전체 | 얼굴 인식 라이브러리 = **OpenCV 내장 YuNet + SFace**. `requirements.txt`에 추가할 패키지 없음 |
| 팀 전체 | `MATCH_THRESHOLD` 임시값 **0.55** (단위: 1 − 코사인 유사도) |
| 팀 전체 | `config.py`에 A 값 7개 추가 ([9.2절](#92-a가-추가한-값-팀-공유-필요)) |
| B | `config.py`, `common/types.py`, `common/imageio.py`, `requirements.txt`, `.gitignore`를 명세대로 먼저 만들어 둠. 그대로 써도 되고 B 버전으로 바꿔도 됨 (A 값만 유지) |
| B | 메인 루프에서 `init_recognizer()` → `recognize()` → `vote()` → `needs_face_hint()` ([6.4절](#64-b의-메인-루프에서-쓰는-법-인수인계서-44)). 미등록자 중복 방지에는 `embedding_distance()` |
| B | `.gitignore`에 `experiments/known_faces.npz` 추가함 (얼굴 벡터라 개인정보) |
| C | A는 `sqlite_1` 브랜치의 **실제 함수 형태**에 맞춰 호출함 (명세 3.5와 다름 → 명세 수정 필요). `model_key="sface_2021dec"`, `db_path=config.DB_PATH`를 넘김. 파일은 `db/database.py`·루트 어디에 있어도 됨. 이름 중복 방지(UNIQUE)는 없으니 A가 같은 이름이면 `add_embeddings()`로 추가함 |
| C | 특징 벡터는 float32 128차원. DB에는 `tobytes()`로, 꺼낼 때 `np.frombuffer(..., dtype=np.float32)` |

---

## 12. 남은 일

| 주차 | 할 일 |
|---|---|
| 2주 | 웹캠으로 `--demo` 확인, 팀원 3명 등록 (본인 동의). C의 DB가 나오면 DB로 다시 등록 |
| 3주 | `collect_distances.py`로 5개 조건 측정 (2명 등록 + 1명 미등록, 역할 바꿔 반복) → `plot_distances.py` → `MATCH_THRESHOLD`·`USE_CLAHE` 확정 → 팀에 알림 |
| 4주 | B의 `main.py`와 연결해 등록자 → 문 열림, 미등록자 → 안내 확인 (**최소 완성본**). `FACE_HINT_SEC`·`UNKNOWN_TIMEOUT_SEC` 확정 |
| 5주 | 인식률·오인식률·판정 시간 측정 (인수인계서 7장), `VOTE_*`·`FACE_INTERVAL` 확정, 보고서 실험 부분 초안 |
| 6주 | 시연 장소 조명에서 재확인, 보고서·발표 |

인수인계서 7장 테스트 목표:

| 항목 | 목표 |
|---|---|
| 등록자 인식률 (팀원별 20회) | 정면·실내 조명에서 대부분 성공 |
| 미등록자 오인식률 (20회) | **0회** |
| 판정 시간 | 약 2초 이내 |
| 얼굴 못 찾음 | 2초부터 안내, 5초에 `unknown` |
| 여러 명 | 추적 번호별 결과가 섞이지 않음 |

---

## 13. 문제 해결

| 증상 | 원인·해결 |
|---|---|
| `모델 파일이 없습니다: models\...` | 프로젝트 루트에서 실행했는지 확인. 아니면 [3.2절](#32-설치)대로 모델을 받는다 |
| `카메라 0번을 열 수 없습니다` | 다른 프로그램(Zoom 등)이 카메라를 쓰는지 확인. `--camera 1` 시도 |
| 키를 눌러도 창이 안 꺼짐 | 영상 창을 클릭해 선택한 뒤 `q`/`ESC`. 안 되면 터미널에서 `Ctrl + C` |
| `[register] C의 database.py가 아직 없어 임시 파일...` | 정상. C의 `database.py`가 아직 저장소에 없을 때 npz를 쓴다 |
| DB 쪽 오류 (`지원하지 않는 DB 버전` 등) | C의 DB 문제. 오류 메시지를 C에게 전달 |
| 등록했는데 데모에서 미등록으로 나옴 | 거리가 `MATCH_THRESHOLD`보다 큰 경우. 화면의 `nearest ... d=` 값 확인, 등록 사진 보강(`--add`) |
| `USE_CLAHE`를 바꾼 뒤 인식이 나빠짐 | 등록과 인식의 전처리가 달라졌기 때문. **전원 재등록** |
| 콘솔 한글이 깨짐 | PowerShell에서 `chcp 65001` 실행 후 다시 실행 |
| `setPreferableTarget ... not supported` 경고 | OpenCV 5.x에서만 나오는 경고로 동작에 문제 없음. 팀 표준(4.x)에서는 나오지 않음 |
| 휴대폰 사진으로도 인식됨 | 위조 방지는 이번 범위 밖 (계획서). 발표 질의응답에서 설명 |
