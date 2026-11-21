# B → C 공통 명세 요청 반영 결과

작성일: 2026-11-21 / 대상 로컬 브랜치: arduinoide_1

## 반영 내역

| 요청 | 결과 |
|---|---|
| DB 구현 위치 | `db/database.py`로 정리. 루트 database.py는 기존 import/실행용 호환 진입점만 유지 |
| save_visit 형식 | `save_visit(decision, message, frame, approval_status=None) -> visit_id` 반영 |
| 방문 사진 저장 | 공통 imwrite_unicode()로 `data/captures/YYYY-MM-DD/HHMMSS_<visit_id>.jpg` 저장 |
| DB 사진 경로 | `captures/...` 상대 경로 저장. get_records 키는 visits 컬럼 이름 그대로 유지 |
| 미등록 벡터 | Decision.embedding을 방문과 함께 저장. 없는 경우 방문/사진만 저장 |
| 저장 시간 | 모든 시간 컬럼을 한국 시간 `YYYY-MM-DD HH:MM:SS`로 통일 |
| 방문 조회 | `get_records(category=None, date=None, name=None, limit=200)` 반영 |
| 미등록자 중복 확인 | `recent_unknown_embeddings(minutes=10)`에서 np.float32 1차원 배열 목록 반환 |
| 문 이벤트/통계 | save_door_event(event, visit_id), get_today_stats()의 호출 형식·결과 키 유지 |
| 아두이노 위치 | `arduino/door_controller/door_controller.ino`로 변경 |
| 반복 OPEN | 문이 열린 동안 다시 받으면 마지막 수신부터 30초를 다시 계산 |
| 통과 센서 | 요청대로 보류. 센서/PASS/통과 후 3초 닫힘은 구현하지 않음 |
| HINT | 구현. 노랑 LED 깜빡임. 이전 RETRY 명령도 같은 동작으로 허용 |

추가로 B의 Door가 마지막 OPEN의 방문 번호를 사용하므로, 닫히는 중 새 OPEN을 받으면
현재 각도에서 다시 열도록 처리했습니다. 전원 투입 직후 초기 닫힘 중에는 READY를 기다립니다.

## 오류 방지와 기존 데이터

- 사진 저장이나 사진 경로 DB 갱신에 실패하면 방문·임시 벡터를 함께 취소하고 새 사진을 정리합니다.
- 같은 경로에 기존 사진이 있으면 덮어쓰지 않고 오류를 반환합니다.
- A가 등록 얼굴 없음으로 전달하는 +inf 거리는 미등록/인식 불가 기록에서 NULL로 보관합니다.
- 기존 A의 등록자 관리 함수는 그대로 유지합니다. model_key=를 받는 load_embeddings()는 dict 목록입니다.
- 미등록 벡터 조회는 B 요청대로 ndarray 목록이며 등록 벡터 조회와 반환 형식이 다릅니다.
- 모델 키 기본값은 실제 A 코드의 sface_2021dec/CLAHE 설정에 맞췄습니다.
  모델을 바꾸면 save_visit()/recent_unknown_embeddings()의 선택 인수 model_key=에 실제 키를 전달할 수 있습니다.
- v1 DB는 다음 init_db()에서 원본 백업 후 v2로 시간 형식을 변환합니다.
  백업은 DB 옆 backups/*.backup.db, 변환 실패 시 원본은 rollback됩니다.
- 검증에는 임시 DB/사진을 사용했으며 기존 data/visitor.db는 변경하지 않았습니다.

## 검증 결과

| 검증 | 결과 |
|---|---|
| Python | 3.13.2 전용 가상환경 |
| 의존성 | B requirements의 NumPy 2.5.4 / OpenCV 5.0.0.93 |
| DB 테스트 | 27개 통과. JPEG·한글 경로·float32·날짜 경계·실패 시 취소·백업/변환 등 |
| 실제 A 코드 호출 | ab/develop의 vision.face._CStore로 등록·추가·조회·삭제 확인 |
| 실제 B 호출 형식 | b/develop의 main.handle 함수를 원문에서 추출해 DB 저장/문 콜백 호출 확인 |
| B 대시보드 API | category/date/name/limit 조회와 통계 키 확인 |
| Arduino UNO | CLI 1.6.0 / AVR Boards 1.8.8 / Servo 1.3.0, 컴파일 성공 |
| 보드 자원 | 플래시 4412 bytes (13%), 전역 RAM 307 bytes (14%) |

A/B 확인은 카메라·얼굴 모델·전체 앱 실행 없이 DB 호출 경계를 확인한 스모크 테스트입니다.
실물 서보·LED, 반복 OPEN의 실제 30초 타이밍, USB 연결과 전체 카메라 흐름은 함께 시험해야 합니다.

## B가 연결할 때

```python
from db import database as db

db.init_db()
save_visit = db.save_visit
save_door_event = db.save_door_event

# 기존 B handle() 호출 그대로 사용 가능
# visit_id = save_visit(decision, message, frame)
# door.on_event(lambda event, visit_id: save_door_event(event, visit_id))

rows = db.get_records(category=None, date=None, name=None, limit=200)
stats = db.get_today_stats()
recent_vectors = db.recent_unknown_embeddings(minutes=10)
```

- B Door.open()은 전송 성공 시 이미 open 콜백을 보냅니다. OPENED를 같은 open 이벤트로 다시 기록하지 않습니다.
- 센서가 없으므로 실물 보드는 PASS를 보내지 않습니다. CLOSED만으로 완료 처리하고 passed_at은 NULL로 둡니다.
- B의 FakeDoor는 현재 가짜 pass를 생성합니다. 센서 없는 실물 동작과 같은 시연을 원하면 B에서 별도로 조정해야 합니다.
- 보드 재시작/재연결 후 현재 방문 번호 처리와 시리얼 모니터 COM 포트 점유 여부도 연결 시험에서 확인하세요.
- 실제 B main/dashboard의 가짜 함수 교체는 이 C 브랜치에서 수정하지 않았습니다.

## 실행 방법

프로젝트 환경을 활성화한 터미널에서:

```powershell
python -m pip install -r requirements-db.txt
python -m db.database
python -m unittest discover -s tests -v
python db_example.py
```

아두이노는 새 위치의 door_controller.ino를 Arduino IDE에서 열어 Arduino Uno/COM 포트를 선택하고 업로드합니다.
시리얼은 9600bps, 줄바꿈 사용입니다.

## 확인 근거와 범위

B 요청 메시지와 GitHub b/develop의 공통 타입·이미지 함수·호출 코드를 기준으로 반영했습니다.
로컬 C 브랜치에 없던 common/types.py와 common/imageio.py는 B의 내용을 그대로 가져왔습니다.
작업.md 원문은 확인한 로컬/원격 브랜치에 없어 해당 문서 전체의 추가 규칙까지 검증한 것은 아닙니다.
다른 브랜치를 병합하거나 B의 실행 코드를 덮어쓰지 않았으며, 커밋·푸시도 하지 않았습니다.
