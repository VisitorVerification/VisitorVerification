# SQLite 첫 실습

이 예제는 실제 얼굴 정보 없이 가상 방문 기록으로 SQLite의 기본 동작을 확인합니다.
계획서의 전체 DB 설계는 다음 단계에서 별도 구현합니다.

## 실행 방법

1. VS Code에서 VisitorVerification 폴더를 엽니다.
2. 터미널 > 새 터미널을 선택합니다.
3. Python 3.10 이상인지 확인합니다: `py --version`
4. 실행합니다: `py -X utf8 sqlite_demo.py`

py 명령이 인식되지 않으면 VS Code에서 Python 인터프리터와 설치 상태를 확인하세요.
추가 Python 패키지는 필요 없습니다. sqlite3는 Python 기본 모듈입니다.

## 실행 결과

- `data/demo.db` 파일이 자동 생성됩니다.
- 파일 안에 `visits` 테이블이 생성됩니다.
- 가상의 등록자 방문 기록 한 건이 저장됩니다.
- 저장된 모든 방문 기록이 터미널에 최신순으로 출력됩니다.
- 다시 실행하면 기존 기록을 유지하고 한 건을 추가합니다.
- VS Code 실행 폴더가 달라도 DB 파일은 이 스크립트 옆 data 폴더에 생성됩니다.

## 코드 읽는 순서

1. `DB_PATH`: 데이터를 저장할 파일 위치
2. `init_db()`: CREATE TABLE로 테이블 생성
3. `save_visit()`: INSERT로 방문 기록 추가
4. `with conn`: 성공 시 변경 내용을 저장하고 실패 시 취소
5. `get_records()`: SELECT로 저장한 데이터 조회
6. `conn.close()`: DB 연결 종료

이 예제의 함수와 테이블은 학습용이며 팀 공용 인터페이스가 아닙니다.
DB 파일과 방문 사진은 Git에 올리지 않고, 코드와 문서만 공유합니다.

