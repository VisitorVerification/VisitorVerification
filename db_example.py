"""실제 DB를 건드리지 않는 공용 함수 사용 예제. 임시 DB는 실행 후 삭제됩니다."""

from pathlib import Path
from tempfile import TemporaryDirectory
import database as db


def main():
    with TemporaryDirectory() as folder:
        path = Path(folder) / "example.db"
        db.init_db(path)
        # 이 3차원 벡터는 설명용 숫자이며 실제 얼굴 특징이 아닙니다.
        person_id = db.register_person(
            "연습용 방문자", [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
            model_key="example-only-v1", db_path=path,
        )
        visit_id = db.save_visit("registered", person_id=person_id, db_path=path)
        db.save_door_event("open", visit_id, event_key="example-open", db_path=path)
        db.update_passed(visit_id, event_key="example-pass", db_path=path)
        db.save_door_event("close", visit_id, db_path=path)
        db.save_visit("unregistered", db_path=path)
        db.save_visit("unknown", db_path=path)
        print("등록자:", db.list_persons(db_path=path))
        print("방문 기록:")
        for record in db.get_records(db_path=path):
            print(record)
        print("문 동작:", db.get_door_events(visit_id, db_path=path))
        print("오늘 통계:", db.get_today_stats(db_path=path))
    print("예제 완료. 임시 DB는 삭제되었고 실제 visitor.db는 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
