"""임시 DB로 B의 Decision/사진 저장/조회 호출을 시험합니다. 실제 DB는 변경하지 않습니다."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from common.types import Decision
from db import database as db


def main():
    with TemporaryDirectory() as folder:
        path = Path(folder) / "example.db"
        db.init_db(path)
        # 설명용 가짜 숫자 벡터이며 실제 얼굴 특징이 아닙니다.
        person_id = db.register_person("연습용 방문자", [[0.1, 0.2, 0.3]],
                                      model_key="sface_2021dec", db_path=path)
        frame = np.full((120, 160, 3), 128, dtype=np.uint8)
        decision = Decision("registered", 1, person_id=person_id, name="연습용 방문자", distance=0.2)
        visit_id = db.save_visit(decision, "연습용 방문자님 환영합니다.", frame, db_path=path)
        db.save_door_event("open", visit_id, db_path=path)
        # 현재 센서가 없으므로 pass는 만들지 않고, 자동 닫힘만 기록합니다.
        db.save_door_event("close", visit_id, db_path=path)
        unknown = Decision("unregistered", 2, embedding=np.array([0.4, 0.5, 0.6], dtype=np.float32))
        db.save_visit(unknown, "카운터에 가서 문의하세요.", frame, db_path=path)
        for row in db.get_records(db_path=path):
            print(row)
            print("사진 존재:", (path.parent / row["image_path"]).is_file())
        print("중복 확인용 벡터:", db.recent_unknown_embeddings(db_path=path))
        print("오늘 통계:", db.get_today_stats(db_path=path))
    print("임시 DB/사진 예제 완료. 실제 visitor.db는 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
