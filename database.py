"""기존 import database 및 실행 명령을 위한 호환 진입점.

공용 구현은 db/database.py에 있습니다. 새 코드는 from db import database를 사용하세요.
save_visit 등 함수의 인수는 변경된 공통 명세를 따릅니다.
"""
from db.database import *  # noqa: F401,F403


if __name__ == "__main__":
    from db.database import main
    main()
