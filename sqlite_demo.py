"""SQLite 기초 예제: 테이블 생성 -> 방문 기록 저장 -> 조회.

실행: python sqlite_demo.py
실행할 때마다 연습용 방문 기록이 한 건 추가됩니다.
"""

from datetime import datetime
from pathlib import Path
import sqlite3


# 터미널의 현재 위치와 관계없이 프로젝트의 data 폴더를 사용합니다.
DB_PATH = Path(__file__).resolve().parent / "data" / "demo.db"


def init_db(conn: sqlite3.Connection) -> None:
    """연습용 방문 기록 테이블을 만듭니다. 기존 테이블은 유지합니다."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY,
            visited_at TEXT NOT NULL,
            category TEXT NOT NULL CHECK (
                category IN ('registered', 'unregistered', 'unknown')
            ),
            person_name TEXT,
            message TEXT NOT NULL
        )
    """)


def save_visit(conn: sqlite3.Connection, name: str) -> int:
    """가상의 등록자 방문 기록 한 건을 저장하고 기록 번호를 반환합니다."""
    cursor = conn.execute(
        """INSERT INTO visits (visited_at, category, person_name, message)
           VALUES (?, ?, ?, ?)""",
        (
            datetime.now().astimezone().isoformat(timespec="seconds"),
            "registered",
            name,
            f"{name}님 환영합니다.",
        ),
    )
    return cursor.lastrowid


def get_records(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """방문 기록을 최신순으로 조회합니다."""
    return conn.execute("SELECT * FROM visits ORDER BY id DESC").fetchall()


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        # 블록이 정상 종료되면 저장(commit), 실패하면 취소(rollback)합니다.
        with conn:
            init_db(conn)
            visit_id = save_visit(conn, "연습용 방문자")

        print(f"DB 파일: {DB_PATH}")
        print(f"저장한 기록 번호: {visit_id}")
        print("방문 기록 (최신순):")
        for record in get_records(conn):
            print(dict(record))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
