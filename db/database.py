"""A/B/C 공통 명세에 맞춘 SQLite 저장·조회 함수.

초기화: python -m db.database
함수는 연결을 직접 열고 닫으므로 Flask와 영상 처리 프로그램에서 각각 호출합니다.
"""

from contextlib import closing, contextmanager
from datetime import date as Date, datetime, time, timedelta, timezone
import importlib
import logging
from pathlib import Path
import math
import sqlite3
import struct

import numpy as np

from common import imageio
from common.types import Decision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "visitor.db"
SCHEMA_PATH = PROJECT_ROOT / "schema.sql"
SCHEMA_VERSION = 2
KST = timezone(timedelta(hours=9))
CATEGORIES = {"registered", "unregistered", "unknown"}
log = logging.getLogger(__name__)


def _moment(value=None):
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
    value = datetime.now(KST) if value is None else value
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("시간은 시간대가 포함된 datetime으로 전달하세요.")
    return value.astimezone(KST)


def _stamp(value=None):
    return _moment(value).strftime("%Y-%m-%d %H:%M:%S")


def _active_model_key(model_key=None):
    """명시한 키 우선. 기본값은 A의 SFace/CLAHE 설정과 맞춥니다."""
    if model_key is not None:
        return _text(model_key, "model_key")
    try:
        config = importlib.import_module("config")
    except ModuleNotFoundError as exc:
        if exc.name != "config":
            raise
        return "sface_2021dec"
    key = getattr(config, "FACE_MODEL_KEY", "sface_2021dec")
    return _text(key, "FACE_MODEL_KEY") + ("_clahe" if getattr(config, "USE_CLAHE", False) else "")


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: 빈 문자열은 사용할 수 없습니다.")
    return value.strip()


def _pack(vector):
    try:
        values = [float(item) for item in vector]
        if not values or not all(math.isfinite(item) for item in values):
            raise ValueError
        blob = struct.pack(f"<{len(values)}f", *values)
        # Overflow can also round finite Python floats to infinity in float32.
        if not all(math.isfinite(item) for item in struct.unpack(f"<{len(values)}f", blob)):
            raise ValueError
    except (TypeError, ValueError, OverflowError, struct.error) as exc:
        raise ValueError("벡터는 float32로 표현할 수 있는 유한한 숫자의 1차원 목록이어야 합니다.") from exc
    return len(values), blob


def _decode(row):
    result = dict(row)
    result["embedding"] = list(struct.unpack(f"<{result['dimension']}f", result["embedding"]))
    return result


def _connect(db_path):
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def _connection(db_path=DEFAULT_DB_PATH):
    if not Path(db_path).is_file():
        raise FileNotFoundError("DB가 없습니다. 먼저 init_db()를 호출하세요.")
    conn = _connect(db_path)
    try:
        if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise RuntimeError("지원하지 않는 DB 버전입니다. 기존 DB를 덮어쓰지 않습니다.")
        with conn:
            yield conn
    finally:
        conn.close()


def _migrate_v1(conn, path):
    """기존 UTC 시각을 KST로 변환합니다. 백업을 만든 뒤 한 번에 갱신합니다."""
    conn.execute("BEGIN IMMEDIATE")
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{path.stem}.v1-{datetime.now(KST):%Y%m%d-%H%M%S-%f}.backup.db"
    # 쓰기 잠금을 유지하고 별도 읽기 연결의 backup API로 변환 전 상태를 보관합니다.
    with closing(_connect(path)) as source, closing(sqlite3.connect(backup_path)) as target:
        source.backup(target)
    columns = {
        "persons": ("created_at",),
        "face_embeddings": ("created_at",),
        "visits": ("visited_at", "decided_at", "passed_at"),
        "visit_embeddings": ("expires_at",),
        "door_events": ("event_at",),
    }
    for table, fields in columns.items():
        # 테이블/컬럼 이름은 위의 고정된 값만 사용합니다.
        for field in fields:
            rows = conn.execute(f"SELECT rowid, {field} FROM {table} WHERE {field} IS NOT NULL").fetchall()
            for row in rows:
                value = datetime.fromisoformat(row[1].replace("Z", "+00:00"))
                if value.utcoffset() is None:
                    value = value.replace(tzinfo=KST)
                conn.execute(f"UPDATE {table} SET {field} = ? WHERE rowid = ?", (_stamp(value), row[0]))
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    log.info("DB v1 → v2 완료. 변환 전 백업: %s", backup_path)


def init_db(db_path=DEFAULT_DB_PATH):
    """새 DB와 테이블 생성. 같은 버전의 기존 DB/데이터는 유지합니다."""
    path = Path(db_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            conn.execute("PRAGMA journal_mode = WAL")
            return path
        if version == 1:
            conn.execute("PRAGMA journal_mode = WAL")
            _migrate_v1(conn, path)
            return path
        if version != 0:
            raise RuntimeError(f"지원하지 않는 DB 버전: {version}")
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if tables:
            raise RuntimeError("버전 없는 기존 DB입니다. demo.db와 분리된 새 경로를 사용하세요.")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8-sig"))
        return path
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _check_dimension(conn, model_key, dimension):
    rows = conn.execute(
        """SELECT dimension FROM face_embeddings WHERE model_key = ?
           UNION SELECT dimension FROM visit_embeddings WHERE model_key = ?""",
        (model_key, model_key),
    ).fetchall()
    if any(row[0] != dimension for row in rows):
        raise ValueError("같은 model_key의 벡터 길이가 다릅니다. 모델/전처리 버전을 확인하세요.")


def register_person(name, embeddings, *, model_key, photo_path=None, db_path=DEFAULT_DB_PATH):
    """등록자와 한 개 이상의 얼굴 벡터를 함께 저장하고 person_id를 반환합니다."""
    name, model_key = _text(name, "name"), _text(model_key, "model_key")
    vectors = [_pack(vector) for vector in embeddings]
    if not vectors or len({dimension for dimension, _ in vectors}) != 1:
        raise ValueError("동일한 길이의 얼굴 벡터를 한 개 이상 전달하세요.")
    created_at = _stamp()
    with _connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _check_dimension(conn, model_key, vectors[0][0])
        person_id = conn.execute(
            "INSERT INTO persons(name, photo_path, created_at) VALUES (?, ?, ?)",
            (name, photo_path, created_at),
        ).lastrowid
        conn.executemany(
            """INSERT INTO face_embeddings
               (person_id, model_key, dimension, embedding, created_at) VALUES (?, ?, ?, ?, ?)""",
            [(person_id, model_key, dim, blob, created_at) for dim, blob in vectors],
        )
        return person_id


def add_embeddings(person_id, embeddings, *, model_key, db_path=DEFAULT_DB_PATH):
    """기존 등록자에게 얼굴 벡터를 추가합니다."""
    model_key = _text(model_key, "model_key")
    vectors = [_pack(vector) for vector in embeddings]
    if not vectors or len({dim for dim, _ in vectors}) != 1:
        raise ValueError("동일한 길이의 벡터가 한 개 이상 필요합니다.")
    with _connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _check_dimension(conn, model_key, vectors[0][0])
        conn.executemany(
            """INSERT INTO face_embeddings
               (person_id, model_key, dimension, embedding, created_at) VALUES (?, ?, ?, ?, ?)""",
            [(person_id, model_key, dim, blob, _stamp()) for dim, blob in vectors],
        )
    return len(vectors)


def list_persons(*, db_path=DEFAULT_DB_PATH):
    with _connection(db_path) as conn:
        return [dict(row) for row in conn.execute(
            """SELECT p.*, COUNT(f.id) AS embedding_count FROM persons p
               LEFT JOIN face_embeddings f ON f.person_id = p.id
               GROUP BY p.id ORDER BY p.id"""
        )]


def load_embeddings(*, model_key, dimension=None, db_path=DEFAULT_DB_PATH):
    """같은 모델의 등록 벡터만 반환. embedding은 Python float 목록입니다."""
    params = [_text(model_key, "model_key")]
    sql = """SELECT f.*, p.name AS person_name FROM face_embeddings f
             JOIN persons p ON p.id = f.person_id WHERE f.model_key = ?"""
    if dimension is not None:
        sql += " AND f.dimension = ?"
        params.append(dimension)
    with _connection(db_path) as conn:
        return [_decode(row) for row in conn.execute(sql + " ORDER BY f.id", params)]


def delete_person(person_id, *, db_path=DEFAULT_DB_PATH):
    """등록자와 등록 벡터를 삭제하고 과거 방문의 이름을 지웁니다. 사진 파일은 별도입니다."""
    with _connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE visits SET person_name = NULL, message = '등록자 방문 기록입니다.' WHERE person_id = ?",
            (person_id,),
        )
        return conn.execute("DELETE FROM persons WHERE id = ?", (person_id,)).rowcount == 1


def save_visit(decision: Decision, message: str, frame: np.ndarray, approval_status=None, *,
               model_key=None, visited_at=None, db_path=DEFAULT_DB_PATH, data_dir=None):
    """B의 Decision·안내 문구·원본 BGR 프레임을 받아 저장하고 visit_id 반환."""
    category = decision.category
    person_id = decision.person_id
    face_distance = decision.distance
    embedding = decision.embedding if category == "unregistered" else None
    if category == "unknown" and decision.embedding is not None:
        raise ValueError("인식 불가 판정에는 얼굴 벡터가 없어야 합니다.")
    if category not in CATEGORIES:
        raise ValueError("category는 registered / unregistered / unknown이어야 합니다.")
    status = approval_status or ("auto" if category == "registered" else "counter")
    if (category == "registered" and (person_id is None or status != "auto")) or (
        category != "registered" and (person_id is not None or status not in {"counter", "pending"})
    ):
        raise ValueError("구분·등록자 번호·초기 승인 상태의 조합이 올바르지 않습니다.")
    if face_distance is not None:
        face_distance = float(face_distance)
        # A는 등록 얼굴이 없으면 +inf를 반환합니다. 비교 대상 없음은 SQL NULL로 저장합니다.
        if math.isinf(face_distance) and face_distance > 0 and category != "registered":
            face_distance = None
        elif not math.isfinite(face_distance) or face_distance < -1e-6:
            raise ValueError("face_distance는 유한한 0 이상의 숫자여야 합니다.")
        else:
            face_distance = max(0.0, face_distance)  # 코사인 거리의 작은 부동소수점 오차 보정
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.size == 0:
        raise ValueError("frame은 비어 있지 않은 uint8 BGR 영상이어야 합니다.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame은 높이×너비×3의 BGR 영상이어야 합니다.")
    message = _text(message, "message")
    packed = None
    if embedding is not None:
        model_key = _active_model_key(model_key)
        packed = _pack(embedding)
    instant = _moment(visited_at)
    image_file = None
    created_image = False
    try:
        with _connection(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            name = None
            if category == "registered":
                row = conn.execute("SELECT name FROM persons WHERE id = ?", (person_id,)).fetchone()
                if row is None:
                    raise ValueError("등록자 번호가 존재하지 않습니다.")
                name = decision.name if decision.name is not None else row[0]
            if packed:
                _check_dimension(conn, model_key, packed[0])
            visit_id = conn.execute(
                """INSERT INTO visits (visited_at, category, person_id, person_name,
                   face_distance, message, approval_status) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (_stamp(instant), category, person_id, name, face_distance, message, status),
            ).lastrowid
            if packed:
                dim, blob = packed
                conn.execute(
                    """INSERT INTO visit_embeddings
                       (visit_id, model_key, dimension, embedding, expires_at) VALUES (?, ?, ?, ?, ?)""",
                    (visit_id, model_key, dim, blob, _stamp(instant + timedelta(minutes=10))),
                )
            relative = Path("captures") / instant.strftime("%Y-%m-%d") / f"{instant:%H%M%S}_{visit_id}.jpg"
            root = Path(data_dir) if data_dir is not None else Path(db_path).resolve().parent
            image_file = root / relative
            image_file.parent.mkdir(parents=True, exist_ok=True)
            # 배타적으로 파일을 생성해 기존 사진 덮어쓰기를 방지합니다.
            with image_file.open("xb"):
                pass
            created_image = True
            if not imageio.imwrite_unicode(str(image_file), frame):
                raise OSError(f"방문 사진 저장에 실패했습니다: {image_file}")
            conn.execute("UPDATE visits SET image_path = ? WHERE id = ?", (relative.as_posix(), visit_id))
        return visit_id
    except BaseException:
        if created_image:
            try:
                image_file.unlink(missing_ok=True)
            except OSError:
                log.exception("실패한 방문의 사진 정리가 필요합니다: %s", image_file)
        raise


def get_visit(visit_id, *, db_path=DEFAULT_DB_PATH):
    with _connection(db_path) as conn:
        row = conn.execute("SELECT * FROM visits WHERE id = ?", (visit_id,)).fetchone()
        return dict(row) if row else None


def _day_bounds(day):
    day = Date.fromisoformat(day) if isinstance(day, str) else day
    if not isinstance(day, Date) or isinstance(day, datetime):
        raise ValueError("날짜는 YYYY-MM-DD 문자열 또는 date로 전달하세요.")
    start = datetime.combine(day, time.min, tzinfo=KST)
    return _stamp(start), _stamp(start + timedelta(days=1))


def get_records(category=None, date=None, name=None, limit=200, *, offset=0,
                db_path=DEFAULT_DB_PATH):
    """KST 날짜·구분·이름(부분 일치)으로 방문 조회. 최대 1000건씩 반환."""
    if type(limit) is not int or not 1 <= limit <= 1000 or type(offset) is not int or offset < 0:
        raise ValueError("limit은 1~1000 정수, offset은 0 이상 정수여야 합니다.")
    clauses, params = [], []
    if category is not None:
        if category not in CATEGORIES:
            raise ValueError("잘못된 category입니다.")
        clauses.append("category = ?")
        params.append(category)
    if date is not None:
        start, end = _day_bounds(date)
        clauses.append("visited_at >= ? AND visited_at < ?")
        params.extend([start, end])
    if name is not None:
        literal = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clauses.append("person_name LIKE ? ESCAPE '\\'")
        params.append(f"%{literal}%")
    sql = "SELECT * FROM visits"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY visited_at DESC, id DESC LIMIT ? OFFSET ?"
    with _connection(db_path) as conn:
        return [dict(row) for row in conn.execute(sql, params + [limit, offset])]


def get_today_stats(*, day=None, db_path=DEFAULT_DB_PATH):
    """KST 오늘(또는 지정일)의 세 구분별 방문 수와 합계."""
    day = datetime.now(KST).date() if day is None else day
    start, end = _day_bounds(day)
    result = {"registered": 0, "unregistered": 0, "unknown": 0}
    with _connection(db_path) as conn:
        for row in conn.execute(
            """SELECT category, COUNT(*) AS count FROM visits
               WHERE visited_at >= ? AND visited_at < ? GROUP BY category""", (start, end)
        ):
            result[row["category"]] = row["count"]
    result["total"] = sum(result.values())
    return result


def recent_unknown_embeddings(minutes=10, *, model_key=None, dimension=None, now=None, db_path=DEFAULT_DB_PATH):
    """최근 10분의 미등록 벡터만 반환합니다. 얼굴 추출 불가인 unknown은 제외합니다."""
    if type(minutes) is not int or minutes < 1:
        raise ValueError("minutes는 1 이상의 정수여야 합니다.")
    instant = _moment(now)
    sql = """SELECT e.*, v.visited_at FROM visit_embeddings e
             JOIN visits v ON v.id = e.visit_id
             WHERE e.model_key = ? AND e.expires_at > ? AND v.visited_at <= ? AND v.visited_at > ?"""
    params = [_active_model_key(model_key), _stamp(instant), _stamp(instant), _stamp(instant - timedelta(minutes=minutes))]
    if dimension is not None:
        sql += " AND e.dimension = ?"
        params.append(dimension)
    with _connection(db_path) as conn:
        return [np.asarray(_decode(row)["embedding"], dtype=np.float32)
                for row in conn.execute(sql + " ORDER BY v.visited_at DESC, e.visit_id DESC", params)]


def cleanup_expired_embeddings(*, now=None, db_path=DEFAULT_DB_PATH):
    """만료된 임시 벡터를 실제 삭제. 방문 기록과 사진은 유지합니다."""
    with _connection(db_path) as conn:
        return conn.execute("DELETE FROM visit_embeddings WHERE expires_at <= ?", (_stamp(now),)).rowcount


def save_door_event(event, visit_id=None, *, event_at=None, event_key=None,
                    db_path=DEFAULT_DB_PATH):
    """문 이벤트와 방문 요약을 함께 저장. event_key는 수신 재시도 중복 방지용."""
    if event not in {"open", "close", "pass"} or (event != "close" and visit_id is None):
        raise ValueError("open/pass에는 visit_id가 필요하며 event는 open/close/pass입니다.")
    if event_key is not None:
        event_key = _text(event_key, "event_key")
    stamp = _stamp(event_at)
    with _connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if event_key is not None:
            old = conn.execute("SELECT * FROM door_events WHERE event_key = ?", (event_key,)).fetchone()
            if old:
                if old["event"] != event or old["visit_id"] != visit_id:
                    raise ValueError("event_key가 다른 이벤트에 이미 사용되었습니다.")
                return old["id"]
        if visit_id is not None:
            visit = conn.execute("SELECT * FROM visits WHERE id = ?", (visit_id,)).fetchone()
            if visit is None:
                raise ValueError("방문 기록이 없습니다.")
            if event == "open" and visit["approval_status"] not in {"auto", "approved"}:
                raise ValueError("자동 출입 또는 승인된 방문만 열림 기록을 남길 수 있습니다.")
            if event == "pass" and not visit["door_opened"]:
                raise ValueError("열림 기록이 없는 방문에는 통과를 기록할 수 없습니다.")
        event_id = conn.execute(
            "INSERT INTO door_events(event_at, event, visit_id, event_key) VALUES (?, ?, ?, ?)",
            (stamp, event, visit_id, event_key),
        ).lastrowid
        if event == "open":
            conn.execute("UPDATE visits SET door_opened = 1 WHERE id = ?", (visit_id,))
        elif event == "pass":
            conn.execute("UPDATE visits SET passed_at = COALESCE(passed_at, ?) WHERE id = ?", (stamp, visit_id))
        return event_id


def update_passed(visit_id, *, event_at=None, event_key=None, db_path=DEFAULT_DB_PATH):
    """PASS 기록과 passed_at 갱신을 하나의 함수로 처리합니다."""
    return save_door_event("pass", visit_id, event_at=event_at, event_key=event_key, db_path=db_path)


def get_door_events(visit_id, *, db_path=DEFAULT_DB_PATH):
    with _connection(db_path) as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM door_events WHERE visit_id = ? ORDER BY event_at, id", (visit_id,)
        )]


def expire_pending(*, now=None, db_path=DEFAULT_DB_PATH):
    """F7용. 10분이 지난 pending만 expired로 변경합니다."""
    instant = _moment(now)
    with _connection(db_path) as conn:
        return conn.execute(
            """UPDATE visits SET approval_status = 'expired', decided_at = ?
               WHERE approval_status = 'pending' AND visited_at <= ?""",
            (_stamp(instant), _stamp(instant - timedelta(minutes=10))),
        ).rowcount


def decide_approval(visit_id, approved, *, now=None, db_path=DEFAULT_DB_PATH):
    """F7의 DB 상태만 변경. 기한 내 pending을 한 번만 결정할 수 있으면 True입니다."""
    if type(approved) is not bool:
        raise ValueError("approved에는 True 또는 False를 전달하세요.")
    instant = _moment(now)
    stamp, cutoff = _stamp(instant), _stamp(instant - timedelta(minutes=10))
    with _connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """UPDATE visits SET approval_status = 'expired', decided_at = ?
               WHERE id = ? AND approval_status = 'pending' AND visited_at <= ?""",
            (stamp, visit_id, cutoff),
        )
        return conn.execute(
            """UPDATE visits SET approval_status = ?, decided_at = ?
               WHERE id = ? AND approval_status = 'pending'
               AND visited_at > ? AND visited_at <= ?""",
            ("approved" if approved else "rejected", stamp, visit_id, cutoff, stamp),
        ).rowcount == 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description="방문자 DB 초기화 (기존 데이터 유지)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    print(f"DB 준비 완료: {init_db(args.db)}")


if __name__ == "__main__":
    main()
