"""임시 파일 DB로 실제 저장, 연결 재개, 제약조건과 동시 갱신을 검증합니다."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

import database as db


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "visitor.db"
        db.init_db(self.path)
        self.now = datetime(2026, 11, 7, 3, 0, tzinfo=timezone.utc)

    def person(self, name="김동진", **kwargs):
        return db.register_person(name, [[0.1, 0.2, 0.3]], model_key="test-v1", db_path=self.path, **kwargs)

    def visit(self, category="unregistered", **kwargs):
        kwargs.setdefault("visited_at", self.now)
        return db.save_visit(category, db_path=self.path, **kwargs)

    def read(self, sql, params=()):
        with closing(sqlite3.connect(self.path)) as conn, conn:
            return conn.execute(sql, params).fetchall()

    def test_init_is_idempotent_and_uses_wal(self):
        self.person()
        db.init_db(self.path)
        self.assertEqual(len(db.list_persons(db_path=self.path)), 1)
        self.assertEqual(self.read("PRAGMA journal_mode")[0][0], "wal")
        self.assertEqual(self.read("PRAGMA user_version")[0][0], 1)
        self.assertEqual(self.read("PRAGMA foreign_key_check"), [])

    def test_demo_and_future_schemas_are_not_overwritten(self):
        demo = Path(self.folder.name) / "demo.db"
        with closing(sqlite3.connect(demo)) as conn, conn:
            conn.execute("CREATE TABLE visits (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO visits VALUES (1)")
        with self.assertRaises(RuntimeError):
            db.init_db(demo)
        with closing(sqlite3.connect(demo)) as conn, conn:
            self.assertEqual(conn.execute("SELECT * FROM visits").fetchall(), [(1,)])
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("PRAGMA user_version = 2")
        with self.assertRaises(RuntimeError):
            db.init_db(self.path)

    def test_registration_round_trip_models_and_additional_vectors(self):
        person_id = self.person()
        db.add_embeddings(person_id, [[0.4, 0.5, 0.6]], model_key="test-v1", db_path=self.path)
        db.register_person("다른 모델", [[1, 2]], model_key="test-v2", db_path=self.path)
        rows = db.load_embeddings(model_key="test-v1", dimension=3, db_path=self.path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["person_name"], "김동진")
        self.assertAlmostEqual(rows[0]["embedding"][0], 0.1, places=6)
        self.assertEqual(db.load_embeddings(model_key="missing", db_path=self.path), [])
        self.assertEqual(db.list_persons(db_path=self.path)[0]["embedding_count"], 2)

    def test_invalid_vectors_do_not_create_partial_persons(self):
        for vectors in ([], [[]], [[float("nan")]], [[float("inf")]], [[1e100]], [[1], [1, 2]]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                db.register_person("invalid", vectors, model_key="test", db_path=self.path)
        self.assertEqual(db.list_persons(db_path=self.path), [])
        self.person()
        with self.assertRaises(ValueError):
            db.register_person("wrong dimension", [[1, 2]], model_key="test-v1", db_path=self.path)
        self.assertEqual(len(db.list_persons(db_path=self.path)), 1)
        with self.assertRaises(sqlite3.IntegrityError):
            db.add_embeddings(999, [[1, 2, 3]], model_key="test-v1", db_path=self.path)

    def test_visit_defaults_name_snapshot_and_missing_record(self):
        person_id = self.person()
        visit_id = self.visit("registered", person_id=person_id, face_distance=0.3)
        row = db.get_visit(visit_id, db_path=self.path)
        self.assertEqual(row["person_name"], "김동진")
        self.assertEqual(row["message"], "김동진님 환영합니다.")
        self.assertEqual(row["approval_status"], "auto")
        self.assertEqual(row["door_opened"], 0)
        other = db.get_visit(self.visit(), db_path=self.path)
        self.assertEqual(other["approval_status"], "counter")
        self.assertEqual(other["message"], "카운터에 가서 문의하세요.")
        self.assertIsNone(db.get_visit(999, db_path=self.path))

    def test_invalid_decisions_and_naive_times_are_rejected(self):
        for category, kwargs in [
            ("bad", {}), ("registered", {}), ("registered", {"person_id": 999}),
            ("unregistered", {"person_id": 1}), ("unregistered", {"approval_status": "approved"}),
            ("unregistered", {"face_distance": float("inf")}),
            ("unregistered", {"face_distance": -1}),
            ("unknown", {"embedding": [1, 2], "model_key": "test"}),
            ("unregistered", {"visited_at": datetime(2026, 11, 7)}),
        ]:
            with self.subTest(category=category, kwargs=kwargs), self.assertRaises(ValueError):
                self.visit(category, **kwargs)
        self.assertEqual(db.get_records(db_path=self.path), [])

    def test_failed_vector_save_rolls_back_the_visit(self):
        self.person()
        with self.assertRaises(ValueError):
            self.visit(embedding=[1, 2], model_key="test-v1")
        self.assertEqual(db.get_records(db_path=self.path), [])
        self.assertEqual(self.read("SELECT * FROM visit_embeddings"), [])

    def test_recent_vectors_expiry_and_cleanup(self):
        visit_id = self.visit(embedding=[1, 2, 3], model_key="test-v1")
        rows = db.recent_unknown_embeddings(model_key="test-v1", now=self.now, db_path=self.path)
        self.assertEqual(rows[0]["visit_id"], visit_id)
        self.assertEqual(rows[0]["embedding"], [1, 2, 3])
        self.assertEqual(db.recent_unknown_embeddings(
            model_key="test-v1", now=self.now - timedelta(seconds=1), db_path=self.path), [])
        later = self.now + timedelta(minutes=10)
        self.assertEqual(db.recent_unknown_embeddings(model_key="test-v1", now=later, db_path=self.path), [])
        self.assertEqual(db.cleanup_expired_embeddings(now=later, db_path=self.path), 1)
        self.assertIsNotNone(db.get_visit(visit_id, db_path=self.path))

    def test_kst_date_boundaries_and_stats(self):
        # KST 11/7 midnight = UTC 11/6 15:00. End is exclusive.
        start = datetime(2026, 11, 6, 15, tzinfo=timezone.utc)
        self.visit(visited_at=start - timedelta(microseconds=1))
        self.visit(visited_at=start)
        self.visit("unknown", visited_at=start + timedelta(hours=23))
        self.visit(visited_at=start + timedelta(days=1))
        records = db.get_records(day="2026-11-07", db_path=self.path)
        self.assertEqual(len(records), 2)
        self.assertEqual(db.get_today_stats(day="2026-11-07", db_path=self.path),
                         {"registered": 0, "unregistered": 1, "unknown": 1, "total": 2})

    def test_literal_name_search_filters_and_pagination(self):
        person_id = self.person("김_%'동진")
        for _ in range(3):
            self.visit("registered", person_id=person_id)
        self.visit()
        rows = db.get_records(name="_%'", category="registered", db_path=self.path)
        self.assertEqual(len(rows), 3)
        self.assertEqual(db.get_records(name="' OR 1=1 --", db_path=self.path), [])
        page = db.get_records(category="registered", limit=1, offset=1, db_path=self.path)
        self.assertEqual(page[0]["id"], rows[1]["id"])
        for kwargs in ({"limit": 0}, {"offset": -1}, {"limit": 1001}, {"category": "bad"}):
            with self.assertRaises(ValueError):
                db.get_records(db_path=self.path, **kwargs)

    def test_door_events_update_visit_and_are_idempotent(self):
        visit_id = self.visit("registered", person_id=self.person())
        event_id = db.save_door_event("open", visit_id, event_key="cycle-1-open", db_path=self.path)
        self.assertEqual(event_id, db.save_door_event(
            "open", visit_id, event_key="cycle-1-open", db_path=self.path))
        first_pass = self.now + timedelta(seconds=1)
        db.update_passed(visit_id, event_at=first_pass, event_key="cycle-1-pass", db_path=self.path)
        db.update_passed(visit_id, event_at=first_pass + timedelta(seconds=1), db_path=self.path)
        db.save_door_event("close", visit_id, db_path=self.path)
        row = db.get_visit(visit_id, db_path=self.path)
        self.assertEqual(row["door_opened"], 1)
        self.assertEqual(row["passed_at"], "2026-11-07T03:00:01.000000Z")
        self.assertEqual(len(db.get_door_events(visit_id, db_path=self.path)), 4)
        with self.assertRaises(ValueError):
            db.save_door_event("close", visit_id, event_key="cycle-1-open", db_path=self.path)

    def test_invalid_door_events_leave_no_partial_state(self):
        visit_id = self.visit()
        for event, related_id in (("open", visit_id), ("pass", visit_id), ("open", None), ("close", 999)):
            with self.assertRaises(ValueError):
                db.save_door_event(event, related_id, db_path=self.path)
        self.assertEqual(db.get_door_events(visit_id, db_path=self.path), [])
        self.assertEqual(db.get_visit(visit_id, db_path=self.path)["door_opened"], 0)
        # Closing at boot can have no related visit.
        db.save_door_event("close", db_path=self.path)

    def test_approval_once_and_no_automatic_door_operation(self):
        visit_id = self.visit(approval_status="pending")
        self.assertTrue(db.decide_approval(visit_id, True, now=self.now, db_path=self.path))
        self.assertFalse(db.decide_approval(visit_id, False, now=self.now, db_path=self.path))
        self.assertEqual(db.get_visit(visit_id, db_path=self.path)["approval_status"], "approved")
        self.assertEqual(db.get_visit(visit_id, db_path=self.path)["door_opened"], 0)
        db.save_door_event("open", visit_id, db_path=self.path)
        rejected = self.visit(approval_status="pending")
        self.assertTrue(db.decide_approval(rejected, False, now=self.now, db_path=self.path))
        with self.assertRaises(ValueError):
            db.save_door_event("open", rejected, db_path=self.path)

    def test_expired_approval_cannot_be_revived(self):
        first = self.visit(approval_status="pending")
        second = self.visit(approval_status="pending")
        counter = self.visit()
        later = self.now + timedelta(minutes=10)
        self.assertFalse(db.decide_approval(first, True, now=later, db_path=self.path))
        self.assertEqual(db.expire_pending(now=later, db_path=self.path), 1)
        self.assertEqual(db.get_visit(first, db_path=self.path)["approval_status"], "expired")
        self.assertEqual(db.get_visit(second, db_path=self.path)["approval_status"], "expired")
        self.assertEqual(db.get_visit(counter, db_path=self.path)["approval_status"], "counter")

    def test_two_counter_requests_cannot_both_decide(self):
        visit_id = self.visit(approval_status="pending")
        def decide(approved):
            return db.decide_approval(visit_id, approved, now=self.now, db_path=self.path)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(decide, [True, False]))
        self.assertEqual(sorted(results), [False, True])

    def test_delete_registration_keeps_visit_and_removes_vectors_and_name(self):
        person_id = self.person()
        visit_id = self.visit("registered", person_id=person_id)
        self.assertTrue(db.delete_person(person_id, db_path=self.path))
        self.assertFalse(db.delete_person(person_id, db_path=self.path))
        self.assertEqual(db.load_embeddings(model_key="test-v1", db_path=self.path), [])
        row = db.get_visit(visit_id, db_path=self.path)
        self.assertIsNone(row["person_id"])
        self.assertIsNone(row["person_name"])
        self.assertNotIn("김동진", row["message"])
        self.assertEqual(row["category"], "registered")

    def test_sql_constraints_are_enforced(self):
        person_id = self.person()
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("PRAGMA foreign_keys = ON")
            for values in ((person_id, 3, b"bad"), (999, 1, b"1234")):
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(
                        """INSERT INTO face_embeddings
                           (person_id, model_key, dimension, embedding, created_at)
                           VALUES (?, 'test', ?, ?, '2026-11-07T00:00:00.000000Z')""", values)
        self.assertEqual(self.read("PRAGMA integrity_check"), [("ok",)])


if __name__ == "__main__":
    unittest.main()
