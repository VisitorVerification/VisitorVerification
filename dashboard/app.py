import argparse
import functools
import logging
import os
import random
import re
from datetime import datetime, timedelta

from flask import Flask, abort, render_template, request, send_from_directory, url_for

from config import CAPTURES_DIR, DASHBOARD_HOST, DASHBOARD_PORT, DATA_DIR, DB_PATH

log = logging.getLogger("dashboard")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURES_ABS = os.path.join(PROJECT_ROOT, CAPTURES_DIR)   # 이 폴더만 주소로 공개 (5.7절)
DB_ABS = os.path.join(PROJECT_ROOT, DB_PATH)              # main.py를 프로젝트 폴더에서 켰을 때와 같은 DB
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

REFRESH_SEC = 5        # / 자동 새로고침 간격
RECENT_LIMIT = 10      # / 최근 기록 수
RECORDS_LIMIT = 200    # /records 최대 건수

CATEGORY_LABELS = {"registered": "등록자", "unregistered": "미등록자", "unknown": "인식 불가"}
APPROVAL_LABELS = {"auto": "자동 통과", "counter": "카운터 안내", "pending": "승인 대기",
                   "approved": "승인", "rejected": "거절", "expired": "미처리"}

app = Flask(__name__)

# 조회 함수 (3.5절 C의 함수 형태). run()에서 실제 함수나 가짜 함수로 채운다
data = {"get_records": None, "get_today_stats": None}


# ── 가짜 조회 함수 (--fake에서만): 예시 기록으로 화면을 확인할 때 쓴다. 진짜 DB는 읽지 않음 ──

def _make_fake_records() -> list[dict]:
    """visits 테이블(3.7절) 컬럼 이름을 키로 쓰는 예시 기록. 오늘·어제·그제."""
    rng = random.Random(0)
    names = ["김성민", "김동진", "정재열"]
    records = []
    now = datetime.now().replace(microsecond=0)
    midnight = now.replace(hour=0, minute=0, second=0)
    for i in range(1, 31):
        days_ago = (30 - i) // 12            # 오늘 12건, 어제 12건, 그제 6건
        if days_ago == 0:                    # 오늘: 자정부터 지금 사이
            visited = midnight + timedelta(seconds=rng.randint(0, max(int((now - midnight).total_seconds()), 1)))
        else:                                # 지난 날: 9시~18시 사이
            visited = midnight - timedelta(days=days_ago) + timedelta(seconds=rng.randint(9 * 3600, 18 * 3600))
        category = rng.choice(["registered", "registered", "unregistered", "unknown"])
        registered = category == "registered"
        name = rng.choice(names) if registered else None
        opened = 1 if registered else 0
        records.append({
            "id": i,
            "visited_at": visited.strftime("%Y-%m-%d %H:%M:%S"),
            "category": category,
            "person_id": names.index(name) + 1 if name else None,
            "person_name": name,
            "face_distance": round(rng.uniform(0.2, 0.45) if registered else rng.uniform(0.55, 0.9), 3)
                             if category != "unknown" else None,
            "message": f"{name}님 환영합니다." if registered else "카운터에 가서 문의하세요.",
            "image_path": f"captures/fake/{i}.jpg",   # 예시 사진은 이 폴더에만 (실제 사진과 안 섞임)
            "approval_status": "auto" if registered else "counter",
            "decided_at": None,
            "door_opened": opened,
            "passed_at": (visited + timedelta(seconds=4)).strftime("%Y-%m-%d %H:%M:%S") if opened else None,
        })
    return records


_FAKE_RECORDS: list[dict] = []


def fake_get_records(category: str | None = None, date: str | None = None,
                     name: str | None = None, limit: int = 200) -> list[dict]:
    rows = [r for r in _FAKE_RECORDS
            if (not category or r["category"] == category)
            and (not date or r["visited_at"].startswith(date))
            and (not name or (r["person_name"] and name in r["person_name"]))]
    rows.sort(key=lambda r: r["visited_at"], reverse=True)
    return [dict(r) for r in rows[:limit]]


def fake_get_today_stats() -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    rows = [r for r in _FAKE_RECORDS if r["visited_at"].startswith(today)]
    stats = {"total": len(rows), "registered": 0, "unregistered": 0, "unknown": 0}
    for r in rows:
        stats[r["category"]] += 1
    return stats


def make_fake_captures() -> None:
    """가짜 기록의 사진 파일을 data/captures/에 만든다 (.gitignore라 GitHub에 안 올라감)."""
    import cv2
    import numpy as np
    from common.imageio import imwrite_unicode

    colors = {"registered": (67, 160, 46), "unregistered": (47, 47, 211), "unknown": (97, 97, 97)}
    for r in _FAKE_RECORDS:
        path = os.path.join(PROJECT_ROOT, DATA_DIR, r["image_path"])
        if os.path.exists(path):
            continue
        img = np.full((240, 320, 3), colors[r["category"]], dtype=np.uint8)
        cv2.putText(img, f"VISIT {r['id']}", (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        imwrite_unicode(path, img)


# ── 화면 ──

def image_url(image_path: str | None) -> str | None:
    """DB의 image_path(captures/...)를 /captures/ 주소로 바꾼다. captures/ 밖이면 None."""
    if not image_path or not image_path.startswith("captures/"):
        return None
    return url_for("captures", filename=image_path[len("captures/"):])


@app.context_processor
def helpers():
    return {"category_label": lambda c: CATEGORY_LABELS.get(c, c),
            "approval_label": lambda a: APPROVAL_LABELS.get(a, a or "-"),
            "image_url": image_url}


@app.route("/")
def index():
    error = None
    stats = {"total": 0, "registered": 0, "unregistered": 0, "unknown": 0}
    recent = []
    try:
        stats = data["get_today_stats"]()
        recent = data["get_records"](limit=RECENT_LIMIT)
    except Exception:
        log.exception("오늘 현황을 불러오지 못했습니다")
        error = "기록을 불러오지 못했습니다."
    return render_template("index.html", stats=stats, recent=recent, error=error,
                           refresh=REFRESH_SEC, today=datetime.now().strftime("%Y-%m-%d"))


@app.route("/records")
def records():
    category = request.args.get("category") or None
    if category not in CATEGORY_LABELS:
        category = None
    date = request.args.get("date") or None
    if date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        date = None
    name = (request.args.get("name") or "").strip() or None

    error = None
    rows = []
    try:
        rows = data["get_records"](category=category, date=date, name=name, limit=RECORDS_LIMIT)
    except Exception:
        log.exception("기록을 불러오지 못했습니다")
        error = "기록을 불러오지 못했습니다."
    return render_template("records.html", rows=rows, error=error, labels=CATEGORY_LABELS,
                           category=category or "", date=date or "", name=name or "", limit=RECORDS_LIMIT)


@app.route("/captures/<path:filename>")
def captures(filename: str):
    """방문 사진만 보여준다. captures/ 밖(등록 얼굴, DB)은 절대 열리지 않는다."""
    if not filename.lower().endswith(IMAGE_EXTS):
        abort(404)
    return send_from_directory(CAPTURES_ABS, filename)   # ../ 같은 경로 탈출은 404


def open_db(db_path: str = DB_ABS) -> tuple:
    """C의 DB를 준비하고 (get_records, get_today_stats)를 돌려준다. 실패하면 예외를 그대로 올린다."""
    from db import database as db
    db.init_db(db_path)
    return (functools.partial(db.get_records, db_path=db_path),
            functools.partial(db.get_today_stats, db_path=db_path))


def run(fake: bool) -> None:
    if fake:
        _FAKE_RECORDS[:] = _make_fake_records()
        make_fake_captures()
        data["get_records"], data["get_today_stats"] = fake_get_records, fake_get_today_stats
        print("※ --fake: 예시 기록을 보여 줍니다 (진짜 DB는 읽지 않음).")
    else:
        try:
            data["get_records"], data["get_today_stats"] = open_db()
        except Exception:
            log.exception("DB를 열지 못했습니다 (data/ 폴더 확인)")
            return
        print(f"기록 읽는 위치: {DB_ABS}")
    print(f"대시보드: http://<노트북 IPv4>:{DASHBOARD_PORT}  (노트북에서는 http://127.0.0.1:{DASHBOARD_PORT})")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="방문 기록 대시보드")
    parser.add_argument("--fake", action="store_true", help="진짜 DB 대신 예시 기록으로 화면 확인")
    run(fake=parser.parse_args().fake)


if __name__ == "__main__":
    main()
