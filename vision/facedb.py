"""A가 C의 DB 함수를 부르는 곳 (A 담당).

C의 database.py(sqlite_1 브랜치) 함수 형태에 맞춘다.
    register_person(name, embeddings, *, model_key, photo_path=None, db_path=...)
    add_embeddings(person_id, embeddings, *, model_key, db_path=...)
    list_persons(*, db_path=...)
    load_embeddings(*, model_key, dimension=None, db_path=...)  -> [{"person_id", "person_name", "embedding": [float...]}, ...]
    delete_person(person_id, *, db_path=...)

- database.py는 db/database.py 와 루트 database.py 를 둘 다 찾는다.
- DB 경로는 항상 config.DB_PATH 를 넘긴다 (database.py 위치와 상관없이 같은 DB를 쓰도록).
- database.py가 없으면 임시 파일 experiments/known_faces.npz 를 쓴다.
"""
import importlib
import os

import numpy as np

import config

NPZ_PATH = os.path.join("experiments", "known_faces.npz")

_module = None
_module_name = None
_searched = False


def _find_module():
    """C의 database 모듈을 찾는다. 없으면 None."""
    global _module, _module_name, _searched
    if _searched:
        return _module
    _searched = True
    for name in ("db.database", "database"):
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name not in (name, name.split(".")[0]):   # 모듈 안에서 다른 import가 실패한 경우는 숨기지 않음
                raise
            continue
        if hasattr(mod, "load_embeddings") and hasattr(mod, "register_person"):
            _module, _module_name = mod, name
            return mod
    return None


def backend() -> str:
    """"DB (db.database)" 같은 설명 문자열. DB가 없으면 "npz"."""
    mod = _find_module()
    return f"DB ({_module_name})" if mod else "npz"


def model_key() -> str:
    """등록·인식에 쓰는 모델 이름. CLAHE를 켜면 다른 이름이 되어 예전 벡터와 섞이지 않는다."""
    return config.FACE_MODEL_KEY + ("_clahe" if config.USE_CLAHE else "")


def _db():
    mod = _find_module()
    mod.init_db(config.DB_PATH)          # 없으면 만들고, 있으면 그대로 둔다
    return mod


# ───────────────────────── 공개 함수 ─────────────────────────

def load_known() -> list[tuple[int, str, np.ndarray]]:
    """등록 벡터 전체 [(person_id, name, float32 벡터), ...]"""
    if _find_module() is None:
        return _Npz().load_known()
    rows = _db().load_embeddings(model_key=model_key(), db_path=config.DB_PATH)
    return [(int(r["person_id"]), str(r["person_name"]), np.asarray(r["embedding"], dtype=np.float32))
            for r in rows]


def list_persons() -> list[dict]:
    """[{"id", "name", "photo_path", "created_at", "embedding_count"}, ...]"""
    if _find_module() is None:
        return _Npz().list_persons()
    return _db().list_persons(db_path=config.DB_PATH)


def find_ids(name: str) -> list[int]:
    """이름이 같은 등록자 번호 목록 (C의 DB는 이름 중복을 막지 않으므로 여러 개일 수 있음)."""
    return [int(p["id"]) for p in list_persons() if p["name"] == name]


def register(name: str, photo_paths: list[str], embeddings: list[np.ndarray]) -> int:
    """새 사람이면 등록, 이미 있으면 그 사람에게 벡터 추가. person_id를 돌려준다."""
    vectors = [np.asarray(e, dtype=np.float32).ravel() for e in embeddings]
    if _find_module() is None:
        return _Npz().register(name, photo_paths, vectors)
    mod = _db()
    ids = find_ids(name)
    if ids:
        mod.add_embeddings(ids[0], vectors, model_key=model_key(), db_path=config.DB_PATH)
        return ids[0]
    return int(mod.register_person(name, vectors, model_key=model_key(),
                                   photo_path=photo_paths[0] if photo_paths else None,
                                   db_path=config.DB_PATH))


def delete(name: str) -> int:
    """이름이 같은 등록자를 모두 삭제하고 지운 수를 돌려준다."""
    if _find_module() is None:
        return 1 if _Npz().delete(name) else 0
    mod = _db()
    return sum(1 for pid in find_ids(name) if mod.delete_person(pid, db_path=config.DB_PATH))


# ───────────────────────── 임시 저장소 (DB가 오기 전) ─────────────────────────

class _Npz:
    """experiments/known_faces.npz. 형식: ids, names, embeddings(N×128), photos"""

    def __init__(self, path: str = NPZ_PATH):
        self.path = path

    def _load(self):
        if not os.path.exists(self.path):
            return [], [], [], []
        d = np.load(self.path, allow_pickle=False)
        photos = [str(p) for p in d["photos"]] if "photos" in d else [""] * len(d["ids"])
        return ([int(i) for i in d["ids"]], [str(n) for n in d["names"]],
                [e.astype(np.float32) for e in d["embeddings"]], photos)

    def _save(self, ids, names, embs, photos):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        np.savez(self.path,
                 ids=np.array(ids, dtype=np.int64),
                 names=np.array(names, dtype=str),
                 embeddings=np.array(embs, dtype=np.float32).reshape(-1, 128),
                 photos=np.array(photos, dtype=str))

    def load_known(self):
        ids, names, embs, _ = self._load()
        return list(zip(ids, names, embs))

    def list_persons(self):
        ids, names, _, photos = self._load()
        out = {}
        for i, n, p in zip(ids, names, photos):
            if n not in out:
                out[n] = {"id": i, "name": n, "photo_path": p, "created_at": "-", "embedding_count": 0}
            out[n]["embedding_count"] += 1
        return list(out.values())

    def register(self, name, photo_paths, embeddings):
        ids, names, embs, photos = self._load()
        pid = ids[names.index(name)] if name in names else (max(ids) + 1 if ids else 1)
        for k, e in enumerate(embeddings):
            ids.append(pid)
            names.append(name)
            embs.append(e)
            photos.append(photo_paths[k] if k < len(photo_paths) else "")
        self._save(ids, names, embs, photos)
        return pid

    def delete(self, name):
        ids, names, embs, photos = self._load()
        keep = [k for k, n in enumerate(names) if n != name]
        if len(keep) == len(names):
            return False
        self._save([ids[k] for k in keep], [names[k] for k in keep],
                   [embs[k] for k in keep], [photos[k] for k in keep])
        return True
