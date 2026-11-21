-- Schema version 1. All timestamps are UTC ISO 8601 strings produced by database.py.
BEGIN IMMEDIATE;

CREATE TABLE persons (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    photo_path TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE face_embeddings (
    id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    model_key TEXT NOT NULL CHECK (length(trim(model_key)) > 0),
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    embedding BLOB NOT NULL CHECK (length(embedding) = dimension * 4),
    created_at TEXT NOT NULL
);
CREATE INDEX idx_faces_model ON face_embeddings(model_key, dimension, person_id);
CREATE INDEX idx_faces_person ON face_embeddings(person_id);

CREATE TABLE visits (
    id INTEGER PRIMARY KEY,
    visited_at TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('registered', 'unregistered', 'unknown')),
    person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    person_name TEXT,
    face_distance REAL CHECK (face_distance IS NULL OR face_distance >= 0),
    message TEXT NOT NULL,
    image_path TEXT,
    approval_status TEXT NOT NULL CHECK (
        approval_status IN ('auto', 'counter', 'pending', 'approved', 'rejected', 'expired')
    ),
    decided_at TEXT,
    door_opened INTEGER NOT NULL DEFAULT 0 CHECK (door_opened IN (0, 1)),
    passed_at TEXT,
    CHECK (
        (category = 'registered' AND approval_status = 'auto') OR
        (category <> 'registered' AND approval_status <> 'auto' AND person_id IS NULL)
    ),
    CHECK (passed_at IS NULL OR door_opened = 1),
    CHECK (
        (approval_status IN ('approved', 'rejected', 'expired') AND decided_at IS NOT NULL) OR
        (approval_status IN ('auto', 'counter', 'pending') AND decided_at IS NULL)
    )
);
CREATE INDEX idx_visits_time ON visits(visited_at DESC, id DESC);
CREATE INDEX idx_visits_category_time ON visits(category, visited_at DESC);
CREATE INDEX idx_visits_pending ON visits(approval_status, visited_at DESC);
CREATE INDEX idx_visits_person ON visits(person_id);

-- Short-lived vectors for duplicate detection. Actual similarity comparison belongs to A/B.
CREATE TABLE visit_embeddings (
    visit_id INTEGER PRIMARY KEY REFERENCES visits(id) ON DELETE CASCADE,
    model_key TEXT NOT NULL CHECK (length(trim(model_key)) > 0),
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    embedding BLOB NOT NULL CHECK (length(embedding) = dimension * 4),
    expires_at TEXT NOT NULL
);
CREATE INDEX idx_visit_embeddings_expiry ON visit_embeddings(expires_at);
CREATE INDEX idx_visit_embeddings_model ON visit_embeddings(model_key, dimension);

CREATE TABLE door_events (
    id INTEGER PRIMARY KEY,
    event_at TEXT NOT NULL,
    event TEXT NOT NULL CHECK (event IN ('open', 'close', 'pass')),
    visit_id INTEGER REFERENCES visits(id) ON DELETE CASCADE,
    event_key TEXT UNIQUE,
    CHECK (event = 'close' OR visit_id IS NOT NULL)
);
CREATE INDEX idx_door_events_visit ON door_events(visit_id, event_at, id);

PRAGMA user_version = 1;
COMMIT;
