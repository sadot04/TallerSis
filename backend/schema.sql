PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    row_count INTEGER NOT NULL CHECK (row_count > 0),
    classes_json TEXT NOT NULL CHECK (json_valid(classes_json)),
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    dataset_id TEXT REFERENCES datasets(id),
    metrics_json TEXT NOT NULL CHECK (json_valid(metrics_json)),
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL CHECK (active IN (0, 1))
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS one_active_model ON models(active) WHERE active = 1;

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(id),
    operator TEXT NOT NULL,
    input_json TEXT NOT NULL CHECK (json_valid(input_json)),
    result_json TEXT NOT NULL CHECK (json_valid(result_json)),
    created_at TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS prediction_history ON predictions(operator, id DESC);

CREATE TABLE IF NOT EXISTS training_jobs (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id),
    model_id TEXT REFERENCES models(id),
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'rejected', 'failed')),
    metrics_json TEXT CHECK (metrics_json IS NULL OR json_valid(metrics_json)),
    message TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS one_running_training ON training_jobs(status) WHERE status = 'running';
