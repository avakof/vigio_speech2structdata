-- speechflow persistence schema, migration 1.
-- History tables are append-only: triggers abort every UPDATE or DELETE.
-- runs.current_save_id is the one mutable pointer (see runs_pointer_only).

CREATE TABLE sources (
  source_id      TEXT PRIMARY KEY,
  kind           TEXT NOT NULL CHECK (kind IN ('file', 'mic')),
  digest         TEXT NOT NULL,
  name           TEXT NOT NULL,
  native_rate    INTEGER NOT NULL CHECK (native_rate > 0),
  native_channels INTEGER NOT NULL CHECK (native_channels > 0),
  sample_rate    INTEGER NOT NULL CHECK (sample_rate = 16000),
  sample_count   INTEGER NOT NULL CHECK (sample_count >= 0),
  channel_mix    TEXT NOT NULL,
  first_seen_at  TEXT NOT NULL
) STRICT;

CREATE TABLE runs (
  run_id             TEXT PRIMARY KEY,
  source_id          TEXT NOT NULL REFERENCES sources (source_id),
  mode               TEXT NOT NULL CHECK (mode IN ('transcribe', 'run', 'mic')),
  processing_version TEXT NOT NULL,
  schema_version     TEXT NOT NULL,
  created_at         TEXT NOT NULL,
  current_save_id    INTEGER REFERENCES run_saves (save_id),
  UNIQUE (source_id, mode, processing_version)
) STRICT;

-- One row per distinct stored result of a run. A retry of the same content adds
-- no row (see save_events); only a run whose current save has status 'error'
-- can gain a later generation.
CREATE TABLE run_saves (
  save_id               INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id                TEXT NOT NULL REFERENCES runs (run_id),
  generation            INTEGER NOT NULL CHECK (generation >= 1),
  content_digest        TEXT NOT NULL,
  manifest_digest       TEXT NOT NULL,
  status                TEXT NOT NULL CHECK (status IN ('ok', 'no_speech', 'needs_review', 'error')),
  transcription_status  TEXT NOT NULL,
  extraction_status     TEXT NOT NULL,
  manifest_json         TEXT NOT NULL CHECK (json_valid(manifest_json)),
  saved_at              TEXT NOT NULL,
  UNIQUE (run_id, generation),
  UNIQUE (run_id, content_digest)
) STRICT;

CREATE TABLE save_events (
  event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id          TEXT NOT NULL REFERENCES runs (run_id),
  save_id         INTEGER NOT NULL REFERENCES run_saves (save_id),
  disposition     TEXT NOT NULL CHECK (disposition IN ('inserted', 'duplicate', 'replaced')),
  manifest_digest TEXT NOT NULL,
  event_at        TEXT NOT NULL
) STRICT;

CREATE TABLE segments (
  save_id         INTEGER NOT NULL REFERENCES run_saves (save_id),
  segment_id      TEXT NOT NULL,
  source_id       TEXT NOT NULL REFERENCES sources (source_id),
  ordinal         INTEGER NOT NULL CHECK (ordinal >= 0),
  input_start     INTEGER NOT NULL,
  core_start      INTEGER NOT NULL,
  core_end        INTEGER NOT NULL,
  input_end       INTEGER NOT NULL,
  raw_json        TEXT NOT NULL CHECK (json_valid(raw_json)),
  normalized_text TEXT NOT NULL,
  status          TEXT NOT NULL CHECK (status IN ('ok', 'no_speech', 'needs_review', 'error')),
  errors_json     TEXT NOT NULL CHECK (json_valid(errors_json)),
  PRIMARY KEY (save_id, segment_id),
  UNIQUE (save_id, ordinal),
  CHECK (input_start <= core_start AND core_start < core_end AND core_end <= input_end)
) STRICT;

CREATE TABLE observations (
  save_id        INTEGER NOT NULL REFERENCES run_saves (save_id),
  observation_id TEXT NOT NULL,
  ordinal        INTEGER NOT NULL CHECK (ordinal >= 0),
  revision       INTEGER NOT NULL CHECK (revision >= 1),
  status         TEXT NOT NULL CHECK (status IN ('ok', 'needs_review')),
  operation      TEXT NOT NULL CHECK (operation IN ('count', 'correction', 'negation', 'unknown')),
  product_spoken TEXT,
  catalog_match  TEXT,
  quantity_text  TEXT,
  quantity       TEXT,
  quantity_max   TEXT,
  qualifier      TEXT NOT NULL CHECK (qualifier IN ('exact', 'approximate', 'range', 'unknown')),
  unit_spoken    TEXT,
  unit           TEXT,
  packaging_json TEXT CHECK (packaging_json IS NULL OR json_valid(packaging_json)),
  location       TEXT,
  correction_of  TEXT,
  errors_json    TEXT NOT NULL CHECK (json_valid(errors_json)),
  PRIMARY KEY (save_id, observation_id),
  UNIQUE (save_id, ordinal)
) STRICT;

CREATE TABLE observation_evidence (
  save_id        INTEGER NOT NULL,
  observation_id TEXT NOT NULL,
  ordinal        INTEGER NOT NULL CHECK (ordinal >= 0),
  segment_id     TEXT NOT NULL,
  span_start     INTEGER NOT NULL CHECK (span_start >= 0),
  span_end       INTEGER NOT NULL CHECK (span_end > span_start),
  quote          TEXT NOT NULL CHECK (length(quote) > 0),
  PRIMARY KEY (save_id, observation_id, ordinal),
  FOREIGN KEY (save_id, observation_id) REFERENCES observations (save_id, observation_id),
  FOREIGN KEY (save_id, segment_id) REFERENCES segments (save_id, segment_id)
) STRICT;

CREATE INDEX idx_runs_source ON runs (source_id);
CREATE INDEX idx_saves_run ON run_saves (run_id);
CREATE INDEX idx_observations_correction ON observations (save_id, correction_of);

-- Immutability -----------------------------------------------------------

CREATE TRIGGER sources_immutable_update BEFORE UPDATE ON sources
BEGIN SELECT RAISE(ABORT, 'sources are immutable'); END;
CREATE TRIGGER sources_immutable_delete BEFORE DELETE ON sources
BEGIN SELECT RAISE(ABORT, 'sources are immutable'); END;

CREATE TRIGGER run_saves_immutable_update BEFORE UPDATE ON run_saves
BEGIN SELECT RAISE(ABORT, 'run_saves are immutable'); END;
CREATE TRIGGER run_saves_immutable_delete BEFORE DELETE ON run_saves
BEGIN SELECT RAISE(ABORT, 'run_saves are immutable'); END;

CREATE TRIGGER save_events_immutable_update BEFORE UPDATE ON save_events
BEGIN SELECT RAISE(ABORT, 'save_events are immutable'); END;
CREATE TRIGGER save_events_immutable_delete BEFORE DELETE ON save_events
BEGIN SELECT RAISE(ABORT, 'save_events are immutable'); END;

CREATE TRIGGER segments_immutable_update BEFORE UPDATE ON segments
BEGIN SELECT RAISE(ABORT, 'segments are immutable'); END;
CREATE TRIGGER segments_immutable_delete BEFORE DELETE ON segments
BEGIN SELECT RAISE(ABORT, 'segments are immutable'); END;

CREATE TRIGGER observations_immutable_update BEFORE UPDATE ON observations
BEGIN SELECT RAISE(ABORT, 'observations and corrections are immutable; append a revision instead'); END;
CREATE TRIGGER observations_immutable_delete BEFORE DELETE ON observations
BEGIN SELECT RAISE(ABORT, 'observations and corrections are immutable; append a revision instead'); END;

CREATE TRIGGER evidence_immutable_update BEFORE UPDATE ON observation_evidence
BEGIN SELECT RAISE(ABORT, 'observation evidence is immutable'); END;
CREATE TRIGGER evidence_immutable_delete BEFORE DELETE ON observation_evidence
BEGIN SELECT RAISE(ABORT, 'observation evidence is immutable'); END;

CREATE TRIGGER runs_delete_forbidden BEFORE DELETE ON runs
BEGIN SELECT RAISE(ABORT, 'runs are immutable'); END;
CREATE TRIGGER runs_pointer_only BEFORE UPDATE ON runs
WHEN OLD.run_id IS NOT NEW.run_id OR OLD.source_id IS NOT NEW.source_id
  OR OLD.mode IS NOT NEW.mode OR OLD.processing_version IS NOT NEW.processing_version
  OR OLD.schema_version IS NOT NEW.schema_version OR OLD.created_at IS NOT NEW.created_at
  OR NEW.current_save_id IS NULL
  OR NOT EXISTS (SELECT 1 FROM run_saves
                 WHERE save_id = NEW.current_save_id AND run_id = NEW.run_id)
BEGIN SELECT RAISE(ABORT, 'only runs.current_save_id may change, to a save of the same run'); END;

-- Views derived from the normalized tables of each run's current save ------

CREATE VIEW current_runs AS
SELECT r.run_id, r.source_id, r.mode, r.processing_version, r.schema_version,
       s.save_id, s.generation, s.status, s.transcription_status, s.extraction_status,
       s.content_digest, s.saved_at
FROM runs r JOIN run_saves s ON s.save_id = r.current_save_id;

CREATE VIEW current_segments AS
SELECT c.run_id, g.*
FROM current_runs c JOIN segments g ON g.save_id = c.save_id;

-- is_final: no 'ok' revision of the same save names this observation as its
-- correction target. corrected_by lists every revision that names it.
CREATE VIEW current_observations AS
SELECT c.run_id, o.*,
  (SELECT group_concat(x.observation_id, ';')
     FROM (SELECT observation_id FROM observations
            WHERE save_id = o.save_id AND correction_of = o.observation_id
            ORDER BY ordinal) x) AS corrected_by,
  NOT EXISTS (SELECT 1 FROM observations k
               WHERE k.save_id = o.save_id AND k.correction_of = o.observation_id
                 AND k.status = 'ok') AS is_final
FROM current_runs c JOIN observations o ON o.save_id = c.save_id;

-- Every revision of every observation across all saves of all runs.
CREATE VIEW observation_history AS
SELECT s.run_id, s.generation, s.saved_at, o.*
FROM observations o JOIN run_saves s ON s.save_id = o.save_id;
