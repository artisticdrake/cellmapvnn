"""
SQLite connection factory and schema creation.
Both DB1 (patient data) and DB2 (interpretations) live in the same file.
"""

import sqlite3
from pathlib import Path
from typing import Optional

_DB_PATH: Optional[Path] = None


def init_db(db_path: str | Path) -> None:
    global _DB_PATH
    _DB_PATH = Path(db_path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        _create_db1_schema(conn)
        _create_db2_schema(conn)
        conn.commit()


def connect() -> sqlite3.Connection:
    if _DB_PATH is None:
        raise RuntimeError("Call init_db() before connect()")
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA cache_size=-64000")  # 64 MB page cache
    return conn


def _create_db1_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS patients (
        sample_id                TEXT PRIMARY KEY,
        patient_id               TEXT,
        binary_label             INTEGER,
        dataset                  TEXT,
        logit                    REAL,
        predicted_prob           REAL,
        predicted_class          INTEGER,
        os_months                REAL,
        os_status                TEXT,
        cancer_type              TEXT,
        cancer_type_detailed     TEXT,
        sample_type              TEXT,
        mutation_count           INTEGER,
        tumor_purity             REAL,
        msi_type                 TEXT,
        fraction_genome_altered  REAL,
        gene_panel               TEXT,
        gender                   TEXT,
        race                     TEXT,
        ingested_at              TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS patient_gene_importance (
        sample_id   TEXT NOT NULL,
        gene        TEXT NOT NULL,
        importance  REAL NOT NULL,
        signed_z    REAL,
        PRIMARY KEY (sample_id, gene),
        FOREIGN KEY (sample_id) REFERENCES patients(sample_id)
    );

    CREATE TABLE IF NOT EXISTS patient_term_importance (
        sample_id   TEXT NOT NULL,
        term_id     TEXT NOT NULL,
        importance  REAL NOT NULL,
        PRIMARY KEY (sample_id, term_id),
        FOREIGN KEY (sample_id) REFERENCES patients(sample_id)
    );

    CREATE TABLE IF NOT EXISTS patient_rlipp (
        sample_id   TEXT NOT NULL,
        term_id     TEXT NOT NULL,
        rlipp_score REAL NOT NULL,
        PRIMARY KEY (sample_id, term_id),
        FOREIGN KEY (sample_id) REFERENCES patients(sample_id)
    );

    CREATE TABLE IF NOT EXISTS cohort_rlipp (
        term_id TEXT PRIMARY KEY,
        p_rho   REAL,
        p_pval  REAL,
        c_rho   REAL,
        c_pval  REAL,
        rlipp   REAL
    );

    CREATE TABLE IF NOT EXISTS cohort_gene_scores (
        gene    TEXT PRIMARY KEY,
        rho     REAL,
        p_val   REAL
    );

    CREATE TABLE IF NOT EXISTS cohort_boolean_logic (
        parent      TEXT NOT NULL,
        child1      TEXT NOT NULL,
        child2      TEXT NOT NULL,
        logic       TEXT,
        consistency REAL,
        n_samples   INTEGER,
        PRIMARY KEY (parent, child1, child2)
    );

    CREATE TABLE IF NOT EXISTS subsystem_gene_weights (
        term_id       TEXT NOT NULL,
        gene          TEXT NOT NULL,
        rlipp         REAL,
        pc1_corr      REAL,
        pc1_corr_pval REAL,
        rank          INTEGER,
        PRIMARY KEY (term_id, gene)
    );

    CREATE TABLE IF NOT EXISTS ontology (
        parent TEXT NOT NULL,
        child  TEXT NOT NULL,
        itype  TEXT,
        PRIMARY KEY (parent, child)
    );

    CREATE TABLE IF NOT EXISTS ingestion_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        source_file TEXT,
        rows_loaded INTEGER,
        started_at  TEXT,
        finished_at TEXT,
        status      TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_pgi_sample ON patient_gene_importance(sample_id);
    CREATE INDEX IF NOT EXISTS idx_pti_sample ON patient_term_importance(sample_id);
    CREATE INDEX IF NOT EXISTS idx_prl_sample ON patient_rlipp(sample_id);
    CREATE INDEX IF NOT EXISTS idx_pat_class  ON patients(predicted_class);
    CREATE INDEX IF NOT EXISTS idx_pat_prob   ON patients(predicted_prob);
    CREATE INDEX IF NOT EXISTS idx_pgi_gene   ON patient_gene_importance(gene);
    """)


def _create_db2_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS interpretations (
        sample_id            TEXT PRIMARY KEY,
        confidence_tier      TEXT,
        top_driver_system    TEXT,
        top_driver_gene      TEXT,
        interpretation_json  TEXT NOT NULL,
        ai_backend           TEXT,
        ai_model             TEXT,
        prompt_version       TEXT,
        batch_id             TEXT,
        tokens_used          INTEGER,
        hallucination_flag   INTEGER DEFAULT 0,
        interpreted_at       TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (sample_id) REFERENCES patients(sample_id)
    );

    CREATE TABLE IF NOT EXISTS interpretation_batches (
        batch_id      TEXT PRIMARY KEY,
        sample_ids    TEXT,
        ai_backend    TEXT,
        ai_model      TEXT,
        prompt_version TEXT,
        status        TEXT,
        error_message TEXT,
        started_at    TEXT,
        finished_at   TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_interp_tier   ON interpretations(confidence_tier);
    CREATE INDEX IF NOT EXISTS idx_interp_system ON interpretations(top_driver_system);
    CREATE INDEX IF NOT EXISTS idx_interp_gene   ON interpretations(top_driver_gene);
    CREATE INDEX IF NOT EXISTS idx_batch_status  ON interpretation_batches(status);
    """)
