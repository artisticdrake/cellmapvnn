"""
One-time (idempotent) ingestion of all NeST-VNN source files into DB1.
Run:  python ingest.py [--config config.yaml]
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import db


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_paths(config: dict) -> dict:
    root = Path(config["nest_vnn_root"])
    ann = root / config["paths"]["mlruns_annotation"]
    trn = root / config["paths"]["mlruns_training"]

    def p(section, key):
        return root / config["paths"][key]

    def ann_p(key):
        return ann / config["paths"][key]

    def trn_p(key):
        return trn / config["paths"][key]

    return {
        "patient_gene_importance": ann_p("patient_gene_importance"),
        "patient_term_importance": ann_p("patient_term_importance"),
        "patient_rlipp":           ann_p("patient_rlipp"),
        "patient_gene_signed_z":   ann_p("patient_gene_signed_z") if config["paths"].get("patient_gene_signed_z") else None,
        "rlipp_scores":            ann_p("rlipp_scores"),
        "gene_scores":             ann_p("gene_scores"),
        "boolean_logic":           ann_p("boolean_logic"),
        "subsystem_gene_weights":  ann_p("subsystem_gene_weights"),
        "ontology":                trn_p("ontology"),
        "training_data":           p(None, "training_data"),
        "clinical_outcomes":       p(None, "clinical_outcomes"),
        "predict_probabilities":   p(None, "predict_probabilities"),
        "predict_predictions":     p(None, "predict_predictions"),
        "predict_logits":          p(None, "predict_logits"),
    }


# ── Logging helper ────────────────────────────────────────────────────────────

def _log(conn: sqlite3.Connection, source: str, rows: int, started: str, status: str) -> None:
    conn.execute(
        "INSERT INTO ingestion_log(source_file, rows_loaded, started_at, finished_at, status) VALUES (?,?,?,?,?)",
        (source, rows, started, datetime.now().isoformat(), status),
    )


# ── Ingestion functions ───────────────────────────────────────────────────────

def ingest_patients(conn: sqlite3.Connection, paths: dict) -> int:
    started = datetime.now().isoformat()
    try:
        train = pd.read_csv(paths["training_data"], sep="\t")
        # rename to canonical column name
        train = train.rename(columns={"cell_line": "sample_id", "binary_os_months": "binary_label"})

        probs  = pd.read_csv(paths["predict_probabilities"], header=None, names=["predicted_prob"])
        preds  = pd.read_csv(paths["predict_predictions"],  header=None, names=["predicted_class"])
        logits = pd.read_csv(paths["predict_logits"],       header=None, names=["logit"])

        merged = pd.concat(
            [train.reset_index(drop=True), probs, preds, logits], axis=1
        )
        merged["patient_id"] = merged["sample_id"].str.split("-").str[:2].str.join("-")

        # join clinical data
        clin = pd.read_csv(paths["clinical_outcomes"])
        clin = clin.rename(columns={"sampleId": "sample_id"})
        clin_cols = [
            "sample_id", "OS_MONTHS", "OS_STATUS", "CANCER_TYPE", "CANCER_TYPE_DETAILED",
            "SAMPLE_TYPE", "MUTATION_COUNT", "TUMOR_PURITY", "MSI_TYPE",
            "FRACTION_GENOME_ALTERED", "GENE_PANEL", "GENDER", "RACE",
        ]
        clin_sub = clin[[c for c in clin_cols if c in clin.columns]].drop_duplicates("sample_id")

        merged = merged.merge(clin_sub, on="sample_id", how="left")

        col_map = {
            "OS_MONTHS": "os_months",
            "OS_STATUS": "os_status",
            "CANCER_TYPE": "cancer_type",
            "CANCER_TYPE_DETAILED": "cancer_type_detailed",
            "SAMPLE_TYPE": "sample_type",
            "MUTATION_COUNT": "mutation_count",
            "TUMOR_PURITY": "tumor_purity",
            "MSI_TYPE": "msi_type",
            "FRACTION_GENOME_ALTERED": "fraction_genome_altered",
            "GENE_PANEL": "gene_panel",
            "GENDER": "gender",
            "RACE": "race",
        }
        merged = merged.rename(columns=col_map)

        db_cols = [
            "sample_id", "patient_id", "binary_label", "dataset",
            "logit", "predicted_prob", "predicted_class",
            "os_months", "os_status", "cancer_type", "cancer_type_detailed",
            "sample_type", "mutation_count", "tumor_purity", "msi_type",
            "fraction_genome_altered", "gene_panel", "gender", "race",
        ]
        for col in db_cols:
            if col not in merged.columns:
                merged[col] = None

        rows = merged[db_cols].replace({np.nan: None})
        conn.executemany(
            f"INSERT OR REPLACE INTO patients({','.join(db_cols)}) VALUES ({','.join(['?']*len(db_cols))})",
            rows.itertuples(index=False, name=None),
        )
        conn.commit()
        n = len(rows)
        _log(conn, "patients", n, started, "ok")
        conn.commit()
        return n
    except Exception as e:
        _log(conn, "patients", 0, started, str(e))
        conn.commit()
        raise


def _ingest_wide_matrix(
    conn: sqlite3.Connection,
    path: Path,
    table: str,
    id_col: str,
    value_col: str,
    label_col: str,
    extra_path: Path | None = None,
    extra_col: str | None = None,
    chunk_size: int = 50000,
) -> int:
    """Generic wide-to-long ingestor for patient × feature matrices."""
    started = datetime.now().isoformat()
    try:
        df = pd.read_csv(path, sep="\t", index_col=id_col)

        # Optional second matrix (e.g. signed_z alongside importance)
        extra_df = None
        if extra_path and extra_col and Path(extra_path).exists():
            extra_df = pd.read_csv(extra_path, sep="\t", index_col=id_col)

        long = df.stack().reset_index()
        long.columns = [id_col, label_col, value_col]
        # Normalise index column name to sample_id for all tables
        long = long.rename(columns={id_col: "sample_id"})

        if extra_df is not None:
            long_extra = extra_df.stack().reset_index()
            long_extra.columns = [id_col, label_col, extra_col]
            long_extra = long_extra.rename(columns={id_col: "sample_id"})
            long = long.merge(long_extra, on=["sample_id", label_col], how="left")

        long = long.replace({np.nan: None})

        if extra_df is not None and extra_col:
            cols = ["sample_id", label_col, value_col, extra_col]
            sql = f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
        else:
            cols = ["sample_id", label_col, value_col]
            sql = f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES (?,?,?)"

        total = 0
        for start in range(0, len(long), chunk_size):
            chunk = long.iloc[start : start + chunk_size]
            conn.executemany(sql, chunk[cols].itertuples(index=False, name=None))
            conn.commit()
            total += len(chunk)

        _log(conn, str(path.name), total, started, "ok")
        conn.commit()
        return total
    except Exception as e:
        _log(conn, str(path.name), 0, started, str(e))
        conn.commit()
        raise


def ingest_gene_importance(conn: sqlite3.Connection, paths: dict) -> int:
    signed_z_path = paths.get("patient_gene_signed_z")
    if signed_z_path and not Path(signed_z_path).exists():
        signed_z_path = None
    return _ingest_wide_matrix(
        conn,
        path=paths["patient_gene_importance"],
        table="patient_gene_importance",
        id_col="cell_id",
        value_col="importance",
        label_col="gene",
        extra_path=signed_z_path,
        extra_col="signed_z",
    )


def ingest_term_importance(conn: sqlite3.Connection, paths: dict) -> int:
    return _ingest_wide_matrix(
        conn,
        path=paths["patient_term_importance"],
        table="patient_term_importance",
        id_col="cell_id",
        value_col="importance",
        label_col="term_id",
    )


def ingest_patient_rlipp(conn: sqlite3.Connection, paths: dict) -> int:
    return _ingest_wide_matrix(
        conn,
        path=paths["patient_rlipp"],
        table="patient_rlipp",
        id_col="cell_id",
        value_col="rlipp_score",
        label_col="term_id",
    )


def ingest_cohort_rlipp(conn: sqlite3.Connection, paths: dict) -> int:
    started = datetime.now().isoformat()
    try:
        df = pd.read_csv(paths["rlipp_scores"], sep="\t")
        df.columns = [c.strip() for c in df.columns]
        rows = df[["term", "p_rho", "p_pval", "c_rho", "c_pval", "rlipp"]].replace({np.nan: None})
        conn.executemany(
            "INSERT OR REPLACE INTO cohort_rlipp(term_id,p_rho,p_pval,c_rho,c_pval,rlipp) VALUES (?,?,?,?,?,?)",
            rows.itertuples(index=False, name=None),
        )
        conn.commit()
        n = len(rows)
        _log(conn, "rlipp_scores.txt", n, started, "ok")
        conn.commit()
        return n
    except Exception as e:
        _log(conn, "rlipp_scores.txt", 0, started, str(e))
        conn.commit()
        raise


def ingest_gene_scores(conn: sqlite3.Connection, paths: dict) -> int:
    started = datetime.now().isoformat()
    try:
        df = pd.read_csv(paths["gene_scores"], sep="\t")
        rows = df[["gene", "rho", "p_val"]].replace({np.nan: None})
        conn.executemany(
            "INSERT OR REPLACE INTO cohort_gene_scores(gene,rho,p_val) VALUES (?,?,?)",
            rows.itertuples(index=False, name=None),
        )
        conn.commit()
        n = len(rows)
        _log(conn, "gene_scores.txt", n, started, "ok")
        conn.commit()
        return n
    except Exception as e:
        _log(conn, "gene_scores.txt", 0, started, str(e))
        conn.commit()
        raise


def ingest_boolean_logic(conn: sqlite3.Connection, paths: dict) -> int:
    started = datetime.now().isoformat()
    try:
        df = pd.read_csv(paths["boolean_logic"], sep="\t")
        rows = df[["parent", "child1", "child2", "logic", "consistency", "n_samples"]].replace({np.nan: None})
        conn.executemany(
            "INSERT OR REPLACE INTO cohort_boolean_logic(parent,child1,child2,logic,consistency,n_samples) VALUES (?,?,?,?,?,?)",
            rows.itertuples(index=False, name=None),
        )
        conn.commit()
        n = len(rows)
        _log(conn, "boolean_logic.txt", n, started, "ok")
        conn.commit()
        return n
    except Exception as e:
        _log(conn, "boolean_logic.txt", 0, started, str(e))
        conn.commit()
        raise


def ingest_subsystem_gene_weights(conn: sqlite3.Connection, paths: dict) -> int:
    started = datetime.now().isoformat()
    try:
        df = pd.read_csv(paths["subsystem_gene_weights"], sep="\t")
        needed = ["term", "gene", "pc1_corr", "pc1_corr_pval", "rank"]
        # rlipp column may or may not be present
        if "rlipp" in df.columns:
            needed = ["term", "gene", "rlipp", "pc1_corr", "pc1_corr_pval", "rank"]
            rows = df[needed].rename(columns={"term": "term_id"}).replace({np.nan: None})
            conn.executemany(
                "INSERT OR REPLACE INTO subsystem_gene_weights(term_id,gene,rlipp,pc1_corr,pc1_corr_pval,rank) VALUES (?,?,?,?,?,?)",
                rows.itertuples(index=False, name=None),
            )
        else:
            rows = df[["term", "gene", "pc1_corr", "pc1_corr_pval", "rank"]].rename(columns={"term": "term_id"}).replace({np.nan: None})
            conn.executemany(
                "INSERT OR REPLACE INTO subsystem_gene_weights(term_id,gene,pc1_corr,pc1_corr_pval,rank) VALUES (?,?,?,?,?)",
                rows.itertuples(index=False, name=None),
            )
        conn.commit()
        n = len(rows)
        _log(conn, "subsystem_gene_weights.txt", n, started, "ok")
        conn.commit()
        return n
    except Exception as e:
        _log(conn, "subsystem_gene_weights.txt", 0, started, str(e))
        conn.commit()
        raise


def ingest_ontology(conn: sqlite3.Connection, paths: dict) -> int:
    started = datetime.now().isoformat()
    try:
        # ontology.txt format: index \t parent \t child \t type
        df = pd.read_csv(paths["ontology"], sep="\t", header=None)
        if df.shape[1] == 4:
            df.columns = ["idx", "parent", "child", "itype"]
        elif df.shape[1] == 3:
            df.columns = ["parent", "child", "itype"]
        rows = df[["parent", "child", "itype"]].replace({np.nan: None})
        conn.executemany(
            "INSERT OR IGNORE INTO ontology(parent,child,itype) VALUES (?,?,?)",
            rows.itertuples(index=False, name=None),
        )
        conn.commit()
        n = len(rows)
        _log(conn, "ontology.txt", n, started, "ok")
        conn.commit()
        return n
    except Exception as e:
        _log(conn, "ontology.txt", 0, started, str(e))
        conn.commit()
        raise


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_ingestion(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    paths = resolve_paths(config)
    db_path = Path(config_path).parent / config["db"]["path"]

    db.init_db(db_path)

    steps = [
        ("patients",                ingest_patients),
        ("gene importance",         ingest_gene_importance),
        ("term importance",         ingest_term_importance),
        ("patient RLIPP",           ingest_patient_rlipp),
        ("cohort RLIPP",            ingest_cohort_rlipp),
        ("cohort gene scores",      ingest_gene_scores),
        ("boolean logic",           ingest_boolean_logic),
        ("subsystem gene weights",  ingest_subsystem_gene_weights),
        ("ontology",                ingest_ontology),
    ]

    print(f"\n{'Source':<30} {'Rows':>10}  Status")
    print("-" * 50)
    total_ok = 0
    for name, fn in steps:
        try:
            with db.connect() as conn:
                n = fn(conn, paths)
            print(f"{name:<30} {n:>10}  OK")
            total_ok += 1
        except Exception as e:
            print(f"{name:<30} {'ERR':>10}  {e}")

    print("-" * 50)
    print(f"Completed {total_ok}/{len(steps)} steps.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest NeST-VNN outputs into pipeline DB")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_ingestion(args.config)
