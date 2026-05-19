"""
Prompt builder for the 5-patient interpretation batches.
Fully testable without any API calls.
"""

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import yaml


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_template(template_path: str | Path) -> dict:
    with open(template_path) as f:
        return yaml.safe_load(f)


def load_system_prompt(system_prompt_path: str | Path) -> str:
    return Path(system_prompt_path).read_text().strip()


def get_prompt_version(system_prompt: str, template: dict) -> str:
    raw = system_prompt + yaml.dump(template, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ── DB1 query helpers ─────────────────────────────────────────────────────────

def _fetch_patient_row(conn: sqlite3.Connection, sample_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM patients WHERE sample_id = ?", (sample_id,)
    ).fetchone()
    return dict(row) if row else None


def _fetch_top_genes(
    conn: sqlite3.Connection, sample_id: str, top_n: int
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT gi.gene, gi.importance, gi.signed_z, cg.rho AS cohort_rho
        FROM patient_gene_importance gi
        LEFT JOIN cohort_gene_scores cg ON gi.gene = cg.gene
        WHERE gi.sample_id = ?
        ORDER BY gi.importance DESC
        LIMIT ?
        """,
        (sample_id, top_n),
    ).fetchall()
    return [dict(r) for r in rows]


def _fetch_top_systems(
    conn: sqlite3.Connection, sample_id: str, top_n: int
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT pr.term_id, pr.rlipp_score AS patient_rlipp,
               cr.rlipp AS cohort_rlipp, cr.p_rho AS cohort_p_rho
        FROM patient_rlipp pr
        LEFT JOIN cohort_rlipp cr ON pr.term_id = cr.term_id
        WHERE pr.sample_id = ?
        ORDER BY pr.rlipp_score DESC
        LIMIT ?
        """,
        (sample_id, top_n),
    ).fetchall()
    return [dict(r) for r in rows]


def _fetch_top_genes_for_system(
    conn: sqlite3.Connection, sample_id: str, term_id: str, top_n: int = 2
) -> list[dict]:
    # Get genes annotated to this system, ranked by patient importance
    rows = conn.execute(
        """
        SELECT gi.gene, gi.importance
        FROM patient_gene_importance gi
        JOIN ontology o ON o.child = gi.gene AND o.parent = ? AND o.itype = 'gene'
        WHERE gi.sample_id = ?
        ORDER BY gi.importance DESC
        LIMIT ?
        """,
        (term_id, sample_id, top_n),
    ).fetchall()
    return [dict(r) for r in rows]


def _fetch_boolean_logic_for_systems(
    conn: sqlite3.Connection, term_ids: list[str]
) -> list[dict]:
    if not term_ids:
        return []
    placeholders = ",".join(["?"] * len(term_ids))
    rows = conn.execute(
        f"""
        SELECT parent, child1, child2, logic, consistency, n_samples
        FROM cohort_boolean_logic
        WHERE parent IN ({placeholders})
        ORDER BY consistency DESC
        LIMIT 3
        """,
        term_ids,
    ).fetchall()
    return [dict(r) for r in rows]


# ── Patient block builders ────────────────────────────────────────────────────

def build_patient_block(
    sample_id: str,
    conn: sqlite3.Connection,
    config: dict,
) -> dict:
    top_n_genes = config["ai"].get("top_n_genes", 10)
    top_n_systems = config["ai"].get("top_n_systems", 5)

    patient = _fetch_patient_row(conn, sample_id)
    if patient is None:
        raise ValueError(f"sample_id not found in DB: {sample_id}")

    genes = _fetch_top_genes(conn, sample_id, top_n_genes)
    systems = _fetch_top_systems(conn, sample_id, top_n_systems)

    # Enrich each system with its top genes (from ontology gene membership)
    for sys in systems:
        sys["top_genes_in_system"] = _fetch_top_genes_for_system(
            conn, sample_id, sys["term_id"]
        )

    top_system_ids = [s["term_id"] for s in systems]
    boolean_logic = _fetch_boolean_logic_for_systems(conn, top_system_ids)

    return {
        "sample_id": sample_id,
        "patient": patient,
        "genes": genes,
        "systems": systems,
        "boolean_logic": boolean_logic,
    }


def format_patient_block_as_text(data: dict) -> str:
    p = data["patient"]
    lines = [
        f"PATIENT: {data['sample_id']}",
        f"  Prediction: predicted_prob={_fmt(p.get('predicted_prob'))}, "
        f"predicted_class={p.get('predicted_class')}, "
        f"binary_label={p.get('binary_label')}",
        f"  Clinical: os_months={_fmt(p.get('os_months'))}, "
        f"os_status={p.get('os_status')}, "
        f"cancer_type={p.get('cancer_type_detailed') or p.get('cancer_type')}, "
        f"sample_type={p.get('sample_type')}, "
        f"mutation_count={p.get('mutation_count')}, "
        f"tumor_purity={_fmt(p.get('tumor_purity'))}",
        "  Top genes by importance:",
    ]
    for g in data["genes"]:
        lines.append(
            f"    {g['gene']}: importance={_fmt(g['importance'])}, "
            f"signed_z={_fmt(g.get('signed_z'))}, "
            f"cohort_rho={_fmt(g.get('cohort_rho'))}"
        )
    lines.append("  Top systems by patient RLIPP:")
    for s in data["systems"]:
        sys_genes = ", ".join(
            f"{g['gene']}(imp={_fmt(g['importance'])})"
            for g in s.get("top_genes_in_system", [])
        )
        lines.append(
            f"    {s['term_id']}: patient_rlipp={_fmt(s['patient_rlipp'])}, "
            f"cohort_rlipp={_fmt(s.get('cohort_rlipp'))}, "
            f"cohort_p_rho={_fmt(s.get('cohort_p_rho'))}"
            + (f", top_genes=[{sys_genes}]" if sys_genes else "")
        )
    if data["boolean_logic"]:
        lines.append("  Boolean logic (highest consistency gates):")
        for bl in data["boolean_logic"]:
            lines.append(
                f"    {bl['parent']} <- {bl['child1']} vs {bl['child2']}: "
                f"gate={bl['logic']}, consistency={_fmt(bl['consistency'])}, "
                f"n={bl['n_samples']}"
            )
    return "\n".join(lines)


def _fmt(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


# ── Batch message builder ─────────────────────────────────────────────────────

def build_batch_user_message(
    sample_ids: list[str],
    conn: sqlite3.Connection,
    config: dict,
    template: dict,
) -> str:
    blocks = []
    for sid in sample_ids:
        data = build_patient_block(sid, conn, config)
        blocks.append(format_patient_block_as_text(data))

    field_instructions = _build_field_instructions(template)

    os_threshold = config.get("study", {}).get("os_threshold_months", 46)
    parts = [
        f"Below are {len(sample_ids)} patients. OS threshold = {os_threshold} months.",
        "",
        "--- PATIENT DATA ---",
        "",
        "\n\n".join(blocks),
        "",
        "--- OUTPUT INSTRUCTIONS ---",
        field_instructions,
        "",
        f"Return a single JSON object with exactly {len(sample_ids)} top-level keys, "
        f"one per sample_id listed above. Each value must contain all required fields.",
    ]
    return "\n".join(parts)


def _build_field_instructions(template: dict) -> str:
    lines = ["For each patient, produce a JSON object with these fields:"]
    for field in template["fields"]:
        desc = field["description"].strip().replace("\n", " ")
        lines.append(f'  "{field["key"]}": {desc}')
    return "\n".join(lines)


def build_messages(
    sample_ids: list[str],
    conn: sqlite3.Connection,
    config: dict,
    system_prompt: str,
    template: dict,
) -> list[dict]:
    user_content = build_batch_user_message(sample_ids, conn, config, template)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
