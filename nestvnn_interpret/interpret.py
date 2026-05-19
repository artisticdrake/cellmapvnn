"""
Batch AI interpretation pipeline: DB1 → Claude → DB2.
Run:  python interpret.py [--config config.yaml] [--limit N] [--dry-run]
"""

import argparse
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterator

import yaml

import db
import prompt as prompt_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Config + backend loader ───────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_backend(config: dict):
    backend = config["ai"]["backend"].lower()
    if backend == "anthropic":
        from ai_backends.anthropic_backend import AnthropicBackend
        return AnthropicBackend(config)
    elif backend == "openai":
        from ai_backends.openai_backend import OpenAIBackend
        return OpenAIBackend(config)
    elif backend == "ollama":
        from ai_backends.ollama_backend import OllamaBackend
        return OllamaBackend(config)
    else:
        raise ValueError(f"Unknown AI backend: {backend!r}")


# ── Queue helpers ─────────────────────────────────────────────────────────────

def get_pending_sample_ids(conn, limit: int | None = None) -> list[str]:
    sql = """
        SELECT p.sample_id FROM patients p
        LEFT JOIN interpretations i ON p.sample_id = i.sample_id
        WHERE i.sample_id IS NULL
        ORDER BY p.sample_id
    """
    params = []
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [r["sample_id"] for r in rows]


def batch_iter(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ── Response parsing + validation ────────────────────────────────────────────

def parse_batch_response(
    response_text: str,
    sample_ids: list[str],
    template: dict,
    patient_probs: dict[str, float],
) -> dict[str, dict]:
    """
    Parse and validate the AI JSON response.
    Returns dict: sample_id -> interpretation dict.
    Raises ValueError on structural failures.
    """
    # Strip markdown code fences if the model wrapped in ```json ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", response_text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failure: {e}\nRaw response:\n{response_text[:500]}")

    required_keys = {f["key"] for f in template["fields"]}
    allowed_tiers = {"high", "moderate", "low"}

    result = {}
    missing_ids = []

    for sid in sample_ids:
        if sid not in parsed:
            missing_ids.append(sid)
            continue
        interp = parsed[sid]

        # Check all required keys present
        missing_fields = required_keys - set(interp.keys())
        if missing_fields:
            raise ValueError(f"Patient {sid} missing fields: {missing_fields}")

        # Validate confidence_tier
        tier = interp.get("confidence_tier")
        if tier not in allowed_tiers:
            raise ValueError(f"Patient {sid}: invalid confidence_tier={tier!r}")

        result[sid] = interp

    if missing_ids:
        raise ValueError(f"AI response missing patients: {missing_ids}")

    return result, _check_hallucination(result, patient_probs)


def _check_hallucination(
    parsed: dict[str, dict], patient_probs: dict[str, float]
) -> bool:
    """
    Spot-check: does the AI cite predicted_prob within 0.01 of the DB1 value?
    Returns True if a hallucination is suspected.
    """
    for sid, interp in parsed.items():
        summary = interp.get("survival_prediction_summary") or ""
        if not isinstance(summary, str):
            continue
        # Extract any float-looking number from the summary
        nums = re.findall(r"\b0\.\d+\b", summary)
        db_prob = patient_probs.get(sid)
        if db_prob is None or not nums:
            continue
        cited = min(nums, key=lambda x: abs(float(x) - db_prob))
        if abs(float(cited) - db_prob) > 0.015:
            log.warning(
                f"Hallucination check FAILED for {sid}: "
                f"DB prob={db_prob:.4f}, AI cited={cited}"
            )
            return True
    return False


def extract_structured_fields(interp: dict) -> dict:
    """Extract materialised columns from a single-patient interpretation dict."""
    confidence_tier = interp.get("confidence_tier")

    top_driver_system = None
    top_systems = interp.get("top_driver_systems")
    if isinstance(top_systems, list) and top_systems:
        first = top_systems[0]
        if isinstance(first, dict):
            top_driver_system = first.get("system_id") or first.get("term_id") or str(first)
        elif isinstance(first, str):
            top_driver_system = first
    elif isinstance(top_systems, str):
        top_driver_system = top_systems.split()[0] if top_systems else None

    top_driver_gene = None
    key_genes = interp.get("key_genes")
    if isinstance(key_genes, list) and key_genes:
        first = key_genes[0]
        if isinstance(first, dict):
            top_driver_gene = first.get("gene") or first.get("gene_symbol") or str(first)
        elif isinstance(first, str):
            top_driver_gene = first.split()[0]
    elif isinstance(key_genes, str):
        top_driver_gene = key_genes.split()[0] if key_genes else None

    return {
        "confidence_tier": confidence_tier,
        "top_driver_system": top_driver_system,
        "top_driver_gene": top_driver_gene,
    }


# ── DB2 writers ───────────────────────────────────────────────────────────────

def _insert_batch_record(conn, batch_id: str, sample_ids: list[str], config: dict, prompt_version: str) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO interpretation_batches
           (batch_id, sample_ids, ai_backend, ai_model, prompt_version, status, started_at)
           VALUES (?,?,?,?,?,?,?)""",
        (
            batch_id,
            json.dumps(sample_ids),
            config["ai"]["backend"],
            config["ai"]["model"],
            prompt_version,
            "running",
            datetime.now().isoformat(),
        ),
    )
    conn.commit()


def _update_batch_status(conn, batch_id: str, status: str, error: str | None = None) -> None:
    conn.execute(
        "UPDATE interpretation_batches SET status=?, error_message=?, finished_at=? WHERE batch_id=?",
        (status, error, datetime.now().isoformat(), batch_id),
    )
    conn.commit()


def write_batch_to_db2(
    conn,
    batch_id: str,
    parsed: dict[str, dict],
    config: dict,
    prompt_version: str,
    tokens_used: int,
    hallucination_flag: bool,
) -> None:
    for sid, interp in parsed.items():
        fields = extract_structured_fields(interp)
        conn.execute(
            """INSERT OR REPLACE INTO interpretations
               (sample_id, confidence_tier, top_driver_system, top_driver_gene,
                interpretation_json, ai_backend, ai_model, prompt_version,
                batch_id, tokens_used, hallucination_flag)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sid,
                fields["confidence_tier"],
                fields["top_driver_system"],
                fields["top_driver_gene"],
                json.dumps(interp),
                config["ai"]["backend"],
                config["ai"]["model"],
                prompt_version,
                batch_id,
                tokens_used,
                int(hallucination_flag),
            ),
        )
    conn.commit()


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_interpretation(
    config_path: str = "config.yaml",
    limit: int | None = None,
    dry_run: bool = False,
) -> None:
    config = load_config(config_path)
    config_dir = Path(config_path).parent
    db_path = config_dir / config["db"]["path"]
    db.init_db(db_path)

    system_prompt = prompt_mod.load_system_prompt(config_dir / config["prompt"]["system_prompt_file"])
    template = prompt_mod.load_template(config_dir / config["prompt"]["template_file"])
    prompt_version = prompt_mod.get_prompt_version(system_prompt, template)

    backend = None if dry_run else load_backend(config)
    batch_size = config["ai"]["batch_size"]

    with db.connect() as conn:
        pending = get_pending_sample_ids(conn, limit)

    if not pending:
        print("No pending patients to interpret.")
        return

    print(f"Interpreting {len(pending)} patients in batches of {batch_size}.")
    if dry_run:
        print("DRY RUN — no API calls will be made.\n")

    n_ok = n_fail = 0

    for batch in batch_iter(pending, batch_size):
        batch_id = str(uuid.uuid4())

        with db.connect() as conn:
            _insert_batch_record(conn, batch_id, batch, config, prompt_version)

            # Fetch patient probs for hallucination check
            placeholders = ",".join(["?"] * len(batch))
            rows = conn.execute(
                f"SELECT sample_id, predicted_prob FROM patients WHERE sample_id IN ({placeholders})",
                batch,
            ).fetchall()
            patient_probs = {r["sample_id"]: r["predicted_prob"] for r in rows}

            messages = prompt_mod.build_messages(batch, conn, config, system_prompt, template)

        if dry_run:
            print(f"\n{'='*60}")
            print(f"Batch: {batch}")
            print(f"System prompt length: {len(messages[0]['content'])} chars")
            print(f"User message length:  {len(messages[1]['content'])} chars")
            print("--- USER MESSAGE PREVIEW (first 800 chars) ---")
            print(messages[1]["content"][:800])
            n_ok += 1
            continue

        try:
            response_text, tokens_used = backend.complete(messages, config)
            parsed, hallucination = parse_batch_response(
                response_text, batch, template, patient_probs
            )
            with db.connect() as conn:
                write_batch_to_db2(
                    conn, batch_id, parsed, config, prompt_version, tokens_used, hallucination
                )
                _update_batch_status(conn, batch_id, "done")

            flag = " ⚠ hallucination_flag" if hallucination else ""
            log.info(f"Batch {batch_id[:8]} OK — {len(batch)} patients, {tokens_used} tokens{flag}")
            n_ok += len(batch)

        except Exception as e:
            with db.connect() as conn:
                _update_batch_status(conn, batch_id, "failed", str(e))
            log.error(f"Batch {batch_id[:8]} FAILED: {e}")
            n_fail += len(batch)

    print(f"\nDone. Interpreted: {n_ok} | Failed: {n_fail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI interpretation pipeline")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--limit",    type=int, default=None, help="Max patients to process")
    parser.add_argument("--dry-run",  action="store_true",    help="Build prompts without calling API")
    args = parser.parse_args()
    run_interpretation(args.config, args.limit, args.dry_run)
