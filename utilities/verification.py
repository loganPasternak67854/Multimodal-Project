"""CLI receipt validator for Bidirectional LGMM Audio-Text Retrieval (ECE 532 Fall 2026).

Usage:
    python -m src.verification experiments/bi_lgmm_bi_lgmm_audiocaps_cpu_20261115T134800Z.json
    python -m src.verification --strict experiments/bi_lgmm_bi_lgmm_audiocaps_cpu_20261115T134800Z.json

Exit codes:
    0  receipt is valid
    1  receipt is invalid (structural or semantic errors)
    2  file not found or unparseable JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

VALID_ASSIGNMENTS = {"A1", "A2", "A3", "A4", "project"}
VALID_MODELS = {"bi_lgmm", "asym_lgmm", "mmt"}
VALID_OUTCOMES = {"act", "defer", "request_evidence", "hand_off", "repair"}
VALID_RISK_TIERS = {"low", "medium", "high", "critical"}
VALID_CHECKER_KINDS = {"deterministic", "statistical", "protocol", "human_review"}
VALID_CLAIM_TYPES = {
    "geometric", "causal", "statistical", "protocol",
    "representation", "calibration", "fusion", "identification",
    "observational", "interventional", "counterfactual", "deferred",
}

REQUIRED_CHECKER_IDS = {
    "L2NormalizationGate",
    "ZeroVectorGate",
    "CosineBoundsGate",
    "AttentionDistributionGate",
    "FiniteValueGate",
    "SoftTargetScaleGate",
}

REQUIRED_RETRIEVAL_METRICS = {
    "A2T_R@1", "A2T_R@5", "A2T_R@10",
    "T2A_R@1", "T2A_R@5", "T2A_R@10",
}


def _errors_for_observation(obs: dict[str, Any], idx: int) -> list[str]:
    errors = []
    prefix = f"observations[{idx}]"
    for key in ("source", "modality", "timestamp", "payload_ref"):
        if key not in obs:
            errors.append(f"{prefix}: missing required field '{key}'")
        elif not isinstance(obs[key], str) or not obs[key].strip():
            errors.append(f"{prefix}.{key}: must be a non-empty string")
    return errors


def _errors_for_claim(claim: dict[str, Any], idx: int) -> list[str]:
    errors = []
    prefix = f"claims[{idx}]"
    for key in ("claim_id", "claim_type", "statement"):
        if key not in claim:
            errors.append(f"{prefix}: missing required field '{key}'")
        elif not isinstance(claim[key], str) or not claim[key].strip():
            errors.append(f"{prefix}.{key}: must be a non-empty string")
    if "claim_type" in claim and claim["claim_type"] not in VALID_CLAIM_TYPES:
        errors.append(
            f"{prefix}.claim_type: '{claim['claim_type']}' not in {sorted(VALID_CLAIM_TYPES)}"
        )
    if "risk_tier" in claim and claim["risk_tier"] not in VALID_RISK_TIERS:
        errors.append(
            f"{prefix}.risk_tier: '{claim['risk_tier']}' not in {sorted(VALID_RISK_TIERS)}"
        )
    if "confidence" in claim and claim["confidence"] is not None:
        c = claim["confidence"]
        if not isinstance(c, (int, float)) or isinstance(c, bool) or c < 0 or c > 1:
            errors.append(f"{prefix}.confidence: must be a number in [0, 1] or null")
    return errors


def _errors_for_obligation(obl: dict[str, Any], idx: int) -> list[str]:
    errors = []
    prefix = f"obligations[{idx}]"
    for key in ("obligation_id", "claim_id", "checker_id", "description"):
        if key not in obl:
            errors.append(f"{prefix}: missing required field '{key}'")
    return errors


def _errors_for_checker_result(cr: dict[str, Any], idx: int) -> list[str]:
    errors = []
    prefix = f"checker_results[{idx}]"
    for key in ("checker_id", "checker_kind", "passed", "message", "scope"):
        if key not in cr:
            errors.append(f"{prefix}: missing required field '{key}'")
    if "checker_kind" in cr and cr["checker_kind"] not in VALID_CHECKER_KINDS:
        errors.append(
            f"{prefix}.checker_kind: '{cr['checker_kind']}' not in {sorted(VALID_CHECKER_KINDS)}"
        )
    if "passed" in cr and not isinstance(cr["passed"], bool):
        errors.append(f"{prefix}.passed: must be a boolean")
    return errors


def _errors_for_action_gate(gate: dict[str, Any], idx: int) -> list[str]:
    errors = []
    prefix = f"action_gates[{idx}]"
    for key in ("claim_id", "risk_tier", "outcome", "reason", "obligations_met"):
        if key not in gate:
            errors.append(f"{prefix}: missing required field '{key}'")
    if "outcome" in gate and gate["outcome"] not in VALID_OUTCOMES:
        errors.append(
            f"{prefix}.outcome: '{gate['outcome']}' not in {sorted(VALID_OUTCOMES)}"
        )
    if "risk_tier" in gate and gate["risk_tier"] not in VALID_RISK_TIERS:
        errors.append(
            f"{prefix}.risk_tier: '{gate['risk_tier']}' not in {sorted(VALID_RISK_TIERS)}"
        )
    if "obligations_met" in gate and not isinstance(gate["obligations_met"], bool):
        errors.append(f"{prefix}.obligations_met: must be a boolean")
    return errors


def _errors_for_telemetry(receipt: dict[str, Any]) -> list[str]:
    """Validates Bidirectional LGMM metadata, wall-time, memory, and R@K metrics."""
    errors: list[str] = []

    meta = receipt.get("receipt_metadata")
    if not isinstance(meta, dict):
        errors.append("Missing or invalid 'receipt_metadata' object")
    else:
        for key in ("timestamp_utc", "experiment_name", "model_type", "device"):
            if not isinstance(meta.get(key), str) or not meta[key].strip():
                errors.append(f"receipt_metadata.{key}: must be a non-empty string")
        m_type = meta.get("model_type")
        if m_type and m_type not in VALID_MODELS:
            errors.append(f"receipt_metadata.model_type: '{m_type}' not in {sorted(VALID_MODELS)}")
        params = meta.get("num_trainable_parameters")
        if not isinstance(params, int) or isinstance(params, bool) or params <= 0:
            errors.append("receipt_metadata.num_trainable_parameters: must be a positive integer")

    wt = receipt.get("wall_time")
    if not isinstance(wt, dict):
        errors.append("Missing or invalid 'wall_time' object")
    else:
        for key in ("total_wall_time_sec", "training_loop_sec", "test_evaluation_sec"):
            val = wt.get(key)
            if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
                errors.append(f"wall_time.{key}: must be a non-negative number")

    mem = receipt.get("memory_consumption")
    if not isinstance(mem, dict):
        errors.append("Missing or invalid 'memory_consumption' object")
    else:
        for key in ("initial_rss_mb", "final_rss_mb", "peak_rss_mb", "python_heap_peak_mb"):
            val = mem.get(key)
            if not isinstance(val, (int, float)) or isinstance(val, bool) or val <= 0:
                errors.append(f"memory_consumption.{key}: must be a positive number")
        if (
            isinstance(mem.get("peak_rss_mb"), (int, float))
            and isinstance(mem.get("initial_rss_mb"), (int, float))
            and mem["peak_rss_mb"] < mem["initial_rss_mb"]
        ):
            errors.append("memory_consumption.peak_rss_mb cannot be less than initial_rss_mb")

    metrics = receipt.get("test_retrieval_metrics")
    if not isinstance(metrics, dict):
        errors.append("Missing or invalid 'test_retrieval_metrics' object")
    else:
        for m_key in sorted(REQUIRED_RETRIEVAL_METRICS):
            val = metrics.get(m_key)
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not (0.0 <= val <= 100.0):
                errors.append(f"test_retrieval_metrics.{m_key}: must be a float in [0.0, 100.0]")

    return errors


def _semantic_checks(receipt: dict[str, Any], strict: bool = False) -> list[str]:
    """Cross-field checks enforcing the fail-closed contract."""
    errors = []

    claim_ids = {c.get("claim_id") for c in receipt.get("claims", [])}
    checker_map = {r.get("checker_id"): r for r in receipt.get("checker_results", [])}

    for i, obl in enumerate(receipt.get("obligations", [])):
        cid = obl.get("claim_id")
        chk = obl.get("checker_id")
        required = obl.get("required", True)

        if cid and cid not in claim_ids:
            errors.append(f"obligations[{i}].claim_id: references unknown claim '{cid}'")
        if required and chk and chk not in checker_map:
            errors.append(
                f"obligations[{i}]: required checker '{chk}' has no corresponding "
                f"checker_result — receipt must fail closed"
            )

    for i, gate in enumerate(receipt.get("action_gates", [])):
        cid = gate.get("claim_id")
        outcome = gate.get("outcome")
        met = gate.get("obligations_met", False)
        if cid and cid not in claim_ids:
            errors.append(f"action_gates[{i}].claim_id: references unknown claim '{cid}'")
        if outcome == "act" and not met:
            errors.append(
                f"action_gates[{i}]: outcome is 'act' but obligations_met is false "
                f"— violates fail-closed contract"
            )

    # Check if any required checker failed or was unavailable while admitting 'act'
    for obl in receipt.get("obligations", []):
        if not obl.get("required", True):
            continue
        chk = obl.get("checker_id")
        cr = checker_map.get(chk)
        if cr and (cr.get("unavailable") or cr.get("passed") is False):
            admitted = any(g.get("outcome") == "act" for g in receipt.get("action_gates", []))
            if admitted:
                errors.append(
                    f"Required checker '{chk}' failed or is unavailable, but receipt admits "
                    f"an 'act' outcome — violates fail-closed contract"
                )

    if strict:
        obs_sources = {o.get("source") for o in receipt.get("observations", [])}
        for i, claim in enumerate(receipt.get("claims", [])):
            for eid in claim.get("evidence_ids", []):
                if eid not in obs_sources:
                    errors.append(
                        f"claims[{i}].evidence_ids: references unknown source '{eid}'"
                    )

    return errors


def _project_specific_checks(receipt: dict[str, Any], strict: bool = False) -> list[str]:
    """Project-specific invariants for Bidirectional LGMM (enforced in strict mode)."""
    if not strict:
        return []

    errors: list[str] = []
    assignment = receipt.get("assignment", "")
    if assignment != "project":
        return errors

    # 1. Both audio and text modalities must appear in observations
    modalities = {o.get("modality") for o in receipt.get("observations", [])}
    for req_mod in ("audio", "text"):
        if req_mod not in modalities:
            errors.append(f"project: observations must include '{req_mod}' modality")

    # 2. All LGMM integrity gates must be recorded when running LGMM models
    m_type = receipt.get("receipt_metadata", {}).get("model_type")
    if m_type in {"bi_lgmm", "asym_lgmm"}:
        recorded_checkers = {cr.get("checker_id") for cr in receipt.get("checker_results", [])}
        missing_gates = sorted(REQUIRED_CHECKER_IDS - recorded_checkers)
        if missing_gates:
            errors.append(
                f"project ({m_type}): missing required LGMM integrity checker(s): {missing_gates}"
            )

    # 3. Verify hyper-parameter beta in [0, 1] and CPU RAM budget (<= 16 GB)
    beta = receipt.get("hyperparameters", {}).get("loss", {}).get("beta")
    if beta is not None and (not isinstance(beta, (int, float)) or not (0.0 <= beta <= 1.0)):
        errors.append(f"project: hyperparameters.loss.beta={beta} violates beta in [0, 1]")

    peak_rss_gb = receipt.get("memory_consumption", {}).get("peak_rss_gb")
    if isinstance(peak_rss_gb, (int, float)) and peak_rss_gb > 16.0:
        errors.append(
            f"project: peak_rss_gb={peak_rss_gb} exceeds 16.0 GB CPU hardware boundary"
        )

    # 4. If failure injection was recorded, verify fail-closed deferral
    failures = receipt.get("failures_injected", [])
    if failures:
        if not any(cr.get("passed") is False for cr in receipt.get("checker_results", [])):
            errors.append(
                "project: failures_injected is non-empty, so at least one checker_result "
                "must have passed=false"
            )
        if not any(g.get("outcome") != "act" for g in receipt.get("action_gates", [])):
            errors.append(
                "project: failures_injected is non-empty, so at least one action_gate "
                "must defer/abort (outcome != 'act')"
            )

    return errors


def validate_receipt(receipt: dict[str, Any], strict: bool = False) -> list[str]:
    """Validate a project receipt dict. Returns a list of error messages (empty = valid)."""
    errors: list[str] = []

    if receipt.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version: expected '{SCHEMA_VERSION}', got '{receipt.get('schema_version')}'"
        )

    for key in ("run_id", "assignment"):
        if key not in receipt:
            errors.append(f"Missing required top-level field '{key}'")
        elif not isinstance(receipt[key], str) or not receipt[key].strip():
            errors.append(f"Top-level field '{key}' must be a non-empty string")

    assignment = receipt.get("assignment", "")
    if assignment and assignment not in VALID_ASSIGNMENTS:
        errors.append(f"assignment: '{assignment}' not in {sorted(VALID_ASSIGNMENTS)}")

    list_fields = [
        "observations", "claims", "obligations",
        "checker_results", "action_gates",
    ]
    for lf in list_fields:
        val = receipt.get(lf)
        if not isinstance(val, list):
            errors.append(f"'{lf}' must be a list")

    for i, obs in enumerate(receipt.get("observations", [])):
        if isinstance(obs, dict):
            errors.extend(_errors_for_observation(obs, i))
        else:
            errors.append(f"observations[{i}]: must be an object")

    for i, claim in enumerate(receipt.get("claims", [])):
        if isinstance(claim, dict):
            errors.extend(_errors_for_claim(claim, i))
        else:
            errors.append(f"claims[{i}]: must be an object")

    for i, obl in enumerate(receipt.get("obligations", [])):
        if isinstance(obl, dict):
            errors.extend(_errors_for_obligation(obl, i))
        else:
            errors.append(f"obligations[{i}]: must be an object")

    for i, cr in enumerate(receipt.get("checker_results", [])):
        if isinstance(cr, dict):
            errors.extend(_errors_for_checker_result(cr, i))
        else:
            errors.append(f"checker_results[{i}]: must be an object")

    for i, gate in enumerate(receipt.get("action_gates", [])):
        if isinstance(gate, dict):
            errors.extend(_errors_for_action_gate(gate, i))
        else:
            errors.append(f"action_gates[{i}]: must be an object")

    if "outcome" not in receipt or receipt["outcome"] not in VALID_OUTCOMES:
        errors.append(f"outcome: '{receipt.get('outcome')}' not in {sorted(VALID_OUTCOMES)}")

    errors.extend(_errors_for_telemetry(receipt))
    errors.extend(_semantic_checks(receipt, strict=strict))
    errors.extend(_project_specific_checks(receipt, strict=strict))

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.verification",
        description="Validate a Bidirectional LGMM execution receipt.",
    )
    parser.add_argument("receipt", type=Path, help="Path to receipt JSON file")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode: check evidence_id refs, all 6 LGMM gates, and 16GB RAM limit",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress success message",
    )
    args = parser.parse_args(argv)

    if not args.receipt.exists():
        print(f"Error: file not found: {args.receipt}", file=sys.stderr)
        return 2

    try:
        with open(args.receipt, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON: {exc}", file=sys.stderr)
        return 2

    if not isinstance(data, dict):
        print("Error: receipt must be a JSON object", file=sys.stderr)
        return 1

    errors = validate_receipt(data, strict=args.strict)

    if errors:
        print(f"INVALID — {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  • {err}", file=sys.stderr)
        return 1

    if not args.quiet:
        assignment = data.get("assignment", "?")
        run_id = data.get("run_id", "?")
        model_type = data.get("receipt_metadata", {}).get("model_type", "?")
        outcome = data.get("outcome", "not set")
        n_claims = len(data.get("claims", []))
        n_checkers = len(data.get("checker_results", []))
        n_gates = len(data.get("action_gates", []))
        print(
            f"VALID — {assignment} [{model_type}] run {run_id}: "
            f"{n_claims} claim(s), {n_checkers} checker(s), "
            f"{n_gates} gate(s), outcome={outcome}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())