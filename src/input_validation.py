"""Waste Recognition - Lab 05 input validator.

Checks the values supplied in a request against config/input_contract.json
and decides what happens before prediction:
    pass    -> call_prediction
    warning -> route_for_review  (prediction is NOT called automatically)
    error   -> reject_input      (prediction is NOT called)
When several rules fail, the most severe result wins: error > warning > pass.

Request-level only: this module does not open, find or decode the photo.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FILE = ROOT / "config" / "input_contract.json"

SEVERITY = {"pass": 0, "warning": 1, "error": 2}
ACTIONS = {"pass": "call_prediction", "warning": "route_for_review", "error": "reject_input"}


def load_contract(path: Path = CONTRACT_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rule(contract: dict, name: str) -> dict:
    for rule in contract["rules"]:
        if rule["rule"] == name:
            return rule
    raise KeyError(f"rule {name!r} missing from the contract")


def _check(rule: dict, observed, ok: bool) -> dict:
    status = "pass" if ok else rule["on_fail"]
    return {"rule": rule["rule"], "observed": observed, "status": status, "action": ACTIONS[status]}


def check_input_reference(request: dict, contract: dict) -> dict:
    rule = _rule(contract, "input_reference_present")
    value = request.get("input_ref")
    return _check(rule, value, isinstance(value, str) and value.strip() != "")


def check_format(request: dict, contract: dict) -> dict:
    rule = _rule(contract, "format_supported")
    value = request.get("format")
    return _check(rule, value, isinstance(value, str) and value.lower() in rule["allowed"])


def check_dimensions(request: dict, contract: dict) -> dict:
    rule = _rule(contract, "dimensions_match_tray_camera")
    w, h = request.get("width"), request.get("height")
    ok = w == rule["expected_width"] and h == rule["expected_height"]
    return _check(rule, f"{w}x{h}", ok)


def check_source(request: dict, contract: dict) -> dict:
    rule = _rule(contract, "source_is_tray_camera")
    value = request.get("source")
    return _check(rule, value, value in rule["approved"])


def check_preprocessing(runtime_preprocessing_id: str, contract: dict) -> dict:
    rule = _rule(contract, "preprocessing_identity_match")
    return _check(rule, runtime_preprocessing_id, runtime_preprocessing_id == contract["preprocessing_id"])


def combine(checks: list[dict]) -> tuple[str, str]:
    """Most severe status wins: error > warning > pass."""
    overall = max((c["status"] for c in checks), key=SEVERITY.__getitem__, default="pass")
    return overall, ACTIONS[overall]


def validate_request(request: dict, runtime_preprocessing_id: str, contract: dict | None = None) -> dict:
    contract = contract or load_contract()
    checks = [
        check_input_reference(request, contract),
        check_format(request, contract),
        check_dimensions(request, contract),
        check_source(request, contract),
        check_preprocessing(runtime_preprocessing_id, contract),
    ]
    status, action = combine(checks)
    return {
        "contract_version": contract["contract_version"],
        "checks": checks,
        "status": status,
        "action": action,
    }
