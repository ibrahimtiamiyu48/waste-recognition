"""Waste Recognition — Lab 04 inference demo.

One photo of one waste item goes in; a structured response comes out and is
saved to reports/inference_demo.jsonl. The run makes two attempts on the same
photo:
  1. a normal attempt that uses the Lab 02 baseline model;
  2. a controlled attempt where inference is marked unavailable (simulated).

Run from the repository root (after `python src/baseline_pipeline.py`):
    python src/inference_demo.py --input-ref data/raw/dataset-resized/glass/glass12.jpg
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import joblib
from PIL import Image, UnidentifiedImageError

from baseline_pipeline import MODEL_FILE, ROOT, features  # same preprocessing as Lab 02

EVIDENCE_FILE = ROOT / "reports" / "inference_demo.jsonl"

# Versions and actions (see docs/project_decisions.md, Lab 03)
MODEL_VERSION = "baseline-v1"   # logistic regression from src/baseline_pipeline.py
RULE_VERSION = "decision-v1"    # review rule below
NORMAL_ACTION = "show material on the tray screen; staff put the item in that container"
REVIEW_ACTION = "show 'Check by hand' with the top two materials; staff decide the container"
UNAVAILABLE_ACTION = "show 'Sorting assistant offline'; staff sort the item by hand"

# decision-v1: send to a person when the model is unsure, or when glass is a
# close runner-up (missing glass is our most important failure, Lab 03).
MIN_CONFIDENCE = 0.60
GLASS_RUNNER_UP = 0.25

_model_bundle = None


def load_model() -> dict:
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_FILE.exists():
            sys.exit(
                f"Model not found at {MODEL_FILE.relative_to(ROOT)}.\n"
                "Run `python src/baseline_pipeline.py` first; it trains and saves the baseline."
            )
        _model_bundle = joblib.load(MODEL_FILE)
    return _model_bundle


def check_input(input_ref: str) -> None:
    """Basic input check already used by the Lab 02 baseline: readable photo, 512x384."""
    path = ROOT / input_ref
    if not path.is_file():
        sys.exit(f"Input rejected: file not found ({input_ref}). Retake or choose another photo.")
    try:
        with Image.open(path) as img:
            size = img.size
    except UnidentifiedImageError:
        sys.exit(f"Input rejected: not a readable image ({input_ref}). Retake the photo.")
    if size != (512, 384):
        sys.exit(f"Input rejected: size {size}, expected (512, 384). Use the tray camera format.")


def project_predict(path: Path) -> dict:
    """Prediction step from the Lab 02 baseline + decision-v1 rule."""
    bundle = load_model()
    model, classes = bundle["model"], bundle["classes"]
    proba = model.predict_proba(features(path).reshape(1, -1))[0]
    ranked = sorted(zip(classes, proba), key=lambda kv: kv[1], reverse=True)
    (top, p_top), (second, p_second) = ranked[0], ranked[1]

    needs_review = p_top < MIN_CONFIDENCE or (second == "glass" and p_second >= GLASS_RUNNER_UP)
    return {
        "result": {
            "material": top,
            "confidence": round(float(p_top), 3),
            "runner_up": {"material": second, "confidence": round(float(p_second), 3)},
        },
        "needs_review": bool(needs_review),
    }


def run_attempt(input_ref: str, simulate_unavailable: bool) -> dict:
    """Inference boundary: accepted input in, structured response out."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    try:
        if simulate_unavailable:
            raise RuntimeError("inference marked unavailable for the classroom test")
        out = project_predict(ROOT / input_ref)
        status = "review" if out["needs_review"] else "ok"
        result = out["result"]
        action = REVIEW_ACTION if out["needs_review"] else NORMAL_ACTION
    except RuntimeError:
        status, result, action = "unavailable", None, UNAVAILABLE_ACTION
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    return {
        "request_id": request_id,
        "input_ref": input_ref,
        "status": status,
        "result": result,
        "model_version": MODEL_VERSION,
        "rule_version": RULE_VERSION,
        "latency_ms": latency_ms,
        "action": action,
        "simulated": simulate_unavailable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one waste photo and record the attempt.")
    parser.add_argument("--input-ref", required=True,
                        help="photo path relative to the repo root, e.g. data/raw/dataset-resized/glass/glass12.jpg")
    args = parser.parse_args()

    check_input(args.input_ref)  # rejected input never reaches the model
    load_model()  # load once, so the normal attempt's latency is the prediction itself

    records = [
        run_attempt(args.input_ref, simulate_unavailable=False),
        run_attempt(args.input_ref, simulate_unavailable=True),
    ]

    EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with EVIDENCE_FILE.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    for rec in records:
        print(json.dumps(rec, indent=2))
    print(f"Saved {len(records)} records to {EVIDENCE_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
