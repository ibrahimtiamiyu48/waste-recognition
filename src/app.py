"""Waste Recognition — Lab 06 API service.

The Lab 04-05 inference path behind HTTP:
    POST /predict  photo -> decode -> input checks (Lab 05 contract) -> baseline-v1 -> JSONL record
    GET  /health   which service, model, rule, contract and preprocessing are running

Start from the repository root (after `python src/baseline_pipeline.py`):
    python3 -m uvicorn app:app --app-dir src --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import json
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

import inference_demo as lab04  # Lab 04 model + decision-v1, Lab 05 PREPROCESSING_ID
from baseline_pipeline import MODEL_FILE, ROOT
from input_validation import validate_request  # Lab 05 contract and checks

SERVICE_NAME = "sortvision-waste-api"
SERVICE_VERSION = "service-v1"
LOG_FILE = Path(os.getenv("SERVICE_LOG", ROOT / "reports" / "service_predictions.jsonl"))
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # a 512x384 tray photo is ~10-60 KB

app = FastAPI(
    title="SortVision waste recognition",
    description="Classifies one photo of one waste item as plastic, paper, glass or metal. "
                "Input is checked before the model runs; every request is written to a JSONL log.",
    version=SERVICE_VERSION,
)

STARTED_AT = datetime.now(timezone.utc).isoformat()


def model_loaded() -> bool:
    return lab04._model_bundle is not None


def predict_image(data: bytes) -> dict:
    """Existing Lab 04 prediction: baseline-v1 + decision-v1 on the decoded photo."""
    lab04.load_model()
    return lab04.project_predict(io.BytesIO(data))  # features() opens it with PIL


PREDICT = predict_image  # tests replace this with a spy


def decode(data: bytes) -> dict:
    """Decoded properties of the upload. A file that is not an image gives None values."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            return {"decoded": True, "format": (img.format or "").lower() or None,
                    "width": img.width, "height": img.height, "mode": img.mode}
    except (UnidentifiedImageError, OSError, ValueError):
        return {"decoded": False, "format": None, "width": None, "height": None, "mode": None}


def write_record(record: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "service_version": SERVICE_VERSION,
        "host": socket.gethostname(),
        "started_at": STARTED_AT,
        "model_version": lab04.MODEL_VERSION,
        "model_file": str(MODEL_FILE.relative_to(ROOT)),
        "model_file_present": MODEL_FILE.exists(),
        "model_loaded": model_loaded(),
        "rule_version": lab04.RULE_VERSION,
        "preprocessing_id": lab04.PREPROCESSING_ID,
        "contract_version": json.loads((ROOT / "config" / "input_contract.json").read_text())["contract_version"],
        "log_file": str(LOG_FILE.relative_to(ROOT)) if LOG_FILE.is_relative_to(ROOT) else str(LOG_FILE),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Photo of one waste item (tray camera: JPEG, 512x384)"),
    request_id: Optional[str] = Form(None, description="Optional ID, e.g. req-cloud-001"),
    source: str = Form("tray-camera", description="tray-camera, trashnet-test, or another source"),
):
    request_id = request_id or str(uuid.uuid4())
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    image = decode(data) if len(data) <= MAX_UPLOAD_BYTES else \
        {"decoded": False, "format": None, "width": None, "height": None, "mode": None}

    # Lab 05 request fields, now filled from the decoded upload instead of trusted from the client.
    request = {
        "input_ref": file.filename or "",
        "format": image["format"],
        "width": image["width"],
        "height": image["height"],
        "source": source,
    }
    validation = validate_request(request, lab04.PREPROCESSING_ID)

    prediction_called, prediction, http_status = False, None, 200
    if validation["status"] == "pass":
        if not MODEL_FILE.exists():
            # Lab 04 'unavailable' path: the model file is missing on this machine.
            validation = {**validation, "status": "unavailable", "action": "sort_by_hand"}
            http_status = 503
        else:
            start = time.perf_counter()
            out = PREDICT(data)
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            prediction_called = True
            prediction = {
                "status": "review" if out["needs_review"] else "ok",  # decision-v1
                "result": out["result"],
                "latency_ms": latency_ms,
            }
    elif validation["status"] == "error":
        http_status = 400

    staff_action = {
        "pass": (lab04.REVIEW_ACTION if prediction and prediction["status"] == "review" else lab04.NORMAL_ACTION),
        "warning": lab04.REVIEW_MESSAGE,
        "error": lab04.REJECT_MESSAGE,
        "unavailable": lab04.UNAVAILABLE_ACTION,
    }[validation["status"]]

    record = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": SERVICE_NAME,
        "host": socket.gethostname(),
        "input": {"filename": file.filename, "bytes": len(data), "source": source, **image},
        "contract_version": validation["contract_version"],
        "preprocessing_id": lab04.PREPROCESSING_ID,
        "model_version": lab04.MODEL_VERSION,
        "rule_version": lab04.RULE_VERSION,
        "checks": validation["checks"],
        "status": validation["status"],
        "action": validation["action"],
        "prediction_called": prediction_called,
        "prediction": prediction,
        "staff_action": staff_action,
    }
    write_record(record)
    return JSONResponse(status_code=http_status, content=record)
