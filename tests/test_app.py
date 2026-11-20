"""Lab 06 API tests. Run from the repository root:
    python3 -m unittest tests/test_app.py
A spy replaces the model, so no model file or dataset is needed.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

LOG = Path(tempfile.mkdtemp()) / "test_predictions.jsonl"
os.environ["SERVICE_LOG"] = str(LOG)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import app as service  # noqa: E402


def jpeg(size=(512, 384)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 200, 200)).save(buf, format="JPEG")
    return buf.getvalue()


class Spy:
    def __init__(self):
        self.calls = 0

    def __call__(self, data):
        self.calls += 1
        return {"result": {"material": "glass", "confidence": 0.9,
                           "runner_up": {"material": "plastic", "confidence": 0.05}},
                "needs_review": False}


class TestService(unittest.TestCase):
    def setUp(self):
        self.spy = Spy()
        service.PREDICT = self.spy
        service.MODEL_FILE = Path(__file__)  # any existing file: "model present"
        self.client = TestClient(service.app)
        LOG.unlink(missing_ok=True)

    def post(self, name, data, ctype, **form):
        return self.client.post("/predict", files={"file": (name, data, ctype)}, data=form)

    def records(self):
        return [json.loads(line) for line in LOG.read_text().splitlines()]

    def test_health_identifies_service_and_model(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["service"], "sortvision-waste-api")
        self.assertEqual(body["model_version"], "baseline-v1")
        self.assertEqual(body["preprocessing_id"], "features-v1")

    def test_valid_photo_reaches_prediction(self):
        r = self.post("tray.jpg", jpeg(), "image/jpeg", request_id="t-valid")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "pass")
        self.assertTrue(r.json()["prediction_called"])
        self.assertEqual(self.spy.calls, 1)

    def test_text_file_is_rejected_before_prediction(self):
        r = self.post("notes.txt", b"not an image", "text/plain", request_id="t-text")
        self.assertEqual(r.status_code, 400)
        body = r.json()
        self.assertEqual((body["status"], body["action"], body["prediction_called"]),
                         ("error", "reject_input", False))
        self.assertIsNone(body["prediction"])
        self.assertEqual(self.spy.calls, 0)

    def test_wrong_size_uses_decoded_size_not_client_claim(self):
        r = self.post("big.jpg", jpeg((1280, 720)), "image/jpeg")
        self.assertEqual(r.status_code, 400)
        failed = {c["rule"]: c["observed"] for c in r.json()["checks"] if c["status"] == "error"}
        self.assertEqual(failed, {"dimensions_match_tray_camera": "1280x720"})
        self.assertEqual(self.spy.calls, 0)

    def test_phone_photo_goes_to_review_without_prediction(self):
        r = self.post("phone.jpg", jpeg(), "image/jpeg", source="phone-upload")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["action"], "route_for_review")
        self.assertFalse(r.json()["prediction_called"])
        self.assertEqual(self.spy.calls, 0)

    def test_jsonl_contains_both_decisions(self):
        self.post("tray.jpg", jpeg(), "image/jpeg", request_id="t-a")
        self.post("notes.txt", b"hello", "text/plain", request_id="t-b")
        recs = {r["request_id"]: r for r in self.records()}
        self.assertTrue(recs["t-a"]["prediction_called"])
        self.assertFalse(recs["t-b"]["prediction_called"])
        self.assertEqual(recs["t-b"]["input"]["decoded"], False)


if __name__ == "__main__":
    unittest.main()
