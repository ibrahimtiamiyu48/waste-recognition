"""Lab 05 tests. Run from the repository root:
    python3 -m unittest tests/test_input_validation.py
They use a spy instead of the model, so no model file or dataset is needed.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_demo import PREPROCESSING_ID, route_request  # noqa: E402
from input_validation import validate_request  # noqa: E402

TRAY_PHOTO = {
    "input_ref": "data/raw/dataset-resized/paper/paper10.jpg",
    "format": "jpg", "width": 512, "height": 384, "source": "tray-camera",
}


class SpyPredict:
    """Counts calls; returns the same shape as project_predict()."""

    def __init__(self):
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        return {"result": {"material": "paper", "confidence": 0.9,
                           "runner_up": {"material": "plastic", "confidence": 0.05}},
                "needs_review": False}


class TestInputValidation(unittest.TestCase):
    def test_error_overrides_warning(self):
        request = dict(TRAY_PHOTO, width=1280, height=720, source="phone-upload")
        result = validate_request(request, PREPROCESSING_ID)
        by_rule = {c["rule"]: c["status"] for c in result["checks"]}
        self.assertEqual(by_rule["dimensions_match_tray_camera"], "error")
        self.assertEqual(by_rule["source_is_tray_camera"], "warning")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["action"], "reject_input")

    def test_warning_does_not_call_prediction(self):
        spy = SpyPredict()
        record = route_request(dict(TRAY_PHOTO, source="phone-upload"), spy)
        self.assertEqual(record["validation_status"], "warning")
        self.assertEqual(record["action"], "route_for_review")
        self.assertFalse(record["prediction_called"])
        self.assertEqual(spy.calls, 0)

    def test_pass_calls_prediction(self):
        spy = SpyPredict()
        record = route_request(dict(TRAY_PHOTO), spy)
        self.assertEqual(record["validation_status"], "pass")
        self.assertTrue(record["prediction_called"])
        self.assertEqual(spy.calls, 1)
        self.assertEqual(record["result"]["material"], "paper")


if __name__ == "__main__":
    unittest.main()
