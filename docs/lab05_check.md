# Lab 05 — Path check

A teammate ran the demo from a clean clone:

```bash
python src/inference_demo.py
python -m unittest tests/test_input_validation.py
```

Output: `lab05-pass-001` **pass**, `lab05-error-002` **error**, `lab05-warning-003` **warning**; `Ran 3 tests ... OK`.
Trace: [`reports/validation_demo.jsonl`](../reports/validation_demo.jsonl). Diagram: [`input_validation_flow.png`](input_validation_flow.png).

## Questions and answers

**1. Which request reached prediction?**
Only `lab05-pass-001` (`glass373.jpg`, jpg, 512×384, source `tray-camera`). All five checks passed, so `action = call_prediction` and `prediction_called = true`. `baseline-v1` returned **glass, 0.983**, and `decision-v1` gave `prediction_status = ok`.

**2. Why did the warning request stop?**
`lab05-warning-003` is the same readable 512×384 jpg, but its `source` is `phone-upload`. The `source_is_tray_camera` rule observed `phone-upload`, which is not in the approved list (`tray-camera`, `trashnet-test`), and returned `warning`. Lab 03 narrowed the system to one item on a plain tray under a fixed camera, so a phone photo is outside the conditions the model was evaluated in. A warning means `route_for_review`: the record has `prediction_called = false`, and staff see "Check by hand".

**3. Which checks produced an error, and how did severity precedence make the overall status error?**
For `lab05-error-002`:
- `format_supported` observed `png` → **error** (only jpg/jpeg are allowed);
- `dimensions_match_tray_camera` observed `1280x720` → **error** (512×384 is required);
- `source_is_tray_camera` observed `phone-upload` → **warning**;
- `input_reference_present` and `preprocessing_identity_match` → pass.

`combine()` takes the most severe status (error > warning > pass), so the warning is overridden: `validation_status = error`, `action = reject_input`, `prediction_called = false`. The test `test_error_overrides_warning` checks exactly this.

**4. Which fields connect the validation decision to the generated trace?**
Each request produces one combined record. `request_id` ties the validation decision and the prediction outcome to the same attempt. `checks[]` (`rule`, `observed`, `status`, `action` per rule) explains which rules ran and what they saw. `validation_status` and `action` give the combined decision. `prediction_called`, `prediction_status` and `result` show whether prediction ran and what it returned. `contract_version`, `preprocessing_id`, `model_version` and `rule_version` say which contract, preprocessing, model and review rule were used.

## Completion check
- [x] One command produces pass, warning and error records
- [x] Only the passing request reaches prediction
- [x] Tests pass (error overrides warning; warning does not call prediction; pass does)
- [x] The trace explains every decision
