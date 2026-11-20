# Project decisions

## Lab 03 — Acceptance

### Decision after the baseline
Decision: **Narrow**

Reason: The simple colour-based baseline reached a macro F1 of 0.607 on 332 held-out TrashNet photos. Glass was the weakest class, with a recall of 0.50 (41 of 82). An important limitation is that TrashNet shows only one clean item on a white board. We therefore decided to narrow the operating conditions: the system handles **one item at a time, placed on a plain tray under a fixed camera**. It does not handle a pile of mixed waste or a full bin. The users, the four materials and the useful output stay the same as in the project brief. The baseline shows that there is a learnable signal even without deep learning, so the direction is worth continuing within these conditions.

### Required behavior
- **Normal input:** one item on the tray, sharp and well-lit photo. The system returns the material (plastic, paper, glass or metal) and a confidence score. Staff put the item into that container.
- **Invalid input:** empty tray, more than one item, item partly outside the frame, or a photo that is too blurry or too dark. The system gives no material. Instead it shows the reason, e.g. "Place one item in the middle of the tray", and the staff member retakes the photo.
- **Valid input with an uncertain result:** the top confidence is below the cutoff, or the top two materials are close. The system shows "Check by hand" together with the two most likely materials. A staff member decides, and the photo and their decision are saved for later labelling.

### One quality that matters
- **Response time:** staff should not wait at the tray. Our target is a result within 2 seconds from photo to screen on a normal laptop CPU without a GPU. This is not measured yet. First we need to time the full path (photo → checks → model → screen) once a model is chosen.

### One acceptance check
- **Important failure:** a glass item is sent to the paper, plastic or metal container. Broken glass is dangerous for people who sort by hand, and it can make a whole batch of paper unusable for recycling.
- **Metric that would show this failure:** **glass recall** on the evaluation set, i.e. the share of glass items that are either labelled glass or sent to "Check by hand". This is the same as 1 − glass miss rate. It is taken from the per-class recall that the Lab 02 pipeline already saves.
- **Measured baseline (Lab 02, not a target):** glass recall = **0.50**. 41 of 82 glass test images were correct: 14 were predicted as plastic, 11 as paper and 16 as metal.
- **Preliminary acceptance threshold:** glass recall **≥ 0.90** in a later evaluation, with "Check by hand" counted as caught.
- **Where will we check it?** First on TrashNet's held-out test list (82 glass images). Later on our own labelled photos from a real collection point, if we get permission.
- **Why this number for now?** At a small collection point, staff can catch an occasional mistake, but missing half of the glass is not acceptable. 0.90 is a starting point the team agreed on, not an industry standard. It may become stricter once we know the real tolerance.
- **Evidence still missing:**
  - How much wrong-stream glass the recycling buyer actually accepts.
  - Glass recall on real, dirty, crushed items from a collection point.
  - A limit on how many items may go to "Check by hand": **not set yet**. Without one, the system could reach 0.90 just by sending everything to a person. We first need to count how many items per hour staff can check by hand.

### Failure cases
| Area | What could go wrong and why it matters | How would we notice it? | Planned response |
| --- | --- | --- | --- |
| Data or input | Two items on the tray, or an item half outside the photo. The system gives one material for both, so one item goes to the wrong container. | Input check finds more than one object or an object touching the image edge; staff report "wrong item" corrections. | Reject the photo with "Place one item in the middle of the tray" and ask for a retake. |
| Model behavior | A clear glass bottle is predicted as plastic, because both are transparent and bottle-shaped. Glass ends up in the plastic stream. **(Matches the acceptance check.)** | Glass recall on the evaluation set and the glass→plastic cell of the confusion matrix (14 of 82 in the baseline); staff corrections logged as "glass → plastic". | Send low-confidence glass/plastic results to "Check by hand"; add more clear-glass training photos; re-check glass recall against 0.90. |
| Integration or workflow | Too many items go to "Check by hand" at busy times. Staff start skipping the check and guess, so the fallback stops protecting against mistakes. | Share of items sent to "Check by hand" per hour, and how long those items wait before a decision is logged. | Show the two most likely materials to make checking fast; review the uncertainty cutoff weekly; set a limit once we have staff capacity numbers. |

## Lab 04 — Inference path

- **Trigger:** user request — staff place one item on the tray and take a photo.
- **Intended placement:** a shared laptop at the collection point; the camera only captures the photo.
- **Inference boundary:** `run_attempt()` in `src/inference_demo.py` takes an accepted photo and returns `request_id`, `status`, `result`, `model_version`, `rule_version`, `latency_ms`, `action`, `simulated`.
- **Model:** `baseline-v1`, the Lab 02 logistic regression, reused without retraining.
- **Decision rule `decision-v1`:** `review` when the top confidence is below 0.60, or when glass is the runner-up with at least 0.25. Otherwise `ok`.
- **What the rule does on the 332 test photos (measured):** 19% of items go to "Check by hand"; glass caught (correct or sent to review) rises from 0.50 to 0.62. This is still far below the 0.90 threshold from Lab 03, because the model is often confidently wrong (e.g. `glass12.jpg` is predicted plastic with 0.80 confidence).
- **Rejected input:** a photo that is unreadable or not 512×384 is rejected before the boundary with a reason, and no record is written.
- **Unavailable:** if prediction cannot run, `status = unavailable`, `result = null`, and staff sort by hand. The demo forces this once and marks it `simulated: true`.

## Lab 05 — Input assurance

- **Contract:** `config/input_contract.json` (`input-contract-v1`). The checks are request-level: they read the values in the request and do not open or decode the photo.
- **Rules:**
  - *Structural (error):* `input_reference_present`, `format_supported` (jpg), and `dimensions_match_tray_camera` (512×384, the only size the baseline was trained on, and the same size the Lab 04 check uses).
  - *Project-specific (warning):* `source_is_tray_camera`. The source must be `tray-camera` or `trashnet-test`. A phone photo is readable but falls outside the narrowed operating conditions from Lab 03, so a person checks it ("Check by hand").
  - *Runtime (error):* `preprocessing_identity_match`. `PREPROCESSING_ID = "features-v1"` is defined in `src/inference_demo.py`, not taken from the request. It names `baseline_pipeline.features()`.
- **Decision:** `pass` → `call_prediction`; `warning` → `route_for_review` (no automatic prediction); `error` → `reject_input`. When several rules fail, the most severe wins: error > warning > pass.
- **Trace:** `reports/validation_demo.jsonl` holds one record per request: `request_id`, every check (`rule`, `observed`, `status`, `action`), `validation_status`, `action`, `prediction_called`, and the prediction outcome when it ran.
- **Diagram:** [input_validation_flow.drawio](input_validation_flow.drawio) ([png](input_validation_flow.png)). **Path check:** [lab05_check.md](lab05_check.md).

## Lab 06 — Service on a cloud VM

- **Service:** `src/app.py` (FastAPI, `service-v1`), started with `uvicorn app:app --app-dir src --host 0.0.0.0 --port 8000`.
- **Endpoints:** `GET /health` returns service, host, `model_version`, `rule_version`, `contract_version`, `preprocessing_id` and `model_loaded`. `POST /predict` takes a multipart photo (`file`), plus optional `request_id` and `source` (default `tray-camera`).
- **Input checks before prediction:** the upload is decoded with Pillow first. The Lab 05 rules then run on the **decoded** format, width and height, so the service does not trust values claimed by the client. A file that does not decode gives `format = null` and fails `format_supported` and `dimensions_match_tray_camera`. Uploads over 5 MB are not decoded.
- **Routing:** `pass` → `baseline-v1` + `decision-v1` (HTTP 200); `warning` → `route_for_review`, no prediction (HTTP 200); `error` → `reject_input`, no prediction (HTTP 400). If the model file is missing: `unavailable`, staff sort by hand (HTTP 503).
- **Evidence:** one JSONL line per request in `reports/service_predictions.jsonl` on the VM: `request_id`, decoded input properties, `checks`, `status`, `action`, `prediction_called` and `prediction`. A committed example is in `reports/service_example.jsonl`.
- **Model:** the project's own `baseline-v1`, not the pretrained YOLO model from the reference demo. The demo request uses a TrashNet test photo, so this is a deployment check, not new evaluation evidence.
