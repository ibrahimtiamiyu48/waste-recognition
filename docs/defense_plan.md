# Project defense plan — SortVision (Waste Recognition)

## 5-minute live demo
1. Start the VM in Yandex Cloud and note the public IP (it can change after a restart).
2. `ssh <username>@<public-ip>`, then `cd ~/waste-recognition`.
3. Start the container (README, "Run the service in Docker").
4. From a laptop, open `http://<public-ip>:8000/health`. Point out `model_version`, `rule_version`, `contract_version` and `preprocessing_id`.
5. From the laptop, run `bash scripts/demo_requests.sh <public-ip>`:
   - `examples/glass373.jpg` → HTTP 200, **glass ≈ 0.98**, `prediction_called: true`;
   - `examples/not_an_image.txt` → HTTP 400, `reject_input`, `prediction_called: false`.
6. On the VM, run `tail -n 2 reports/service_predictions.jsonl` and walk through `checks`, `status`, `action`, `prediction_called` and `prediction`.
7. Replace the container, then show `/health` from the new container and that the old records are still in the file.

## The rejected input, explained
The text file does not decode as an image, so the service records `decoded: false`, `format: null` and size `null`. `format_supported` and `dimensions_match_tray_camera` return **error**, and the most severe status wins (error > warning > pass). The action is `reject_input`, the model is never called, and staff see "retake the photo with the tray camera". The JSONL record proves all of this with the same `request_id`.

## Fallback if the live connection fails
- `reports/service_example.jsonl`: real VM trace (one valid and one rejected request, Lab 06)
- `reports/validation_demo.jsonl` and `docs/lab05_check.md`: Lab 05 validation path
- `docs/inference_architecture.png` and `docs/input_validation_flow.png`: diagrams
- Screenshots: VM `tail` output, `/docs` responses, container replacement (save them in the team chat)

## Who explains what (agree as a team)
| Part | Labs | Key files | Presenter |
| --- | --- | --- | --- |
| Problem, users, data (TrashNet, 4 materials, split) | 01–02 | `docs/project_brief.md`, `data/README.md`, `docs/split_strategy.md` | |
| Baseline and acceptance (macro F1 0.607, glass recall 0.50 → target 0.90) | 02–03 | `src/baseline_pipeline.py`, `reports/baseline_metrics.json`, `docs/project_decisions.md` | |
| Inference path and input validation | 04–05 | `src/inference_demo.py`, `src/input_validation.py`, `config/input_contract.json` | |
| Cloud service and container (live demo) | 06–07 | `src/app.py`, `Dockerfile`, `.dockerignore`, `scripts/demo_requests.sh` | |

## Honest limitations to state
- The model is a colour-based logistic regression. Glass recall is only 0.50 (0.62 with "Check by hand"), far below the 0.90 target.
- TrashNet shows one clean item on a white board, so performance on real, dirty items is unknown.
- The demo photo comes from the TrashNet test set, so the demo proves the deployment works, not model quality.
- The service uses plain HTTP and no authentication, which is acceptable for a classroom VM but not for production.
