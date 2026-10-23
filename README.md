# Waste Recognition by Material

**Project direction:** We want to help staff at recycling collection points use photos of waste items to sort each item into the right material stream: plastic, paper, glass or metal.

## Team
- **Team name:** SortVision
- **Members:** Wu Wenxuan, Tiamiyu Ibrahim, Dorzhiev Erdeni, Gunzenov Burin-Jargal
- **Communication:** team Telegram group _(invite link shared inside the team only)_

## Documents
- [Project brief](docs/project_brief.md): user, input/output, data plan (Lab 01)
- [Data instructions](data/README.md): dataset, licence, where to put it (Lab 02)
- [Split strategy](docs/split_strategy.md): what is used for training and evaluation (Lab 02)

## Run the baseline (Lab 02)
Environment: **Python 3.10 or newer** (tested with 3.13). Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/baseline_pipeline.py
```

The first run downloads TrashNet (~43 MB) into `data/raw/` and unzips it. Without internet access, follow [data/README.md](data/README.md) to place the files by hand.

The command creates or updates **`reports/baseline_metrics.json`**, which holds the measured macro F1 and the run's limitation. A run takes about a minute on a laptop.
