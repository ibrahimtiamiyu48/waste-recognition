"""Waste Recognition — Lab 02 baseline pipeline.

Smallest complete path: get TrashNet -> check it -> use the dataset's own
train/test lists -> fit a simple baseline on colour features -> measure macro F1
over 4 materials -> write reports/baseline_metrics.json.

Run from the repository root:
    python src/baseline_pipeline.py
"""

import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ZIP_FILE = RAW / "dataset-resized.zip"
IMAGE_DIR = RAW / "dataset-resized"
REPORT_FILE = ROOT / "reports" / "baseline_metrics.json"
MODEL_FILE = ROOT / "models" / "baseline-v1.joblib"  # reused by src/inference_demo.py

BASE_URL = "https://raw.githubusercontent.com/garythung/trashnet/master/data/"
DOWNLOADS = {
    "dataset-resized.zip": ZIP_FILE,
    "one-indexed-files-notrash_train.txt": RAW / "one-indexed-files-notrash_train.txt",
    "one-indexed-files-notrash_test.txt": RAW / "one-indexed-files-notrash_test.txt",
}

CLASSES = ["plastic", "paper", "glass", "metal"]  # cardboard and trash are left out
THUMB = (32, 24)  # width, height; the source images are 512x384
SEED = 42

# Team-written descriptions saved together with the measured numbers.
EXPERIMENT = "waste_material_baseline_v1"
MODEL = "logistic regression on 32x24 RGB pixels + HSV colour histogram (no deep learning)"
LIMITATION = (
    "one object per photo on a white posterboard in good light; real bins have "
    "mixed, dirty, crumpled items on busy backgrounds, so this score does not "
    "show how the system would work in a real sorting setting"
)


def download_if_missing() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, target in DOWNLOADS.items():
        if target.exists():
            continue
        print(f"Downloading {name} ...")
        try:
            urllib.request.urlretrieve(BASE_URL + name, target)
        except Exception as err:
            target.unlink(missing_ok=True)
            sys.exit(
                f"Could not download {name} ({err}).\n"
                f"Get it from https://github.com/garythung/trashnet/tree/master/data\n"
                f"and place it in data/raw/ (see data/README.md)."
            )
    if not IMAGE_DIR.exists():
        print("Unzipping images ...")
        with zipfile.ZipFile(ZIP_FILE) as zf:
            members = [m for m in zf.namelist() if m.startswith("dataset-resized/")]
            zf.extractall(RAW, members=members)


def read_split(name: str) -> list[tuple[Path, int]]:
    """Read a TrashNet split list and keep only our 4 materials."""
    items = []
    for line in (RAW / f"one-indexed-files-notrash_{name}.txt").read_text().split("\n"):
        if not line.strip():
            continue
        filename = line.split()[0]
        material = re.match(r"[a-z]+", filename).group()
        if material in CLASSES:
            items.append((IMAGE_DIR / material / filename, CLASSES.index(material)))
    return items


def features(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    pixels = np.asarray(img.resize(THUMB), dtype=np.float32).ravel() / 255.0
    hsv = np.asarray(img.convert("HSV"))
    hist = [np.histogram(hsv[..., c], bins=16, range=(0, 256), density=True)[0] for c in range(3)]
    return np.concatenate([pixels, *hist])


def load_and_check(items: list[tuple[Path, int]], split: str) -> tuple[np.ndarray, np.ndarray]:
    missing = [p for p, _ in items if not p.exists()]
    if missing:
        sys.exit(f"{split}: {len(missing)} listed images are missing, e.g. {missing[0]}")
    x, y = [], []
    for path, label in items:
        with Image.open(path) as img:
            if img.size != (512, 384):
                sys.exit(f"{path.name}: size {img.size}, expected (512, 384)")
        x.append(features(path))
        y.append(label)
    y = np.array(y)
    counts = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
    if min(counts.values()) == 0:
        sys.exit(f"{split}: a class has no images: {counts}")
    print(f"{split}: {len(y)} images {counts}")
    return np.stack(x), y


def main() -> None:
    download_if_missing()
    train_items, test_items = read_split("train"), read_split("test")
    overlap = {p for p, _ in train_items} & {p for p, _ in test_items}
    if overlap:
        sys.exit(f"{len(overlap)} images are in both train and test")

    x_train, y_train = load_and_check(train_items, "train")
    x_test, y_test = load_and_check(test_items, "test")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced", random_state=SEED),
    )
    model.fit(x_train, y_train)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": CLASSES}, MODEL_FILE)
    pred = model.predict(x_test)

    macro_f1 = f1_score(y_test, pred, average="macro")
    per_class_recall = recall_score(y_test, pred, average=None, labels=range(len(CLASSES)))
    cm = confusion_matrix(y_test, pred, labels=range(len(CLASSES)))

    result = {
        "experiment": EXPERIMENT,
        "data": f"TrashNet photos of 4 materials ({', '.join(CLASSES)}): {len(y_train)} for training",
        "evaluation": f"TrashNet's own test list, {len(y_test)} images, never used for fitting",
        "model": MODEL,
        "metric": {"name": "macro_f1", "value": round(float(macro_f1), 4)},
        "supporting": {
            "accuracy": round(float((pred == y_test).mean()), 4),
            "recall_per_class": {c: round(float(r), 4) for c, r in zip(CLASSES, per_class_recall)},
            "confusion_matrix": {
                "rows_true_cols_predicted": CLASSES,
                "values": cm.tolist(),
            },
            "seed": SEED,
        },
        "limitation": LIMITATION,
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {REPORT_FILE.relative_to(ROOT)}")
    print(f"Saved model {MODEL_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
