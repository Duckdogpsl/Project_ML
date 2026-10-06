"""ขั้นที่ 5: โหลดโมเดล @champion จาก MLflow Model Registry แล้วทำนายภาพใบมะเขือเทศ

ใช้งาน
    python scripts/05_load_and_predict.py                     # สุ่มภาพจาก test มาทดสอบ
    python scripts/05_load_and_predict.py leaf1.jpg leaf2.jpg # ทำนายภาพที่ระบุ
    python scripts/05_load_and_predict.py path/to/folder      # ทำนายทุกภาพในโฟลเดอร์
"""

import sys
from pathlib import Path

import mlflow
import numpy as np

from common import (
    DATA_DIR,
    IMAGE_EXTS,
    MODEL_ALIAS,
    MODEL_NAME,
    SEED,
    extract_features_from_path,
    list_classes,
    setup_mlflow,
    short_label,
)

SAMPLES_PER_CLASS = 2


def collect_inputs(args: list[str]) -> list[tuple[Path, str | None]]:
    """คืนรายการ (path, label จริงถ้ารู้)"""
    if not args:
        rng = np.random.default_rng(SEED)
        items = []
        for c in list_classes(DATA_DIR / "test"):
            files = sorted((DATA_DIR / "test" / c).iterdir())
            for i in rng.choice(len(files), min(SAMPLES_PER_CLASS, len(files)), replace=False):
                items.append((files[i], short_label(c)))
        return items

    items = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            items += [(f, None) for f in sorted(p.rglob("*")) if f.suffix.lower() in IMAGE_EXTS]
        else:
            items.append((p, None))
    return items


def predict(args: list[str]):
    setup_mlflow()
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    print(f"Loading model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    items = collect_inputs(args)
    if not items:
        raise SystemExit("ไม่พบภาพสำหรับทำนาย")

    X = np.stack([extract_features_from_path(p) for p, _ in items])
    preds = model.predict(X)
    has_proba = hasattr(model, "predict_proba")
    probs = model.predict_proba(X).max(axis=1) if has_proba else [None] * len(items)

    correct = known = 0
    print(f"\n{'image':<45} {'prediction':<25} {'conf':>6}  true")
    for (path, true), pred, prob in zip(items, preds, probs):
        conf = f"{prob:.2f}" if prob is not None else "  -"
        mark = ""
        if true is not None:
            known += 1
            correct += pred == true
            mark = f"{true}  {'✓' if pred == true else '✗'}"
        print(f"{path.name[:44]:<45} {pred:<25} {conf:>6}  {mark}")

    if known:
        print(f"\nAccuracy on these samples: {correct}/{known} = {correct / known:.2%}")


if __name__ == "__main__":
    predict(sys.argv[1:])
