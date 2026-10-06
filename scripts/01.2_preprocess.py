import hashlib
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from PIL import Image

from common import PROCESSED_DIR, extract_features, list_images

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "dataset" / "tomato"))
SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
IMG_SIZE = 64
DROP_DUPLICATES = False
# คลาสที่ใช้ (ชื่อสั้น) ค่าเริ่มต้น 3 คลาสตามโจทย์; ตั้ง CLASSES=all เพื่อใช้ทุกคลาส
_classes_env = os.getenv("CLASSES", "Healthy,Mosaic_virus,Yellow_Leaf_Curl_Virus").strip()
CLASSES = (
    None
    if _classes_env.lower() == "all"
    else {c.strip() for c in _classes_env.split(",") if c.strip()}
)


def short_label(class_dir_name):
    """'Tomato__Tomato_Mosaic_virus' -> 'Mosaic_virus'"""
    return class_dir_name.split("Tomato_")[-1].lstrip("_")


def load_split(split, seen_hashes):
    """อ่านภาพทุกคลาสใน split หนึ่ง คืนค่าเป็น DataFrame (ฟีเจอร์ + target)
    พร้อมนับภาพที่ซ้ำกับภาพที่เคยอ่านแล้ว (dataset นี้มีภาพเดียวกันอยู่หลาย split)"""
    rows, labels, duplicates = [], [], 0
    class_dirs = sorted(
        p
        for p in (DATA_DIR / split).iterdir()
        if p.is_dir() and (CLASSES is None or short_label(p.name) in CLASSES)
    )
    for class_dir in class_dirs:
        label = short_label(class_dir.name)
        for f in list_images(class_dir):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            digest = hashlib.md5(f.read_bytes()).hexdigest()
            if digest in seen_hashes:
                duplicates += 1
                if DROP_DUPLICATES:
                    continue
            seen_hashes.add(digest)
            with Image.open(f) as im:
                rows.append(extract_features(im))
            labels.append(label)

    X = pd.DataFrame(np.stack(rows), columns=[f"f{i}" for i in range(len(rows[0]))])
    y = pd.Series(labels, name="target")
    return X, y, duplicates


def preprocess_data():
    """
    Loads tomato leaf images (already split into train/val/test folders),
    converts each image into a feature vector, saves them as CSV files,
    and logs the resulting datasets as artifacts in MLflow.
    """
    mlflow.set_experiment("Tomato Leaf Disease - Data Preprocessing")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data preprocessing run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")
        print(f"Reading images from: {DATA_DIR}")

        processed_data_dir = str(PROCESSED_DIR)
        os.makedirs(processed_data_dir, exist_ok=True)

        seen_hashes = set()
        for split in SPLITS:
            X, y, duplicates = load_split(split, seen_hashes)
            pd.concat([X, y], axis=1).to_csv(
                os.path.join(processed_data_dir, f"{split}.csv"), index=False, float_format="%.5f"
            )
            action = "dropped" if DROP_DUPLICATES else "kept"
            print(
                f"{split:>5}: {len(X)} images ({action} {duplicates} duplicates), "
                f"classes = {y.value_counts().to_dict()}"
            )

            mlflow.log_metric(f"{split}_set_rows", len(X))
            mlflow.log_metric(f"{split}_duplicates", duplicates)

        print(f"Saved processed data to '{processed_data_dir}' directory.")

        mlflow.log_param("data_dir", str(DATA_DIR))
        mlflow.log_param("img_size", IMG_SIZE)
        mlflow.log_param("drop_duplicates", DROP_DUPLICATES)
        mlflow.log_param("class_filter", "all" if CLASSES is None else sorted(CLASSES))
        mlflow.log_param("features", "shared common.extract_features")
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("classes", sorted(y.unique()))

        mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data")
        print("Logged processed data as artifacts in MLflow.")

        print("-" * 50)
        print("Data preprocessing run finished. Please use the following Run ID for the next step:")
        print(f"Preprocessing Run ID: {run_id}")
        print("-" * 50)


if __name__ == "__main__":
    preprocess_data()
