"""ค่าตั้งต้นและฟังก์ชันที่ทุกขั้นของ pipeline ใช้ร่วมกัน

ปรับค่าได้ผ่าน environment variable โดยไม่ต้องแก้โค้ด เช่น
    DATA_DIR=dataset/sample MAX_PER_CLASS=50 python scripts/01_data_validation.py
"""

import os
from pathlib import Path

import mlflow
import numpy as np
from PIL import Image
from skimage.feature import hog

ROOT = Path(__file__).resolve().parents[1]

# ---------- ข้อมูล ----------
DATA_DIR = ROOT / os.getenv("DATA_DIR", "dataset/tomato")
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# จำกัดจำนวนภาพต่อคลาสต่อ split (0 = ใช้ทั้งหมด) — ใช้ลดเวลาตอนทดลองหรือรันใน CI
MAX_PER_CLASS = int(os.getenv("MAX_PER_CLASS", "0"))
SEED = 42

# ---------- ฟีเจอร์ ----------
IMG_SIZE = 64          # ย่อภาพเป็น 64x64 ก่อนสกัดฟีเจอร์
HIST_BINS = 8          # ฮิสโตแกรมสี HSV แบบ 8x8x8 = 512 ค่า

# ---------- ผลลัพธ์ / MLflow ----------
PROCESSED_DIR = ROOT / "processed_data"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
EXPERIMENT = "Tomato Leaf Disease"
MODEL_NAME = "tomato-leaf-classifier"
MODEL_ALIAS = "champion"
# โมเดลต้องได้ accuracy บน val อย่างน้อยเท่านี้จึงจะถูก register (quality gate)
MIN_VAL_ACCURACY = float(os.getenv("MIN_VAL_ACCURACY", "0.70"))


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    # เก็บ artifact (โมเดล, กราฟ) ไว้ที่ <project>/mlruns เสมอ ไม่ว่าจะรันจากโฟลเดอร์ไหน
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def short_label(class_dir_name: str) -> str:
    """'Tomato__Tomato_Leaf_Mold' -> 'Leaf_Mold'"""
    return class_dir_name.split("Tomato_")[-1].lstrip("_")


def list_classes(split_dir: Path) -> list[str]:
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def list_images(class_dir: Path) -> list[Path]:
    files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if MAX_PER_CLASS and len(files) > MAX_PER_CLASS:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(files), MAX_PER_CLASS, replace=False)
        files = [files[i] for i in sorted(idx)]
    return files


def extract_features(image: Image.Image) -> np.ndarray:
    """แปลงภาพ 1 ภาพเป็นเวกเตอร์ฟีเจอร์ (ฮิสโตแกรมสี HSV + HOG ของรูปทรง/ลวดลาย)

    ต้องใช้ฟังก์ชันนี้ทั้งตอนเทรนและตอนทำนาย เพื่อให้ฟีเจอร์ตรงกันทุกครั้ง
    """
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    hsv = np.asarray(img.convert("HSV"), dtype=np.uint8).reshape(-1, 3)
    hist, _ = np.histogramdd(hsv, bins=HIST_BINS, range=[(0, 256)] * 3)
    hist = hist.ravel() / hsv.shape[0]

    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    return np.concatenate([hist, hog_feat]).astype(np.float32)


def extract_features_from_path(path: Path | str) -> np.ndarray:
    with Image.open(path) as im:
        return extract_features(im)
