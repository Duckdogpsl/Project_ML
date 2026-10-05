import hashlib
import os
import platform
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import mlflow
import numpy as np
import pandas as pd

from common import DATA_DIR, MAX_PER_CLASS, PROCESSED_DIR, ROOT, SEED, SPLITS

TRACKED_PACKAGES = [
    "mlflow", "scikit-learn", "scikit-image", "numpy", "pandas", "pillow", "matplotlib",
]


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def git_info() -> dict[str, str]:
    """อ่านเวอร์ชันโค้ดจาก git (ถ้าไม่มี .git เช่นใน Docker ให้ส่งผ่าน env GIT_COMMIT)"""
    commit = os.getenv("GIT_COMMIT") or _git("rev-parse", "HEAD") or "unknown"
    return {
        "git_commit": commit,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        # True = มีไฟล์ที่แก้แต่ยังไม่ commit ตอนรันทดลอง (ผลอาจย้อนทำซ้ำไม่ได้เป๊ะ)
        "git_dirty": str(bool(_git("status", "--porcelain"))),
    }


def load_processed(split: str):
    """อ่านข้อมูลที่ผ่านขั้น 02 แล้ว คืน (X, y)
    รองรับ 2 รูปแบบ: .csv (02 ของเพื่อน) หรือ .npz (02 เดิม)"""
    csv_path = PROCESSED_DIR / f"{split}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # float32 เพื่อให้ตรงกับฟีเจอร์ที่ 02 สร้าง และประหยัดความจำ
        return df.drop(columns="target").to_numpy(dtype=np.float32), df["target"].to_numpy()
    d = np.load(PROCESSED_DIR / f"{split}.npz")
    return d["X"], d["y"]


@lru_cache(maxsize=1)
def data_version() -> str:
    """hash ของข้อมูลที่โมเดลเรียนจริง (X และ y ของทุก split)

    hash จากเนื้อ array ไม่ใช่จากไฟล์ .npz เพราะไฟล์ zip ฝังเวลาสร้างไว้
    ทำให้ข้อมูลเดิมได้ hash ไม่ตรงกันทุกครั้งที่รัน 02 ใหม่
    """
    h = hashlib.sha256()
    for split in SPLITS:
        X, y = load_processed(split)
        h.update(np.ascontiguousarray(X).tobytes())
        h.update("|".join(y.tolist()).encode())
    return h.hexdigest()[:12]


def environment_info() -> dict:
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages}


def pip_freeze() -> str:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60
        )
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return "pip freeze unavailable"


def log_provenance(full_environment: bool = False) -> None:
    """เรียกภายใน mlflow run เพื่อบันทึกเวอร์ชันโค้ด ข้อมูล และสภาพแวดล้อม"""
    mlflow.set_tags({**git_info(), "data_version": data_version()})
    mlflow.log_params({"seed": SEED, "data_dir": str(DATA_DIR), "max_per_class": MAX_PER_CLASS})
    mlflow.log_dict(environment_info(), "environment.json")
    if full_environment:
        mlflow.log_text(pip_freeze(), "requirements_freeze.txt")