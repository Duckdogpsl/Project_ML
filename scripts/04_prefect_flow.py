"""Pipeline Orchestration ด้วย Prefect — ร้อยสคริปต์ 01 -> 02 -> 03 -> 04 เป็น flow เดียว

โครงสร้างโปรเจกต์ที่คาดไว้
    <project>/
      scripts/   01_data_validation.py, 02_..., 03_..., 04_..., common.py
      pipelines/ prefect_flow.py   <- ไฟล์นี้

แต่ละ task รันสคริปต์เดิมเป็น subprocess (ไม่ต้องแก้สคริปต์เลย) เพราะ
  - common.py อ่าน env (DATA_DIR, CLASSES, ...) ตอน import จึงต้องตั้ง env ก่อนเริ่ม process
  - สคริปต์ใช้ raise SystemExit เป็น quality gate -> exit code != 0 -> task ล้ม -> flow หยุด
    (ถ้า import มารันใน process เดียวกัน SystemExit จะไปฆ่า worker ของ Prefect)

ใช้งาน
    pip install prefect
    python pipelines/prefect_flow.py                          # รัน 1 ครั้งด้วย dataset/sample
    python pipelines/prefect_flow.py --data-dir dataset/tomato --models random_forest,svc_rbf
    python pipelines/prefect_flow.py --serve                  # สร้าง deployment ตั้งเวลารันทุกวัน

ดู UI:  prefect server start   แล้วเปิด http://127.0.0.1:4200
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _run_script(script: str, env_overrides: dict[str, str], args: list[str] | None = None) -> str:
    """รันสคริปต์ 1 ตัว ส่ง log ทีละบรรทัดเข้า Prefect และคืน output ทั้งหมด"""
    logger = get_run_logger()
    env = {**os.environ, **env_overrides, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, str(SCRIPTS / script), *(args or [])]
    logger.info("Running: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    lines = []
    for line in proc.stdout:
        line = line.rstrip()
        lines.append(line)
        logger.info(line)
    proc.wait()

    output = "\n".join(lines)
    if proc.returncode != 0:
        raise RuntimeError(f"{script} ล้มเหลว (exit code {proc.returncode})\n{output[-1500:]}")
    return output


@task(name="1. data_validation", tags=["data"])
def validate(env: dict[str, str]) -> str:
    return _run_script("01_data_validation.py", env)


@task(name="2. data_preprocessing", tags=["data"], retries=1, retry_delay_seconds=10)
def preprocess(env: dict[str, str]) -> str:
    return _run_script("02_data_preprocessing.py", env)


@task(name="3. train_evaluate_register", tags=["model"])
def train_register(env: dict[str, str]) -> str:
    return _run_script("03_train_evaluate_register.py", env)


@task(name="4. load_and_predict (smoke test)", tags=["model"])
def smoke_test(env: dict[str, str]) -> str:
    return _run_script("04_load_and_predict.py", env)


@flow(name="tomato-leaf-mlops-pipeline", log_prints=True)
def tomato_pipeline(
    data_dir: str = "dataset/sample",
    classes: str = "",
    max_per_class: int = 0,
    models: str = "random_forest,svc_rbf",
    min_val_accuracy: float = 0.70,
    dedup: bool = True,
):
    """validate -> preprocess -> train/evaluate/register -> smoke test โมเดล @champion"""
    env = {
        "DATA_DIR": data_dir,
        "MAX_PER_CLASS": str(max_per_class),
        "MODELS": models,
        "MIN_VAL_ACCURACY": str(min_val_accuracy),
        "DEDUP": "1" if dedup else "0",
    }
    if classes:  # ว่าง = ใช้ DEFAULT_CLASSES ใน common.py
        env["CLASSES"] = classes

    # ส่ง future ต่อเป็น wait_for เพื่อบังคับลำดับ: ขั้นก่อนล้ม ขั้นถัดไปจะไม่รัน
    v = validate.submit(env)
    p = preprocess.submit(env, wait_for=[v])
    t = train_register.submit(env, wait_for=[p])
    s = smoke_test.submit(env, wait_for=[t])

    train_out = t.result()
    smoke_out = s.result()

    # สรุปผลเป็น artifact ให้ดูในหน้า UI ของ Prefect
    keep = [
        line for line in train_out.splitlines()
        if "val=" in line or "val_acc" in line or line.startswith(("Best model", "Registered"))
    ]
    create_markdown_artifact(
        key="pipeline-summary",
        markdown=(
            "## Tomato leaf pipeline\n\n"
            f"- data_dir: `{data_dir}`  models: `{models}`  gate: val_acc ≥ {min_val_accuracy}\n\n"
            "### Training\n```\n" + "\n".join(keep) + "\n```\n\n"
            "### Smoke test\n```\n" + "\n".join(smoke_out.splitlines()[-3:]) + "\n```"
        ),
        description="สรุปผลการเทรนและทดสอบโมเดล",
    )
    return {"train": keep, "smoke_test": smoke_out.splitlines()[-1:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/sample")
    ap.add_argument("--classes", default="")
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--models", default="random_forest,svc_rbf")
    ap.add_argument("--min-val-accuracy", type=float, default=0.70)
    ap.add_argument("--serve", action="store_true", help="สร้าง deployment ตั้งเวลารันทุกวัน 02:00")
    a = ap.parse_args()

    params = dict(
        data_dir=a.data_dir, classes=a.classes, max_per_class=a.max_per_class,
        models=a.models, min_val_accuracy=a.min_val_accuracy,
    )
    if a.serve:
        # process นี้ต้องเปิดค้างไว้ — กด Run จาก UI หรือรอให้ถึงเวลาตาม cron
        tomato_pipeline.serve(
            name="daily-retrain",
            cron="0 2 * * *",
            parameters=params,
            tags=["cp413008", "tomato"],
        )
    else:
        tomato_pipeline(**params)
