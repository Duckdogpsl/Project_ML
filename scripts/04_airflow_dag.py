"""(ทางเลือก) Pipeline เดียวกันในรูป Airflow DAG

Airflow รันบน Windows ตรง ๆ ไม่ได้ ต้องใช้ WSL2 หรือ Docker
คัดลอก/ลิงก์ไฟล์นี้ไปไว้ในโฟลเดอร์ dags ของ Airflow แล้วตั้ง env
    export TOMATO_PROJECT_DIR=/path/to/project
    airflow standalone        # เปิด http://localhost:8080
กด Trigger DAG พร้อมแก้ params ได้จากหน้า UI
"""

import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param

try:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # Airflow 2
    from airflow.operators.bash import BashOperator

PROJECT_DIR = os.getenv("TOMATO_PROJECT_DIR", str(Path(__file__).resolve().parents[1]))

ENV = {
    "DATA_DIR": "{{ params.data_dir }}",
    "MAX_PER_CLASS": "{{ params.max_per_class }}",
    "MODELS": "{{ params.models }}",
    "MIN_VAL_ACCURACY": "{{ params.min_val_accuracy }}",
    "PYTHONUTF8": "1",
}

with DAG(
    dag_id="tomato_leaf_mlops_pipeline",
    start_date=datetime(2026, 10, 1),
    schedule="0 2 * * *",
    catchup=False,
    tags=["cp413008", "tomato"],
    params={
        "data_dir": Param("dataset/sample", type="string"),
        "max_per_class": Param(0, type="integer"),
        "models": Param("random_forest,svc_rbf", type="string"),
        "min_val_accuracy": Param(0.70, type="number"),
    },
) as dag:

    def step(task_id: str, script: str) -> BashOperator:
        # exit code != 0 จาก SystemExit ในสคริปต์ -> task failed -> task ถัดไปไม่รัน
        return BashOperator(
            task_id=task_id,
            bash_command=f"cd {PROJECT_DIR} && python scripts/{script}",
            env=ENV,
            append_env=True,
        )

    (
        step("data_validation", "01_data_validation.py")
        >> step("data_preprocessing", "02_data_preprocessing.py")
        >> step("train_evaluate_register", "03_train_evaluate_register.py")
        >> step("load_and_predict", "04_load_and_predict.py")
    )
