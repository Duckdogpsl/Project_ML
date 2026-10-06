"""ขั้นที่ 3: ดูทะเบียนโมเดลและย้อนกลับ (rollback) ไปเวอร์ชันก่อนหน้า

    python scripts/05_rollback.py            # แสดงทุกเวอร์ชัน และตัวที่เป็น @champion
    python scripts/05_rollback.py previous   # ย้าย @champion ไปเวอร์ชันก่อนหน้า
    python scripts/05_rollback.py 1          # ย้าย @champion ไปเวอร์ชัน 1 ตามที่ระบุ

หลัง rollback ต้องรีสตาร์ท API เพื่อให้โหลดโมเดลตัวที่ถูกย้อนกลับ
"""

import sys

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from common import MODEL_ALIAS, MODEL_NAME, setup_mlflow


def champion_version(client):
    try:
        return int(client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version)
    except MlflowException:
        return None


def show(client):
    champ = champion_version(client)
    versions = sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version))
    print(f"{'ver':>4} {'champion':<9} {'status':<10} {'algorithm':<14} {'val_acc':>8} "
          f"{'code':<9} {'data':<13}")
    for v in versions:
        t = v.tags
        mark = "<-- @" + MODEL_ALIAS if int(v.version) == champ else ""
        print(f"{v.version:>4} {mark:<9} {t.get('status', '-'):<10} {t.get('algorithm', '-'):<14} "
              f"{t.get('val_accuracy', '-'):>8} {t.get('git_commit', '-')[:8]:<9} "
              f"{t.get('data_version', '-'):<13}")
    return champ, [int(v.version) for v in versions]


def main(arg):
    setup_mlflow()
    client = MlflowClient()
    champ, all_versions = show(client)
    if arg is None:
        return
    if champ is None:
        raise SystemExit("ยังไม่มี @champion ให้ย้อนกลับ")

    if arg == "previous":
        older = [v for v in all_versions if v < champ]
        if not older:
            raise SystemExit(f"ไม่มีเวอร์ชันก่อนหน้า v{champ} ให้ย้อนกลับ")
        target = max(older)
    else:
        target = int(arg)
        if target not in all_versions:
            raise SystemExit(f"ไม่พบเวอร์ชัน {target} (มี: {all_versions})")

    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, str(target))
    client.set_model_version_tag(MODEL_NAME, str(target), "status", "champion")
    client.set_model_version_tag(MODEL_NAME, str(champ), "status", "rolled_back")
    print(f"\nRollback: @{MODEL_ALIAS} v{champ} -> v{target}")
    print("รีสตาร์ท API เพื่อโหลดโมเดลเวอร์ชันนี้")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)