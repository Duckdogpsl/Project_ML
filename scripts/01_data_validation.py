import hashlib
import json
import os
from collections import defaultdict

import mlflow
from PIL import Image

from common import (
    CLASSES,
    DATA_DIR,
    SPLITS,
    list_classes,
    list_images,
    setup_mlflow,
    short_label,
)

MIN_IMAGES_PER_CLASS = int(os.getenv("MIN_IMAGES_PER_CLASS", "5"))


def validate_data():
    setup_mlflow()

    with mlflow.start_run(run_name="data_validation"):
        mlflow.set_tag("ml.step", "data_validation")
        mlflow.log_param("data_dir", str(DATA_DIR))
        print(f"Validating dataset at: {DATA_DIR}")

        errors: list[str] = []
        warnings: list[str] = []

        # 1. split ครบไหม
        missing = [s for s in SPLITS if not (DATA_DIR / s).is_dir()]
        if missing:
            raise SystemExit(f"Data validation failed — ไม่พบโฟลเดอร์ split: {missing}")

        # 2. คลาสตรงกันทุก split ไหม
        classes = {s: list_classes(DATA_DIR / s) for s in SPLITS}
        reference = classes["train"]
        if CLASSES is not None:
            missing_cls = sorted(set(CLASSES) - set(reference))
            if missing_cls:
                errors.append(f"ไม่พบคลาสที่กำหนดใน CLASSES: {missing_cls}")
            print(f"Using {len(reference)} classes: {[short_label(c) for c in reference]}")
        for s in SPLITS:
            if classes[s] != reference:
                errors.append(f"คลาสใน {s} ไม่ตรงกับ train: {sorted(set(classes[s]) ^ set(reference))}")

        counts: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
        corrupt: list[str] = []
        non_rgb = 0
        sizes: dict[str, int] = defaultdict(int)
        hashes: dict[str, set[str]] = {s: set() for s in SPLITS}

        for s in SPLITS:
            for c in classes[s]:
                files = list_images(DATA_DIR / s / c)
                counts[s][c] = len(files)
                # 3. จำนวนภาพพอไหม
                if len(files) < MIN_IMAGES_PER_CLASS:
                    errors.append(f"{s}/{c} มีภาพเพียง {len(files)} ภาพ (< {MIN_IMAGES_PER_CLASS})")
                for f in files:
                    data = f.read_bytes()
                    hashes[s].add(hashlib.md5(data).hexdigest())
                    # 4. ภาพเปิดได้และเป็นภาพสีไหม
                    try:
                        with Image.open(f) as im:
                            im.verify()
                        with Image.open(f) as im:
                            sizes[f"{im.width}x{im.height}"] += 1
                            if im.mode != "RGB":
                                non_rgb += 1
                    except Exception:
                        corrupt.append(str(f.relative_to(DATA_DIR)))

        if corrupt:
            errors.append(f"พบภาพเสีย {len(corrupt)} ไฟล์ เช่น {corrupt[:5]}")

        # 5. ภาพซ้ำข้าม split
        leak_train_val = len(hashes["train"] & hashes["val"])
        leak_train_test = len(hashes["train"] & hashes["test"])
        if leak_train_val or leak_train_test:
            warnings.append(
                f"พบภาพซ้ำกันข้าม split: train∩val={leak_train_val}, train∩test={leak_train_test}"
            )

        # ---------- สรุปผล ----------
        print(f"\n{'class':<42}" + "".join(f"{s:>8}" for s in SPLITS))
        for c in reference:
            print(f"{c:<42}" + "".join(f"{counts[s].get(c, 0):>8}" for s in SPLITS))
        totals = {s: sum(counts[s].values()) for s in SPLITS}
        print(f"{'TOTAL':<42}" + "".join(f"{totals[s]:>8}" for s in SPLITS))
        print(f"\nImage sizes: {dict(sizes)}")

        for s in SPLITS:
            mlflow.log_metric(f"num_images_{s}", totals[s])
        train_counts = list(counts["train"].values()) or [0]
        mlflow.log_metric("class_imbalance_ratio", max(train_counts) / max(min(train_counts), 1))
        mlflow.log_metric("corrupt_images", len(corrupt))
        mlflow.log_metric("non_rgb_images", non_rgb)
        mlflow.log_metric("dup_train_val", leak_train_val)
        mlflow.log_metric("dup_train_test", leak_train_test)
        mlflow.log_param("num_classes", len(reference))
        mlflow.log_dict(
            {"counts": counts, "image_sizes": dict(sizes), "errors": errors, "warnings": warnings},
            "validation_report.json",
        )

        status = "Failed" if errors else "Success"
        mlflow.log_param("validation_status", status)

        for w in warnings:
            print(f"WARNING: {w}")
        for e in errors:
            print(f"ERROR: {e}")
        print(f"\nValidation status: {status}")
        print(json.dumps({"classes": len(reference), **totals}))

        # ให้ CI หยุดจริงเมื่อข้อมูลไม่ผ่าน
        if errors:
            raise SystemExit("Data validation failed — หยุด pipeline ไม่ให้ไปขั้นถัดไป")


if __name__ == "__main__":
    validate_data()
