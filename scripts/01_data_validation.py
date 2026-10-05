import argparse
import hashlib
import io
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys
 
import mlflow
from PIL import Image
 
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ["train", "val", "test"]
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset" / "tomato"

MIN_CLASSES = 3                 # จำนวน 3 คลาส
MIN_IMAGES_PER_CLASS = 50       # ขอย่างน้อยกี่รูปต่อคลาสอยากได้มาเขียน
MAX_IMBALANCE_RATIO = 5.0       # เอาไว้เขียนแก้ imbalance ของคลาสมากสุดกัยน้อยสุด
MAX_CORRUPT_FILES = 0           # เอาไว้เขียนว่าจะเก็บไฟล์เสียไว้กี่รูป/ไฟล์
MAX_LEAKAGE_RATIO = 0.60         # สัดส่วนรูปใน val/test ที่ซ้ำกับ split อื่น (0 = ห้ามซ้ำเลย)
MAX_WITHIN_DUP_RATIO = 0.05     # รูปซ้ำภายใน split เดียวกัน เกินนี้แค่เตือน ไม่ทำให้ fail
 
 
def scan_split(split_dir: Path):
    class_counts = {}
    corrupt_files = []
    sizes, modes = Counter(), Counter()
    hashes = defaultdict(list)
 
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        n_ok = 0
        for f in sorted(class_dir.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            raw = f.read_bytes()
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    im.verify()                      # ตรวจว่าไฟล์ดูได้ไหม
                with Image.open(io.BytesIO(raw)) as im:  # verify() ใช้ซ้ำไม่ได้
                    sizes[f"{im.size[0]}x{im.size[1]}"] += 1
                    modes[im.mode] += 1
            except Exception:
                corrupt_files.append(str(f))
                continue
            hashes[hashlib.md5(raw).hexdigest()].append((class_dir.name, str(f)))
            n_ok += 1
        class_counts[class_dir.name] = n_ok
 
    return {
        "class_counts": class_counts,
        "corrupt_files": corrupt_files,
        "sizes": sizes,
        "modes": modes,
        "hashes": hashes,
    }
 
 
def validate_data(data_root: str):
    """
    ตรวจ dataset รูปใบมะเขือเทศที่แบ่งเป็น train/val/test
    แล้ว log ผลไปที่ MLflow
    """
    data_root = Path(data_root)
    missing = [s for s in SPLITS if not (data_root / s).is_dir()]
    if missing:
        raise SystemExit(f"ไม่พบโฟลเดอร์ {missing} ใน {data_root}")
 
    mlflow.set_experiment("Tomato Leaf Disease - Data Validation")
 
    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")
        mlflow.log_param("data_root", str(data_root))
 
        failures, warnings = [], []
        results = {}
 
        # ตรวจทีละ split
        for split in SPLITS:
            r = scan_split(data_root / split)
            results[split] = r
            cc = r["class_counts"]
            n_images = sum(cc.values())
            n_unique = len(r["hashes"])
            n_within_dup = n_images - n_unique
            min_c, max_c = (min(cc.values()), max(cc.values())) if cc else (0, 0)
            imbalance = max_c / min_c if min_c else float("inf")
 
            print(f"\n[{split}] {n_images} images, {len(cc)} classes")
            for name, n in cc.items():
                print(f"  - {name}: {n}")
            print(f"  corrupt: {len(r['corrupt_files'])}, "
                  f"duplicates within split: {n_within_dup}, "
                  f"imbalance: {imbalance:.2f}")
 
            mlflow.log_metric(f"{split}_num_images", n_images)
            mlflow.log_metric(f"{split}_num_unique_images", n_unique)
            mlflow.log_metric(f"{split}_num_classes", len(cc))
            mlflow.log_metric(f"{split}_num_corrupt", len(r["corrupt_files"]))
            mlflow.log_metric(f"{split}_within_dup", n_within_dup)
            mlflow.log_metric(f"{split}_imbalance_ratio", imbalance)
            for name, n in cc.items():
                mlflow.log_metric(f"{split}_count_{name}", n)
 
            if len(cc) < MIN_CLASSES:
                failures.append(f"{split}: มีแค่ {len(cc)} คลาส (ต้อง >= {MIN_CLASSES})")
            small = [n for n, c in cc.items() if c < MIN_IMAGES_PER_CLASS]
            if small:
                failures.append(f"{split}: คลาสที่รูปน้อยกว่า {MIN_IMAGES_PER_CLASS}: {small}")
            if imbalance > MAX_IMBALANCE_RATIO:
                failures.append(f"{split}: imbalance {imbalance:.2f} เกิน {MAX_IMBALANCE_RATIO}")
            if len(r["corrupt_files"]) > MAX_CORRUPT_FILES:
                failures.append(f"{split}: มีไฟล์เสีย {len(r['corrupt_files'])} ไฟล์")
            if n_images and n_within_dup / n_images > MAX_WITHIN_DUP_RATIO:
                warnings.append(f"{split}: มีรูปซ้ำในตัวเอง {n_within_dup} ไฟล์ "
                                f"({n_within_dup / n_images:.1%})")
 
        # 2.1 ชื่อคลาสต้องตรงกันทุก split
        class_sets = {s: set(results[s]["class_counts"]) for s in SPLITS}
        class_names = sorted(set.union(*class_sets.values()))
        mlflow.log_param("class_names", ",".join(class_names))
        mlflow.log_param("num_classes", len(class_names))
        for s in SPLITS:
            diff = set(class_names) - class_sets[s]
            if diff:
                failures.append(f"{s}: ขาดคลาส {sorted(diff)}")
 
        # 2.2 data leakage: รูปเดียวกันอยู่หลาย split
        hash_splits = defaultdict(set)
        for s in SPLITS:
            for h in results[s]["hashes"]:
                hash_splits[h].add(s)
 
        leakage_report = {}
        print("\nCross-split leakage (unique images shared):")
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            n = sum(1 for v in hash_splits.values() if a in v and b in v)
            leakage_report[f"{a}-{b}"] = n
            mlflow.log_metric(f"leak_{a}_{b}", n)
            print(f"  {a} ∩ {b}: {n}")
 
        for s in ["val", "test"]:
            uniq = results[s]["hashes"]
            leaked = sum(1 for h in uniq if len(hash_splits[h]) > 1)
            ratio = leaked / len(uniq) if uniq else 0.0
            mlflow.log_metric(f"{s}_leakage_ratio", ratio)
            print(f"  {s}: {leaked}/{len(uniq)} unique images also in another split ({ratio:.1%})")
            if ratio > MAX_LEAKAGE_RATIO:
                failures.append(f"{s}: {ratio:.1%} ของรูปซ้ำกับ split อื่น (data leakage)")
 
        # 2.3 รูปเดียวกันแต่ label ต่างกัน
        hash_labels = defaultdict(set)
        for s in SPLITS:
            for h, items in results[s]["hashes"].items():
                hash_labels[h].update(c for c, _ in items)
        conflicts = sum(1 for v in hash_labels.values() if len(v) > 1)
        mlflow.log_metric("label_conflicts", conflicts)
        if conflicts:
            failures.append(f"มีรูปเดียวกันแต่ label ต่างกัน {conflicts} รูป")
 
        leaked_examples = [
            {"splits": sorted(hash_splits[h]),
             "files": [p for s in SPLITS for _, p in results[s]["hashes"].get(h, [])]}
            for h, v in hash_splits.items() if len(v) > 1
        ]
        mlflow.log_dict({
            s: {
                "class_counts": results[s]["class_counts"],
                "corrupt_files": results[s]["corrupt_files"],
                "image_sizes": dict(results[s]["sizes"]),
                "color_modes": dict(results[s]["modes"]),
            } for s in SPLITS
        } | {"leakage": leakage_report, "label_conflicts": conflicts},
            "validation_report.json")
        mlflow.log_dict({"leaked_images": leaked_examples}, "leaked_images.json")
 
        # ---------- 4. สรุปผล ----------
        validation_status = "Failed" if failures else "Success"
        mlflow.log_param("validation_status", validation_status)
        if failures:
            mlflow.set_tag("validation_failures", " | ".join(failures)[:5000])
        if warnings:
            mlflow.set_tag("validation_warnings", " | ".join(warnings)[:5000])
 
        print(f"\nValidation status: {validation_status}")
        for w in warnings:
            print(f"  ! {w}")
        for f in failures:
            print(f"  ✗ {f}")
 
        if validation_status == "Failed":
            raise SystemExit("Data validation failed — หยุด pipeline ไม่ให้ไปขั้นถัดไป")
 
        print("Data validation run finished.")
 
 
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", str(DEFAULT_DATA_ROOT)),
        help="โฟลเดอร์ที่มี train/ val/ test/ (แต่ละอันมีโฟลเดอร์ย่อยแยกตามคลาส)",
    )
    args = parser.parse_args()
    validate_data(args.data_root)
