import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "dataset" / "tomato"
TARGET_DIR = PROJECT_ROOT / "dataset" / "sample"

SPLITS = ["train", "val", "test"]
NUM_SAMPLES = {"train": 1500, "val": 300, "test": 300}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def make_sample():
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        return

    for split in SPLITS:
        split_src = SOURCE_DIR / split
        if not split_src.exists():
            print(f"Warning: Split directory '{split_src}' not found. Skipping...")
            continue

        for class_dir in sorted(split_src.iterdir()):
            if not class_dir.is_dir():
                continue

            images = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
            if not images:
                continue

            sampled_images = random.sample(images, min(NUM_SAMPLES[split], len(images)))

            target_class_dir = TARGET_DIR / split / class_dir.name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            for img_path in sampled_images:
                shutil.copy2(img_path, target_class_dir / img_path.name)

            print(f"[{split}/{class_dir.name}] Copied {len(sampled_images)} images.")

    print(f"\nSample dataset successfully created at '{TARGET_DIR}'")


if __name__ == "__main__":
    random.seed(42)  # สุ่มผลลัพธ์เดิมทุกครั้งเพื่อให้สอดคล้องกัน
    make_sample()