import argparse

import mlflow
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from common import (
    DATA_DIR,
    MODEL_ALIAS,
    MODEL_NAME,
    ROOT,
    extract_features,
    list_classes,
    list_images,
    setup_mlflow,
    short_label,
)

MONITORING_DATA = ROOT / "monitoring_data"


def image_summary(image, feature):
    """
    สร้าง feature สำหรับ Monitoring

    เราไม่ monitor HOG 2,276 มิติทั้งหมดโดยตรง
    แต่สรุปเป็นค่าที่อธิบาย distribution ของภาพได้ง่ายกว่า
    """

    rgb = image.convert("RGB").resize((64, 64))

    rgb_arr = np.asarray(rgb, dtype=np.float32) / 255.0
    hsv_arr = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0

    return {
        "brightness_mean": float(rgb_arr.mean()),
        "contrast_std": float(rgb_arr.std()),

        "red_mean": float(rgb_arr[..., 0].mean()),
        "green_mean": float(rgb_arr[..., 1].mean()),
        "blue_mean": float(rgb_arr[..., 2].mean()),

        "saturation_mean": float(hsv_arr[..., 1].mean()),

        "feature_mean": float(feature.mean()),
        "feature_std": float(feature.std()),
        "feature_l2": float(np.linalg.norm(feature)),
    }


def collect_split(model, split, simulate_drift=False):

    split_dir = DATA_DIR / split

    rows = []

    print(f"\nProcessing monitoring data: {split}")

    for class_name in list_classes(split_dir):

        class_dir = split_dir / class_name
        true_label = short_label(class_name)

        images = list_images(class_dir)

        print(f"  {class_name}: {len(images)} images")

        for image_path in images:

            with Image.open(image_path) as raw:

                image = raw.convert("RGB")

                # ใช้สำหรับ DEMO เท่านั้น
                # จำลอง distribution shift เช่น
                # แสงเปลี่ยน + ภาพเบลอ
                if simulate_drift:
                    image = ImageEnhance.Brightness(
                        image
                    ).enhance(0.55)

                    image = image.filter(
                        ImageFilter.GaussianBlur(radius=1.2)
                    )

                feature = extract_features(image)

            X = np.expand_dims(feature, axis=0)

            prediction = str(
                model.predict(X)[0]
            )

            confidence = np.nan

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(X)[0]

                confidence = float(
                    np.max(probabilities)
                )

            row = {
                "filename": image_path.name,
                "y_true": true_label,
                "y_pred": prediction,
                "confidence": confidence,

                **image_summary(
                    image,
                    feature,
                ),
            }

            rows.append(row)

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help="จำลอง current data ให้มืดและเบลอเพื่อทดสอบ drift detector",
    )

    args = parser.parse_args()

    MONITORING_DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    # เชื่อม MLflow
    setup_mlflow()

    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    print("Loading model:", model_uri)

    model = mlflow.sklearn.load_model(
        model_uri
    )

    print("Model loaded.")

    # Reference = validation data
    reference = collect_split(
        model,
        "val",
        simulate_drift=False,
    )

    # Current = test data
    current = collect_split(
        model,
        "test",
        simulate_drift=args.simulate_drift,
    )

    reference_path = (
        MONITORING_DATA / "reference.csv"
    )

    current_path = (
        MONITORING_DATA / "current.csv"
    )

    reference.to_csv(
        reference_path,
        index=False,
    )

    current.to_csv(
        current_path,
        index=False,
    )

    print("\n===== RESULT =====")

    print(
        "Reference rows:",
        len(reference),
    )

    print(
        "Current rows:",
        len(current),
    )

    print(
        "Simulated drift:",
        args.simulate_drift,
    )

    print(
        "\nReference:",
        reference_path,
    )

    print(
        "Current:",
        current_path,
    )

    print("\nReference preview:")
    print(reference.head())

    print("\nCurrent preview:")
    print(current.head())


if __name__ == "__main__":
    main()
