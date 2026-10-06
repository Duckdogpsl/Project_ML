import argparse
import os

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

MONITORING_DATA = ROOT / os.getenv("MONITORING_DATA_DIR", "monitoring_data")


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


def collect_split(model, split, simulate_drift=False, brightness_factor=0.55, blur_radius=1.2):

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
                    ).enhance(brightness_factor)

                    image = image.filter(
                        ImageFilter.GaussianBlur(radius=blur_radius)
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

    parser.add_argument("--brightness-factor", type=float, default=0.55)
    parser.add_argument("--blur-radius", type=float, default=1.2)
    args = parser.parse_args()
    if not 0 < args.brightness_factor <= 1 or args.blur_radius < 0:
        parser.error("brightness must be in (0, 1] and blur radius nonnegative")

    MONITORING_DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    # เชื่อม MLflow
    setup_mlflow()

    model_uri = (
        f"models:/{MODEL_NAME}/{os.environ['MODEL_VERSION']}"
        if os.getenv("MODEL_VERSION") else f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
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
        brightness_factor=args.brightness_factor,
        blur_radius=args.blur_radius,
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
