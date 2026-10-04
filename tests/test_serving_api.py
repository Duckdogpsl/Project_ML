"""Integration tests สำหรับ FastAPI Model Serving ที่รันอยู่ใน Docker."""

import os
from pathlib import Path

import requests

BASE_URL = os.getenv("SERVING_URL", "http://localhost:8000")
TEST_IMAGE = Path(
    os.getenv(
        "TEST_IMAGE",
        r"C:\CP_LAB_MLOPS_Project\Project-ML\dataset\sample\test"
        r"\Tomato__Tomato_Healthy\TomatoHealthy(114).jpg",
    )
)


def test_health_endpoint():
    response = requests.get(f"{BASE_URL}/health", timeout=10)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_name"] == "tomato-leaf-classifier"
    assert body["model_alias"] == "champion"


def test_predict_endpoint():
    assert TEST_IMAGE.exists(), f"ไม่พบภาพทดสอบ: {TEST_IMAGE}"

    with TEST_IMAGE.open("rb") as image_file:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={
                "file": (
                    TEST_IMAGE.name,
                    image_file,
                    "image/jpeg",
                )
            },
            timeout=30,
        )

    assert response.status_code == 200
    body = response.json()

    assert body["filename"] == TEST_IMAGE.name
    assert isinstance(body["disease_class"], str)
    assert body["disease_class"]
    assert 0.0 <= body["confidence_score"] <= 1.0
    assert body["inference_latency_ms"] >= 0.0
    assert body["model_name"] == "tomato-leaf-classifier"
    assert body["model_alias"] == "champion"


def test_reject_unsupported_file_type():
    response = requests.post(
        f"{BASE_URL}/predict",
        files={
            "file": (
                "invalid.txt",
                b"not an image",
                "text/plain",
            )
        },
        timeout=10,
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "รองรับเฉพาะไฟล์ JPEG และ PNG"


def test_metrics_endpoint():
    response = requests.get(f"{BASE_URL}/metrics/", timeout=10)

    assert response.status_code == 200
    assert "tomato_serving_requests_total" in response.text
    assert "tomato_serving_predictions_total" in response.text
    assert "tomato_serving_errors_total" in response.text
    assert "tomato_serving_inference_latency_seconds" in response.text
