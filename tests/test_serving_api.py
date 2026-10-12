"""Integration tests สำหรับ FastAPI Model Serving ที่รันอยู่ใน Docker."""

import os
from pathlib import Path

import pytest
import requests

BASE_URL = os.getenv("SERVING_URL", "http://localhost:8000")

ROOT = Path(__file__).resolve().parents[1]
_default_test_image = None
for candidate in [
    ROOT / "dataset/sample/test/Tomato__Tomato_Healthy/TomatoHealthy(114).jpg",
    ROOT / "dataset/tomato/test/Tomato__Tomato_Healthy/TomatoHealthy(114).jpg",
]:
    if candidate.exists():
        _default_test_image = candidate
        break
if not _default_test_image:
    _found = list((ROOT / "dataset").glob("*/*/*/*.[jJ][pP][gG]"))
    if _found:
        _default_test_image = _found[0]

_env_image = os.getenv("TEST_IMAGE")
TEST_IMAGE = Path(_env_image) if _env_image else (_default_test_image or Path("missing_image.jpg"))


def _is_server_available():
    try:
        res = requests.get(f"{BASE_URL}/health", timeout=2)
        return res.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_server_available(),
    reason=f"Serving server at {BASE_URL} is not running (integration test)",
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
