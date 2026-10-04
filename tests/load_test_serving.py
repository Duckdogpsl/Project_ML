"""วัด Latency p50, p95 และ Throughput ของ Model Serving."""

import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

BASE_URL = os.getenv("SERVING_URL", "http://localhost:8000")
TOTAL_REQUESTS = int(os.getenv("TOTAL_REQUESTS", "50"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "5"))

TEST_IMAGE = Path(
    os.getenv(
        "TEST_IMAGE",
        r"C:\CP_LAB_MLOPS_Project\Project-ML\dataset\sample\test"
        r"\Tomato__Tomato_Healthy\TomatoHealthy(114).jpg",
    )
)

IMAGE_BYTES = TEST_IMAGE.read_bytes()


def send_prediction(request_id: int) -> dict:
    started = time.perf_counter()

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={
                "file": (
                    TEST_IMAGE.name,
                    IMAGE_BYTES,
                    "image/jpeg",
                )
            },
            timeout=60,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000

        return {
            "request_id": request_id,
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
            "success": response.status_code == 200,
        }

    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000

        return {
            "request_id": request_id,
            "status_code": None,
            "latency_ms": elapsed_ms,
            "success": False,
            "error": str(exc),
        }


def main():
    if not TEST_IMAGE.exists():
        raise SystemExit(f"ไม่พบภาพทดสอบ: {TEST_IMAGE}")

    print(
        f"Load test: requests={TOTAL_REQUESTS}, "
        f"concurrency={CONCURRENCY}"
    )

    # Warm-up เพื่อลดผลกระทบจากคำขอแรก
    warmup = send_prediction(-1)
    if not warmup["success"]:
        raise SystemExit(f"Warm-up ไม่ผ่าน: {warmup}")

    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [
            executor.submit(send_prediction, request_id)
            for request_id in range(TOTAL_REQUESTS)
        ]
        results = [future.result() for future in as_completed(futures)]

    total_seconds = time.perf_counter() - started
    successful = [item for item in results if item["success"]]
    failed = [item for item in results if not item["success"]]
    latencies = [item["latency_ms"] for item in successful]

    if not latencies:
        raise SystemExit("ไม่มีคำขอที่สำเร็จ")

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    throughput = len(successful) / total_seconds

    print("\n===== LOAD TEST RESULT =====")
    print(f"Total requests : {TOTAL_REQUESTS}")
    print(f"Successful     : {len(successful)}")
    print(f"Failed         : {len(failed)}")
    print(f"Concurrency    : {CONCURRENCY}")
    print(f"Total time     : {total_seconds:.3f} s")
    print(f"Latency mean   : {statistics.mean(latencies):.3f} ms")
    print(f"Latency p50    : {p50:.3f} ms")
    print(f"Latency p95    : {p95:.3f} ms")
    print(f"Throughput     : {throughput:.3f} requests/s")

    print("\n===== SLO CHECK =====")
    print(f"p50 <= 120 ms  : {'PASS' if p50 <= 120 else 'FAIL'}")
    print(f"p95 <= 300 ms  : {'PASS' if p95 <= 300 else 'FAIL'}")
    print(
        f"Throughput >= 50 RPS : "
        f"{'PASS' if throughput >= 50 else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
