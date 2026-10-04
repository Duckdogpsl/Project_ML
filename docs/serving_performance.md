# Model Serving Performance Report

## Test Environment

- Service: FastAPI Model Serving
- Container image: `tomato-serving:1.0`
- Model source: MLflow Model Registry
- Model alias: `champion`
- Model type: Scikit-learn Random Forest
- Feature extraction: HSV histogram and HOG
- Test endpoint: `POST /predict`

## Declared SLO

- Latency p50 <= 120 ms
- Latency p95 <= 300 ms
- Throughput >= 50 requests/second

## Benchmark Results

| Concurrency | Requests | Successful | Failed | p50 (ms) | p95 (ms) | Throughput (RPS) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 20 | 0 | 272.863 | 288.978 | 3.656 |
| 2 | 20 | 20 | 0 | 529.805 | 553.902 | 3.773 |
| 5 | 50 | 50 | 0 | 1300.358 | 1336.159 | 3.837 |

## SLO Evaluation

| Metric | Target | Current baseline | Result |
|---|---:|---:|---|
| Latency p50 | <= 120 ms | 272.863 ms at concurrency 1 | FAIL |
| Latency p95 | <= 300 ms | 288.978 ms at concurrency 1 | PASS |
| Throughput | >= 50 RPS | 3.837 RPS at concurrency 5 | FAIL |

## Findings

1. ทุกคำขอสำเร็จ ไม่มี request failure ระหว่างการทดสอบ
2. Throughput คงที่ประมาณ 3.7-3.8 RPS แม้เพิ่ม concurrency
3. Latency เพิ่มขึ้นตาม concurrency แสดงว่าคำขอเกิดการรอคิว
4. การแปลงภาพและสกัด HSV/HOG เป็นงานที่ใช้ CPU
5. ระบบปัจจุบันใช้เป็น Serving Baseline และยังไม่ผ่าน Production SLO ทุกข้อ

## Improvement Plan

1. วัดเวลา feature extraction และ model prediction แยกกัน
2. ทดลองเพิ่ม Uvicorn workers และวัดผลซ้ำ
3. ลดต้นทุนการสกัด HOG หรือใช้โมเดลที่รองรับ inference ได้เร็วขึ้น
4. ทดสอบการใช้ ONNX Runtime ตามสถาปัตยกรรมใน AI Project Canvas
5. ปรับทรัพยากร CPU และทดสอบบนสภาพแวดล้อม deployment จริง
6. ไม่อนุมัติ Production deployment จนกว่าจะผ่าน Quality Gate และ Performance SLO
