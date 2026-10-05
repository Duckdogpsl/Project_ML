# Project_ML: Tomato Leaf Disease Classification

ระบบจำแนกโรคใบมะเขือเทศจากภาพ โดยโหลดโมเดลจาก MLflow Model Registry และให้บริการผ่าน FastAPI ภายใน Docker Container

ส่วน Model Serving รองรับการตรวจสุขภาพระบบ การทำนายโรค การเก็บ Logs และ Prometheus Metrics รวมถึงมี Integration Test และ Load Test สำหรับตรวจสอบระบบก่อนส่งต่อให้ทีม Monitoring และ CI/CD

---

## ภาพรวมการทำงาน

```text
ภาพใบมะเขือเทศ
        |
        v
FastAPI /predict
        |
        v
ตรวจไฟล์และเตรียมภาพ
        |
        v
Resize 64x64 + HSV Histogram + HOG
        |
        v
MLflow Model Registry
tomato-leaf-classifier@champion
        |
        v
ผลทำนายแบบ JSON
```

ระบบใช้ฟังก์ชันสกัดฟีเจอร์จาก `scripts/common.py` ชุดเดียวกันทั้งตอน Training และ Serving เพื่อป้องกัน Training-Serving Skew

---
## เหตุผลที่เลือก Real-time REST API

ระบบเลือกให้บริการแบบ Real-time ผ่าน REST API เพราะผู้ใช้ต้องส่งภาพใบมะเขือเทศและรับผลการจำแนกกลับทันที เพื่อใช้ประกอบการตัดสินใจในพื้นที่เพาะปลูก

เลือกใช้ FastAPI เพราะรองรับการรับไฟล์ภาพ มีการตรวจสอบ Request สร้างเอกสาร API ผ่าน Swagger UI อัตโนมัติ และเชื่อมต่อ Health Check, Logs และ Prometheus Metrics ได้สะดวก

เลือกใช้ Docker เพื่อทำให้ Source Code, Dependencies และคำสั่งเปิด API อยู่ในสภาพแวดล้อมเดียวกัน ช่วยลดปัญหาความแตกต่างระหว่างเครื่องของผู้พัฒนาและเครื่องที่นำระบบไปรัน

ระบบยังไม่เลือก Batch Serving เป็นรูปแบบหลัก เพราะโจทย์ต้องตอบผลทีละภาพแบบทันที แต่สามารถเพิ่ม Batch Processing ภายหลังสำหรับประมวลผลภาพจำนวนมากจากหลายแปลงได้

## เทคโนโลยีหลัก

- FastAPI และ Uvicorn
- MLflow Model Registry
- Scikit-learn
- Pillow และ Scikit-image
- Docker
- Prometheus Client
- Pytest และ Requests

---

## API Endpoints

| Method | Endpoint | หน้าที่ |
|---|---|---|
| GET | `/` | แสดงข้อมูลพื้นฐานของ Service |
| GET | `/health` | ตรวจสุขภาพ API และสถานะโมเดล |
| POST | `/predict` | รับภาพและทำนายโรค |
| GET | `/metrics/` | ส่งออก Prometheus Metrics |
| GET | `/docs` | เปิด Swagger UI สำหรับทดสอบ API |

### ตัวอย่างผลจาก `/predict`

```json
{
  "filename": "TomatoHealthy(114).jpg",
  "disease_class": "Healthy",
  "confidence_score": 0.33,
  "inference_latency_ms": 295.134,
  "model_name": "tomato-leaf-classifier",
  "model_alias": "champion"
}
```

API รองรับไฟล์ JPEG และ PNG หากส่งไฟล์ประเภทอื่น ระบบจะตอบ HTTP `415 Unsupported Media Type`

---

## โครงสร้างไฟล์สำคัญ

```text
Project_ML/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
├── scripts/
│   ├── common.py
│   ├── 05_load_and_predict.py
│   └── 05_model_serving.py
├── tests/
│   ├── test_serving_api.py
│   └── load_test_serving.py
└── docs/
    └── serving_performance.md
```

- `common.py` เก็บค่าตั้งต้นและฟังก์ชันสกัดฟีเจอร์ร่วม
- `05_load_and_predict.py` ทดสอบโหลดโมเดลและทำนายผ่าน Command Line
- `05_model_serving.py` เป็น FastAPI Application
- `test_serving_api.py` ทดสอบ API ผ่าน HTTP
- `load_test_serving.py` วัด Latency และ Throughput
- `serving_performance.md` บันทึกผล Performance Test

---

## สิ่งที่ต้องเตรียม

ติดตั้ง Docker Desktop และเตรียม MLflow Runtime Artifacts ไว้ที่ Root ของ Repository:

```text
mlflow.db
mlruns/
```

MLflow Registry ต้องมีโมเดล:

```text
Model name: tomato-leaf-classifier
Alias: champion
```

`mlflow.db` และ `mlruns/` ไม่ถูก Commit ลง Git ผู้ใช้งานต้องสร้างผ่าน Training Pipeline หรือรับจากทีม Training และ Model Registry

---

## Build Docker Image

```powershell
docker build -t tomato-serving:1.0 .
```

---

## Run Model Serving

```powershell
docker run -d --name tomato-serving `
  -p 8000:8000 `
  -v "${PWD}\mlflow.db:/app/mlflow.db" `
  -v "${PWD}\mlruns:/app/mlruns" `
  tomato-serving:1.0
```

ตรวจสถานะ:

```powershell
docker ps --filter "name=tomato-serving"
docker logs --tail 30 tomato-serving
```

ระบบพร้อมใช้งานเมื่อ Container แสดง:

```text
Up ... (healthy)
```

---

## ทดสอบ Health Check

```powershell
Invoke-RestMethod "http://localhost:8000/health" |
ConvertTo-Json
```

ผลที่คาดหวัง:

```json
{
  "status": "healthy",
  "model_name": "tomato-leaf-classifier",
  "model_alias": "champion",
  "model_uri": "models:/tomato-leaf-classifier@champion"
}
```

---

## ส่งภาพเพื่อทำนาย

```powershell
$image = "C:\path\to\tomato-leaf.jpg"

curl.exe -X POST `
  -F "file=@$image;type=image/jpeg" `
  "http://localhost:8000/predict"
```

สำหรับไฟล์ PNG ให้ใช้:

```text
type=image/png
```

สามารถทดสอบ API ผ่าน Swagger UI ได้ที่:

```text
http://localhost:8000/docs
```

---

## Prometheus Metrics

เรียกดู Metrics:

```powershell
$metrics = Invoke-WebRequest "http://localhost:8000/metrics/"
$metrics.Content | Select-String "tomato_serving_"
```

Metrics ที่ส่งออก:

```text
tomato_serving_requests_total
tomato_serving_predictions_total
tomato_serving_errors_total
tomato_serving_inference_latency_seconds
```

ทีม Monitoring สามารถใช้ Metrics เหล่านี้ติดตามจำนวนคำขอ ผลการทำนาย Error และ Latency

---

## Integration Tests

Container ต้องมีสถานะ `healthy` ก่อนรัน:

```powershell
python -m pytest .\tests\test_serving_api.py -v
```

กรณีที่ทดสอบ:

1. Health Check
2. Prediction
3. การปฏิเสธไฟล์ผิดประเภท
4. Prometheus Metrics

ผลล่าสุด:

```text
4 passed
```

---

## Performance Load Test

```powershell
$env:TOTAL_REQUESTS="50"
$env:CONCURRENCY="5"

python .\tests\load_test_serving.py
```

สคริปต์รายงาน:

- จำนวนคำขอสำเร็จและล้มเหลว
- Latency เฉลี่ย
- Latency p50 และ p95
- Throughput
- ผลเปรียบเทียบกับ SLO

### SLO ที่ประกาศ

| Metric | เป้าหมาย |
|---|---:|
| Latency p50 | ≤ 120 ms |
| Latency p95 | ≤ 300 ms |
| Throughput | ≥ 50 RPS |

### ผล Baseline

| Concurrency | Requests | สำเร็จ | ล้มเหลว | p50 | p95 | Throughput |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 20 | 0 | 272.863 ms | 288.978 ms | 3.656 RPS |
| 2 | 20 | 20 | 0 | 529.805 ms | 553.902 ms | 3.773 RPS |
| 5 | 50 | 50 | 0 | 1300.358 ms | 1336.159 ms | 3.837 RPS |

ทุกคำขอสำเร็จ แต่ระบบยังไม่ผ่าน Production SLO ทุกข้อ เมื่อ Concurrency เพิ่ม Latency จะเพิ่มขึ้น ขณะที่ Throughput คงที่ประมาณ 3.7 ถึง 3.8 RPS

รายละเอียดอยู่ที่ `docs/serving_performance.md`

---

## สถานะ Quality Gate

ระบบ Serving สามารถรันและรับคำขอได้จริง แต่โมเดลเวอร์ชันปัจจุบันยังไม่อนุมัติสำหรับ Production เนื่องจากยังไม่ผ่าน Quality Gate และ Performance SLO ทุกข้อ

### Model Quality

ผลของโมเดล Baseline ที่ลงทะเบียนเป็น `champion`:

| Metric | ผลปัจจุบัน | เป้าหมาย | สถานะ |
|---|---:|---:|---|
| Test Accuracy | 0.7755 | ใช้ประกอบการประเมิน | Baseline |
| Test Macro F1 | 0.7659 | ≥ 0.93 | ไม่ผ่าน |
| Late Blight Recall | 0.4000 | ≥ 0.96 | ไม่ผ่าน |
| Yellow Leaf Curl Virus Recall | 0.9000 | ≥ 0.96 | ไม่ผ่าน |

### Serving Performance

| Metric | ผล Baseline | SLO | สถานะ |
|---|---:|---:|---|
| Latency p50 | 272.863 ms | ≤ 120 ms | ไม่ผ่าน |
| Latency p95 | 288.978 ms | ≤ 300 ms | ผ่านที่ Concurrency 1 |
| Throughput | 3.837 RPS | ≥ 50 RPS | ไม่ผ่าน |

โมเดลเวอร์ชันนี้จึงใช้เป็น Serving Baseline สำหรับสาธิต API, Docker, MLflow Registry, Metrics, Logs และ Tests เท่านั้น

การนำขึ้น Production ต้องมีเงื่อนไขดังนี้:

1. โมเดลผ่าน Model Quality Gate
2. ระบบผ่าน Latency และ Throughput SLO
3. Integration Tests ผ่านทั้งหมด
4. Docker Health Check แสดงสถานะ `healthy`
5. โมเดลที่ผ่านการอนุมัติถูกกำหนด Alias เป็น `champion`
6. หากโมเดลใหม่มีคุณภาพต่ำกว่าเดิม ต้องเปลี่ยน Alias กลับไปยังเวอร์ชันก่อนหน้า

---


## ข้อจำกัดปัจจุบัน

1. โมเดลเป็น Serving Baseline และยังไม่ผ่าน Model Quality Target ทุกข้อ
2. การสกัด HSV Histogram และ HOG ใช้ CPU และมีผลต่อ Latency
3. Performance ยังไม่ผ่าน p50 และ Throughput SLO
4. ระบบต้องได้รับ `mlflow.db` และ `mlruns/` จาก Training Pipeline
5. Confidence Score ของบางภาพยังต่ำ
6. ยังไม่ได้ปรับแต่งสำหรับคำขอพร้อมกันจำนวนมาก

### แนวทางปรับปรุง

- วัดเวลา Feature Extraction และ Model Prediction แยกกัน
- ทดลองเพิ่ม Uvicorn Workers
- ลดต้นทุนการสกัด HOG
- ใช้ Dataset เต็มแทน Sample Dataset
- ทดลอง Transfer Learning หรือ ONNX Runtime
- ทดสอบบน Server ที่ใกล้เคียง Production
- อนุมัติ Deployment เฉพาะโมเดลที่ผ่าน Quality Gate และ SLO

---

## ส่งต่อทีม Monitoring

ใช้ Endpoint:

```text
GET /health
GET /metrics/
```

Metrics หลัก:

```text
tomato_serving_requests_total
tomato_serving_predictions_total
tomato_serving_errors_total
tomato_serving_inference_latency_seconds
```

ทีม Monitoring สามารถนำไปสร้าง Dashboard และ Alert สำหรับ Service Status, Request Count, Error Rate, Latency และ Prediction Distribution

---

## ส่งต่อทีม CI/CD

Pipeline ควรมีขั้นตอน:

1. ตรวจคุณภาพโค้ดด้วย Ruff
2. รัน Pytest
3. Build Docker Image
4. เปิด Container พร้อม MLflow Artifacts
5. รอ Docker Health Check
6. ทดสอบ `/health`
7. รัน Serving Integration Tests
8. ตรวจ Model Quality Gate
9. Deploy เฉพาะเวอร์ชันที่ผ่านเกณฑ์

---
## การทำซ้ำและข้อจำกัดด้าน Environment

ระบบใช้ Docker เพื่อให้สภาพแวดล้อมของ Model Serving ทำซ้ำได้ โดย Docker Image จะติดตั้ง Dependencies จาก `requirements.txt` และเปิด FastAPI ด้วยคำสั่งเดียวกันทุกเครื่อง

อย่างไรก็ตาม `requirements.txt` ปัจจุบันกำหนดเวอร์ชันขั้นต่ำด้วย `>=` จึงยังไม่รับประกันว่าจะติดตั้ง Minor Version เดียวกันทุกครั้ง ก่อนส่งระบบฉบับสมบูรณ์ควรสร้างไฟล์ล็อกเวอร์ชัน เช่น:

```powershell
python -m pip freeze > requirements-lock.txt
```

ไฟล์ต่อไปนี้ไม่บันทึกลง Git:

```text
.venv/
mlflow.db
mlruns/
processed_data/
```

ไฟล์เหล่านี้เป็น Local Environment หรือ Runtime Artifacts โดย `mlflow.db` และ `mlruns/` ต้องสร้างจาก Training Pipeline หรือได้รับจากทีม Training ก่อนเปิด Model Serving

---

## หยุดระบบ

```powershell
docker stop tomato-serving
docker rm tomato-serving
```

คำสั่งนี้ลบเฉพาะ Container ไม่ลบ Docker Image, MLflow Database หรือ Model Artifacts

---

## สถานะปัจจุบัน

```text
MLflow champion loading    พร้อมใช้งาน
FastAPI /health            พร้อมใช้งาน
FastAPI /predict           พร้อมใช้งาน
Prometheus /metrics        พร้อมใช้งาน
Application logs           พร้อมใช้งาน
Docker image               พร้อมใช้งาน
Docker health check        พร้อมใช้งาน
Integration tests          ผ่าน 4 tests
Performance benchmark      ดำเนินการแล้ว
Monitoring handover        พร้อมส่งต่อ
CI/CD handover             พร้อมส่งต่อ
Production SLO             ยังไม่ผ่านทุกข้อ
```