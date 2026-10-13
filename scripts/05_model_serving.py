"""FastAPI Model Serving สำหรับจำแนกโรคใบมะเขือเทศ."""

import io
import logging
import os
from pathlib import Path
import secrets
import signal
import time

import mlflow
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Histogram, make_asgi_app

from common import (
    MODEL_ALIAS,
    MODEL_NAME,
    ROOT,
    extract_features,
    setup_mlflow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("tomato-serving")

MODEL_VERSION = os.getenv("MODEL_VERSION")
MODEL_URI = (f"models:/{MODEL_NAME}/{MODEL_VERSION}" if MODEL_VERSION
             else f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
REQUEST_COUNT = Counter(
    "tomato_serving_requests_total",
    "จำนวนคำขอที่เข้ามายัง Model Serving",
    ["endpoint", "status"],
)

PREDICTION_COUNT = Counter(
    "tomato_serving_predictions_total",
    "จำนวนผลการทำนายแยกตามคลาส",
    ["disease_class"],
)

ERROR_COUNT = Counter(
    "tomato_serving_errors_total",
    "จำนวนข้อผิดพลาดระหว่างให้บริการ",
    ["error_type"],
)

INFERENCE_LATENCY = Histogram(
    "tomato_serving_inference_latency_seconds",
    "เวลาที่ใช้ประมวลผลคำขอทำนาย หน่วยวินาที",
    buckets=(0.05, 0.1, 0.12, 0.2, 0.3, 0.5, 1.0, 2.0),
)

# เกณฑ์ความเชื่อมั่นขั้นต่ำสำหรับ Out-of-Distribution / Unknown detection
# หากภาพได้ความเชื่อมั่นต่ำกว่าเกณฑ์นี้ จะไม่ฟันธงว่าเป็นโรคใบมะเขือเทศ แต่จะตอบเป็น Unknown
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
MAX_DOMAIN_DISTANCE = float(os.getenv("MAX_DOMAIN_DISTANCE", "5.40"))

CENTROIDS_PATH = Path(__file__).resolve().parent / "class_centroids.npz"
class_centroids = None
if CENTROIDS_PATH.exists():
    try:
        _loaded = np.load(CENTROIDS_PATH)
        class_centroids = {k: _loaded[k] for k in _loaded.files}
        logger.info("Loaded class centroids for domain distance check from %s", CENTROIDS_PATH)
    except Exception as exc:
        logger.warning("Could not load class centroids: %s", exc)

setup_mlflow()
logger.info("Loading model from %s", MODEL_URI)
model = mlflow.sklearn.load_model(MODEL_URI)
logger.info("Model loaded successfully")

app = FastAPI(
    title="Tomato Leaf Disease Model Serving",
    description="API สำหรับจำแนกโรคใบมะเขือเทศด้วย MLflow champion model",
    version="1.0.0",
)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/samples/{sample_name}")
def get_sample(sample_name: str):
    samples = {
        "healthy": ROOT / "dataset/tomato/test/Tomato__Tomato_Healthy/TomatoHealthy(1005).jpg",
        "mosaic": ROOT / "dataset/tomato/test/Tomato__Tomato_Mosaic_virus/TomatoMosaicVirus(1006).jpg",
        "yellow_curl": ROOT / "dataset/tomato/test/Tomato__Tomato_Yellow_Leaf_Curl_Virus/TomatoYellowCurlVirus(1002).jpg",
    }
    p = samples.get(sample_name)
    if p and p.exists():
        return FileResponse(p, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Sample image not found")


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def web_ui():
    return """<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🍅 Tomato Leaf AI Diagnostic & OOD Detection</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --primary: #3b82f6;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg); color: var(--text); padding: 24px 16px; min-height: 100vh; display: flex; justify-content: center; }
  .container { width: 100%; max-width: 800px; }
  header { text-align: center; margin-bottom: 24px; }
  h1 { font-size: 1.8rem; margin-bottom: 8px; color: #fff; }
  .subtitle { color: var(--muted); font-size: 0.95rem; }
  .nav-links { margin-top: 12px; display: flex; justify-content: center; gap: 16px; }
  .nav-links a { color: var(--primary); text-decoration: none; font-size: 0.85rem; padding: 4px 10px; border-radius: 6px; background: rgba(59,130,246,0.1); }
  .nav-links a:hover { background: rgba(59,130,246,0.2); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
  .section-title { font-size: 1rem; font-weight: 600; margin-bottom: 12px; color: #e2e8f0; display: flex; align-items: center; gap: 8px; }
  
  .sample-buttons { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }
  .btn-sample { background: #334155; color: #f1f5f9; border: 1px solid #475569; padding: 8px 14px; border-radius: 8px; cursor: pointer; font-size: 0.85rem; transition: all 0.2s; }
  .btn-sample:hover { background: #475569; border-color: var(--primary); }
  
  .drop-zone { border: 2px dashed #475569; border-radius: 10px; padding: 32px 16px; text-align: center; cursor: pointer; transition: 0.2s; background: rgba(15,23,42,0.4); }
  .drop-zone:hover { border-color: var(--primary); background: rgba(59,130,246,0.05); }
  .drop-zone p { color: var(--muted); font-size: 0.9rem; margin-top: 8px; }
  
  .preview-container { display: none; margin-top: 16px; text-align: center; }
  .preview-img { max-width: 240px; max-height: 240px; border-radius: 8px; border: 2px solid var(--border); object-fit: contain; }
  
  .btn-predict { width: 100%; margin-top: 16px; background: #2563eb; color: #fff; border: none; padding: 12px; border-radius: 8px; font-weight: 600; font-size: 1rem; cursor: pointer; transition: 0.2s; }
  .btn-predict:hover { background: #1d4ed8; }
  .btn-predict:disabled { background: #475569; cursor: not-allowed; }
  
  .result-card { display: none; margin-top: 20px; border-left: 4px solid var(--primary); }
  .badge { display: inline-block; padding: 6px 14px; border-radius: 9999px; font-weight: 600; font-size: 0.95rem; margin-bottom: 12px; }
  .badge-success { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid #059669; }
  .badge-ood { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid #dc2626; }
  .badge-mosaic { background: rgba(168,85,247,0.2); color: #c084fc; border: 1px solid #9333ea; }
  
  .progress-bg { background: #334155; border-radius: 9999px; height: 10px; overflow: hidden; margin: 8px 0; }
  .progress-bar { height: 100%; transition: width 0.4s; }
  
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 14px; }
  .stat-box { background: rgba(15,23,42,0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border); }
  .stat-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-value { font-size: 0.95rem; font-weight: 600; margin-top: 2px; }
  
  .message-box { padding: 12px; border-radius: 8px; margin-top: 14px; font-size: 0.9rem; line-height: 1.4; }
  .msg-ood { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); color: #fca5a5; }
  .msg-ok { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); color: #6ee7b7; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>🍅 Tomato Leaf Disease AI Diagnostic</h1>
    <p class="subtitle">ระบบตรวจวินิจฉัยโรคใบมะเขือเทศ & Out-of-Distribution Detection</p>
    <div class="nav-links">
      <a href="/docs" target="_blank">📖 Swagger API Docs</a>
      <a href="/health" target="_blank">🩺 Health Status</a>
      <a href="/metrics" target="_blank">📊 Prometheus Metrics</a>
    </div>
  </header>

  <div class="card">
    <div class="section-title">✨ ภาพตัวอย่าง 3 คลาสโรคใบมะเขือเทศ (3 Tomato Leaf Classes)</div>
    <div class="sample-buttons">
      <button class="btn-sample" onclick="loadSample('healthy')">🍀 ใบปกติ (Healthy)</button>
      <button class="btn-sample" onclick="loadSample('mosaic')">🍂 ไวรัสโมเสก (Mosaic Virus)</button>
      <button class="btn-sample" onclick="loadSample('yellow_curl')">🟡 ไวรัสใบหงิกเหลือง (Yellow Leaf Curl)</button>
    </div>

    <div class="section-title">📁 หรืออัปโหลดภาพจากเครื่องของคุณ (Any Image File)</div>
    <div class="drop-zone" onclick="document.getElementById('fileInput').click()">
      <div style="font-size: 2.2rem;">📷</div>
      <p><strong>คลิกเพื่อเลือกไฟล์</strong> หรือลากรูปภาพมาวางที่นี่ (JPEG / PNG)</p>
      <input type="file" id="fileInput" accept="image/jpeg,image/png" style="display:none;" onchange="handleFile(this.files[0])">
    </div>

    <div class="preview-container" id="previewContainer">
      <img id="previewImg" class="preview-img" alt="Selected Image">
      <p id="fileNameText" style="font-size: 0.85rem; color: var(--muted); margin-top: 6px;"></p>
    </div>

    <button id="predictBtn" class="btn-predict" onclick="submitPredict()" disabled>🔍 ตรวจสอบภาพ (Analyze)</button>
  </div>

  <div class="card result-card" id="resultCard">
    <div class="section-title">📋 ผลการวินิจฉัยจากโมเดล (Diagnostic Result)</div>
    <div id="badgeContainer"></div>

    <div style="margin: 12px 0;">
      <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
        <span>ความเชื่อมั่น (Confidence Score):</span>
        <strong id="confText">-</strong>
      </div>
      <div class="progress-bg">
        <div id="confBar" class="progress-bar" style="width: 0%;"></div>
      </div>
    </div>

    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-label">Domain Status</div>
        <div class="stat-value" id="domainStatusText">-</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Inference Latency</div>
        <div class="stat-value" id="latencyText">-</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Model Alias</div>
        <div class="stat-value" id="modelAliasText">-</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Action / คำแนะนำ</div>
        <div class="stat-value" id="actionText">-</div>
      </div>
    </div>

    <div id="messageBox" class="message-box"></div>
  </div>
</div>

<script>
  let selectedFile = null;

  async function loadSample(type) {
    try {
      const res = await fetch(`/samples/${type}`);
      if (!res.ok) throw new Error("ไม่สามารถโหลดภาพตัวอย่างได้");
      const blob = await res.blob();
      const file = new File([blob], `sample_${type}.jpg`, { type: "image/jpeg" });
      handleFile(file);
    } catch (err) {
      alert(err.message);
    }
  }

  function handleFile(file) {
    if (!file) return;
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      document.getElementById('previewImg').src = e.target.result;
      document.getElementById('fileNameText').textContent = `${file.name} (${(file.size/1024).toFixed(1)} KB)`;
      document.getElementById('previewContainer').style.display = 'block';
      document.getElementById('predictBtn').disabled = false;
      document.getElementById('resultCard').style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  async function submitPredict() {
    if (!selectedFile) return;
    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
    btn.textContent = "⏳ กำลังประมวลผล...";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      btn.disabled = false;
      btn.textContent = "🔍 ตรวจสอบภาพ (Analyze)";

      if (!res.ok) {
        alert(data.detail || "เกิดข้อผิดพลาดในการทำนาย");
        return;
      }
      displayResult(data);
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "🔍 ตรวจสอบภาพ (Analyze)";
      alert("Error: " + err.message);
    }
  }

  function displayResult(data) {
    const card = document.getElementById('resultCard');
    const badge = document.getElementById('badgeContainer');
    const conf = (data.confidence_score * 100).toFixed(1);
    const isOOD = data.is_out_of_distribution;

    card.style.display = 'block';
    card.scrollIntoView({ behavior: 'smooth' });

    if (isOOD) {
      badge.innerHTML = `<span class="badge badge-ood">🚫 ไม่รู้จัก (ไม่อยู่ใน 3 คลาสของระบบ)</span>`;
      document.getElementById('confBar').style.background = '#ef4444';
      document.getElementById('confBar').style.width = `${conf}%`;
      document.getElementById('confText').textContent = `${conf}%`;
      document.getElementById('domainStatusText').textContent = "🚨 อยู่นอก 3 คลาส (OOD)";
      document.getElementById('domainStatusText').style.color = '#ef4444';
      document.getElementById('actionText').textContent = "ปฏิเสธการจำแนกโรค (ภาพนอกระบบ)";
      document.getElementById('messageBox').className = "message-box msg-ood";
      document.getElementById('messageBox').textContent = data.status_message || "ภาพนี้ไม่ตรงกับโรคใบมะเขือเทศใดๆ ใน 3 คลาสของระบบ (ระบบปฏิเสธการจำแนกโรค)";
    } else {
      let badgeClass = data.disease_class === 'Healthy' ? 'badge-success' : 'badge-mosaic';
      badge.innerHTML = `<span class="badge ${badgeClass}">✅ ผลวินิจฉัย: ${data.disease_class}</span>`;
      document.getElementById('confBar').style.background = '#10b981';
      document.getElementById('confBar').style.width = `${conf}%`;
      document.getElementById('confText').textContent = `${conf}%`;
      document.getElementById('domainStatusText').textContent = "🌿 ตรงกับโรคในระบบ";
      document.getElementById('domainStatusText').style.color = '#10b981';
      document.getElementById('actionText').textContent = "ปฏิบัติตามคู่มือการรักษา";
      document.getElementById('messageBox').className = "message-box msg-ok";
      document.getElementById('messageBox').textContent = data.status_message || "การจำแนกโรคสำเร็จและมีความเชื่อมั่นในระดับปกติ";
    }

    document.getElementById('latencyText').textContent = `${data.inference_latency_ms} ms`;
    document.getElementById('modelAliasText').textContent = `@${data.model_alias || 'champion'}`;
  }
</script>
</body>
</html>
"""


@app.get("/health")
def health():
    REQUEST_COUNT.labels(endpoint="/health", status="success").inc()

    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_uri": MODEL_URI,
        "model_version": MODEL_VERSION,
        "process_id": os.getpid(),
    }

@app.post("/internal/shutdown", include_in_schema=False)
def managed_shutdown(background: BackgroundTasks, authorization: str = Header(default="")):
    token = os.getenv("PIPELINE_CONTROL_TOKEN")
    if not token or not secrets.compare_digest(authorization, "Bearer " + token):
        raise HTTPException(status_code=403, detail="Managed controller authentication required")
    background.add_task(os.kill, os.getpid(), signal.SIGTERM)
    return {"status": "stopping"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    started = time.perf_counter()

    try:
        allowed_types = {"image/jpeg", "image/png"}

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=415,
                detail="รองรับเฉพาะไฟล์ JPEG และ PNG",
            )

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="ไฟล์ภาพว่างเปล่า",
            )

        with Image.open(io.BytesIO(contents)) as image:
            feature = extract_features(image)

        features = np.expand_dims(feature, axis=0)
        raw_prediction = str(model.predict(features)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            confidence = float(np.max(probs))

        # 1. ตรวจสอบระยะห่างทางฟีเจอร์จาก Domain ของทั้ง 3 คลาส (Centroid Distance)
        min_class_dist = None
        is_distance_ood = False
        if class_centroids:
            min_class_dist = float(min(np.linalg.norm(feature - c) for c in class_centroids.values()))
            if min_class_dist > MAX_DOMAIN_DISTANCE:
                is_distance_ood = True

        # 2. ตรวจสอบความเชื่อมั่น (Confidence Threshold)
        is_conf_ood = (confidence is not None and confidence < CONFIDENCE_THRESHOLD)

        is_ood = is_distance_ood or is_conf_ood

        if is_ood:
            # หากภาพไม่ได้อยู่ใน 3 คลาส ต้องระบุชัดเจนว่าไม่รู้จักและไม่ตรงกับคลาสใด
            prediction = "ไม่รู้จัก / Unknown (ไม่อยู่ใน 3 คลาสของระบบ)"
            match_status = "NO_MATCH"
            candidate_output = None
            if is_distance_ood:
                status_message = (
                    f"ภาพนี้ไม่อยู่ใน 3 คลาสของระบบ (ระยะห่าง {min_class_dist:.2f} > ขอบเขต {MAX_DOMAIN_DISTANCE:.2f})"
                )
            else:
                status_message = (
                    f"ภาพนี้ไม่ตรงกับโรคใดในระบบ (ความเชื่อมั่น {confidence:.1%} < เกณฑ์ปลอดภัย {CONFIDENCE_THRESHOLD:.1%})"
                )
        else:
            prediction = raw_prediction
            match_status = "MATCH"
            candidate_output = raw_prediction
            status_message = "จำแนกโรคสำเร็จ (ตรงกับ 1 ใน 3 คลาสของระบบ)"

        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            3,
        )

        REQUEST_COUNT.labels(
            endpoint="/predict",
            status="success",
        ).inc()

        PREDICTION_COUNT.labels(
            disease_class=prediction,
        ).inc()

        INFERENCE_LATENCY.observe(latency_ms / 1000)

        logger.info(
            "prediction=%s match_status=%s dist=%s conf=%s is_ood=%s latency_ms=%s filename=%s",
            prediction,
            match_status,
            min_class_dist,
            confidence,
            is_ood,
            latency_ms,
            file.filename,
        )

        return {
            "filename": file.filename,
            "disease_class": prediction,
            "match_status": match_status,
            "is_out_of_distribution": is_ood,
            "confidence_score": confidence,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "domain_distance": min_class_dist,
            "max_domain_distance": MAX_DOMAIN_DISTANCE,
            "status_message": status_message,
            "suggested_action": "expert_review" if is_ood else "standard_treatment",
            "inference_latency_ms": latency_ms,
            "model_name": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
        }

    except HTTPException:
        REQUEST_COUNT.labels(
            endpoint="/predict",
            status="client_error",
        ).inc()
        ERROR_COUNT.labels(error_type="http_error").inc()
        raise

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        REQUEST_COUNT.labels(
            endpoint="/predict",
            status="client_error",
        ).inc()
        ERROR_COUNT.labels(error_type="invalid_image").inc()

        logger.warning("Invalid image: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="ไฟล์ที่ส่งมาไม่ใช่ภาพที่อ่านได้",
        ) from exc

    except Exception as exc:
        REQUEST_COUNT.labels(
            endpoint="/predict",
            status="server_error",
        ).inc()
        ERROR_COUNT.labels(error_type="prediction_error").inc()

        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="ระบบไม่สามารถประมวลผลภาพได้",
        ) from exc