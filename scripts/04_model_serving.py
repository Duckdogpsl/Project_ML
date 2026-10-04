"""FastAPI Model Serving สำหรับจำแนกโรคใบมะเขือเทศ."""

import io
import logging
import time

import mlflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from common import (
    MODEL_ALIAS,
    MODEL_NAME,
    extract_features,
    setup_mlflow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("tomato-serving")

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

setup_mlflow()
logger.info("Loading model from %s", MODEL_URI)
model = mlflow.sklearn.load_model(MODEL_URI)
logger.info("Model loaded successfully")

app = FastAPI(
    title="Tomato Leaf Disease Model Serving",
    description="API สำหรับจำแนกโรคใบมะเขือเทศด้วย MLflow champion model",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "service": "tomato-leaf-disease-classifier",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_uri": MODEL_URI,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="รองรับเฉพาะไฟล์ JPEG และ PNG",
        )

    started = time.perf_counter()

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="ไฟล์ภาพว่างเปล่า",
            )

        with Image.open(io.BytesIO(contents)) as image:
            feature = extract_features(image)

        features = np.expand_dims(feature, axis=0)
        prediction = str(model.predict(features)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(features)[0]))

        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            3,
        )

        logger.info(
            "prediction=%s confidence=%s latency_ms=%s filename=%s",
            prediction,
            confidence,
            latency_ms,
            file.filename,
        )

        return {
            "filename": file.filename,
            "disease_class": prediction,
            "confidence_score": confidence,
            "inference_latency_ms": latency_ms,
            "model_name": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
        }

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Invalid image: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="ไฟล์ที่ส่งมาไม่ใช่ภาพที่อ่านได้",
        ) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="ระบบไม่สามารถประมวลผลภาพได้",
        ) from exc
