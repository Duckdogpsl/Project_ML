FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=dataset/sample
ENV MODELS=random_forest

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY scripts ./scripts
COPY dataset/sample ./dataset/sample

CMD ["python", "scripts/03_train_evaluate_register.py"]
