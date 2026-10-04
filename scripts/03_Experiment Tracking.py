import inspect
import os
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (
    MIN_VAL_ACCURACY,
    MODEL_ALIAS,
    MODEL_NAME,
    PROCESSED_DIR,
    SEED,
    setup_mlflow,
)
from tracking import data_version, git_info, log_provenance

DEFAULT_MODELS = "baseline,logreg,random_forest,svc_rbf"
FORCE_PROMOTE = os.getenv("FORCE_PROMOTE", "0") == "1"


def build_candidates():
    candidates = {
        "baseline": (
            DummyClassifier(strategy="most_frequent"),
            {"model": "DummyClassifier", "strategy": "most_frequent"},
        ),
        "logreg": (
            make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
            {"model": "LogisticRegression", "C": 0.05},
        ),
        "random_forest": (
            make_pipeline(
                RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
            ),
            {"model": "RandomForest", "n_estimators": 200},
        ),
        "svc_rbf": (
            make_pipeline(
                StandardScaler(),
                PCA(n_components=150, random_state=SEED),
                SVC(C=10, gamma="scale", random_state=SEED),
            ),
            {"model": "PCA+SVC(rbf)", "pca_components": 150, "C": 10},
        ),
    }
    add_sweep_variants(candidates)
    return candidates


def add_sweep_variants(candidates):
    """ชุดทดลองปรับพารามิเตอร์ทีละตัว (ใช้เมื่อ MODELS=sweep)

    ปรับค่าที่ 'คุมความซับซ้อนของโมเดล' ของแต่ละตระกูล เพื่อดูว่าซับซ้อนขึ้น/น้อยลงแล้วผลเปลี่ยนยังไง
      logreg : C  (ยิ่งมาก = ยิ่งยืดหยุ่น เสี่ยง overfit)
      forest : จำนวนต้นไม้ และความลึกสูงสุด (ลึก = ซับซ้อน)
      svc    : C และจำนวนมิติหลัง PCA
    """
    for c in (0.01, 1.0):
        candidates[f"logreg_C{c}"] = (
            make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)),
            {"model": "LogisticRegression", "C": c},
        )
    for n, depth in ((100, 10), (300, None), (200, 5)):
        candidates[f"rf_n{n}_depth{depth}"] = (
            make_pipeline(
                RandomForestClassifier(n_estimators=n, max_depth=depth, n_jobs=-1, random_state=SEED)
            ),
            {"model": "RandomForest", "n_estimators": n, "max_depth": depth},
        )
    for c, comps in ((1, 150), (100, 150), (10, 50)):
        candidates[f"svc_C{c}_pca{comps}"] = (
            make_pipeline(
                StandardScaler(),
                PCA(n_components=comps, random_state=SEED),
                SVC(C=c, gamma="scale", random_state=SEED),
            ),
            {"model": "PCA+SVC(rbf)", "pca_components": comps, "C": c},
        )


def load(split):
    d = np.load(PROCESSED_DIR / f"{split}.npz")
    return d["X"], d["y"]


def evaluate(model, X, y, prefix):
    pred = model.predict(X)
    return pred, {
        f"{prefix}_accuracy": accuracy_score(y, pred),
        f"{prefix}_f1_macro": f1_score(y, pred, average="macro"),
    }


def current_champion(client):
    """คืน (เวอร์ชัน, val_accuracy) ของ @champion ปัจจุบัน หรือ (None, None) ถ้ายังไม่มี"""
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    except MlflowException:
        return None, None
    tag = mv.tags.get("val_accuracy")
    return mv.version, (float(tag) if tag else None)


def train():
    setup_mlflow()
    X_train, y_train = load("train")
    X_val, y_val = load("val")
    X_test, y_test = load("test")
    print(f"train={X_train.shape} val={X_val.shape} test={X_test.shape}")
    print(f"code={git_info()['git_commit'][:8]} data_version={data_version()}")

    candidates = build_candidates()
    raw = os.getenv("MODELS", DEFAULT_MODELS).strip()
    selected = list(candidates) if raw == "sweep" else [m.strip() for m in raw.split(",")]
    unknown = [m for m in selected if m not in candidates]
    if unknown:
        raise SystemExit(f"ไม่รู้จักโมเดล {unknown} — ที่มีให้เลือก: {list(candidates)} หรือ sweep")

    with mlflow.start_run(run_name="train_evaluate_register") as parent:
        mlflow.set_tag("ml.step", "train_evaluate_register")
        log_provenance(full_environment=True)
        results = {}

        # ---------- เทรนและเทียบทุกโมเดลบน val ----------
        for name in selected:
            model, params = candidates[name]
            with mlflow.start_run(run_name=name, nested=True) as child:
                log_provenance()
                mlflow.log_params(params)
                t0 = time.time()
                model.fit(X_train, y_train)
                _, metrics = evaluate(model, X_val, y_val, "val")
                _, train_m = evaluate(model, X_train, y_train, "train")
                metrics["train_accuracy"] = train_m["train_accuracy"]
                # ช่องว่าง train - val : ยิ่งมาก = ยิ่งท่องข้อมูลเก่า (overfit)
                metrics["overfit_gap"] = train_m["train_accuracy"] - metrics["val_accuracy"]
                metrics["train_seconds"] = round(time.time() - t0, 1)
                mlflow.log_metrics(metrics)
                results[name] = (model, metrics, child.info.run_id)
                print(f"{name:>20}: train={metrics['train_accuracy']:.4f} "
                      f"val={metrics['val_accuracy']:.4f} f1={metrics['val_f1_macro']:.4f} "
                      f"({metrics['train_seconds']}s)")

        # ---------- ตารางเปรียบเทียบการทดลอง ----------
        table = pd.DataFrame(
            [
                {
                    "model": n,
                    "train_accuracy": round(m["train_accuracy"], 4),
                    "val_accuracy": round(m["val_accuracy"], 4),
                    "val_f1_macro": round(m["val_f1_macro"], 4),
                    "overfit_gap": round(m["overfit_gap"], 4),
                    "train_seconds": m["train_seconds"],
                    "run_id": rid,
                }
                for n, (_, m, rid) in results.items()
            ]
        ).sort_values(["val_accuracy", "val_f1_macro"], ascending=False)
        print("\n" + table.drop(columns="run_id").to_string(index=False))
        mlflow.log_text(table.to_csv(index=False), "experiment_comparison.csv")

        # ---------- เลือกโมเดลที่ดีที่สุด (ไม่นับ baseline) ----------
        contenders = {n: v for n, v in results.items() if n != "baseline"} or results
        best_name = max(
            contenders,
            key=lambda n: (contenders[n][1]["val_accuracy"], contenders[n][1]["val_f1_macro"]),
        )
        best_model, best_val, _ = results[best_name]
        print(f"\nBest model: {best_name}")
        if "baseline" in results:
            gain = best_val["val_accuracy"] - results["baseline"][1]["val_accuracy"]
            mlflow.log_metric("gain_over_baseline", gain)
            print(f"Gain over baseline: {gain:+.4f}")

        # ---------- ประเมินตัวที่ดีที่สุดบน test ----------
        test_pred, test_metrics = evaluate(best_model, X_test, y_test, "test")
        mlflow.log_param("best_model", best_name)
        mlflow.log_metrics({**best_val, **test_metrics})
        report = classification_report(y_test, test_pred, digits=4)
        print(report)
        mlflow.log_text(report, "classification_report.txt")

        fig, ax = plt.subplots(figsize=(9, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_test, test_pred, ax=ax, xticks_rotation=60, colorbar=False
        )
        ax.set_title(f"Confusion matrix (test) — {best_name}")
        fig.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # ---------- ด่านที่ 1: เกณฑ์ขั้นต่ำ ----------
        if best_val["val_accuracy"] < MIN_VAL_ACCURACY:
            mlflow.set_tag("registered", "false")
            raise SystemExit(
                f"val_accuracy {best_val['val_accuracy']:.4f} < {MIN_VAL_ACCURACY} — ไม่ register โมเดล"
            )

        # ---------- Model Registry ----------
        # MLflow รุ่นใหม่บันทึกโมเดลด้วย skops ซึ่งต้องระบุชนิด object ที่ไว้ใจ (เช่น Tree ของ RandomForest)
        # เราสร้างโมเดลเองจึงไว้ใจได้ — รุ่นเก่าที่ไม่มีพารามิเตอร์นี้จะข้ามไป
        extra = {}
        if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
            extra["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
        client = MlflowClient()
        old_version, old_val_acc = current_champion(client)

        info = mlflow.sklearn.log_model(
            best_model,
            name="model",
            registered_model_name=MODEL_NAME,
            signature=infer_signature(X_train[:5], best_model.predict(X_train[:5])),
            input_example=X_train[:2],
            **extra,
        )
        version = info.registered_model_version
        # แปะ "ใบประวัติ" ไว้กับโมเดลเวอร์ชันนี้ ให้ย้อนดูได้ว่ามาจากโค้ด/ข้อมูล/การทดลองไหน
        for key, value in {
            "algorithm": best_name,
            "val_accuracy": f"{best_val['val_accuracy']:.4f}",
            "test_accuracy": f"{test_metrics['test_accuracy']:.4f}",
            "git_commit": git_info()["git_commit"],
            "data_version": data_version(),
            "run_id": parent.info.run_id,
        }.items():
            client.set_model_version_tag(MODEL_NAME, version, key, value)

        # ---------- ด่านที่ 2: ต้องไม่แย่กว่า champion ตัวเดิม ----------
        worse = old_val_acc is not None and best_val["val_accuracy"] < old_val_acc - 1e-9
        if worse and not FORCE_PROMOTE:
            client.set_model_version_tag(MODEL_NAME, version, "status", "rejected")
            mlflow.set_tag("registered", "true")
            mlflow.set_tag("promoted", "false")
            print(f"Registered {MODEL_NAME} v{version} but NOT promoted: "
                  f"val_acc {best_val['val_accuracy']:.4f} < champion v{old_version} ({old_val_acc:.4f})")
            return

        client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
        client.set_model_version_tag(MODEL_NAME, version, "status", "champion")
        if old_version is not None:
            client.set_model_version_tag(MODEL_NAME, old_version, "status", "archived")
        mlflow.set_tag("registered", "true")
        mlflow.set_tag("promoted", "true")
        print(f"Registered {MODEL_NAME} v{version} as @{MODEL_ALIAS} "
              f"(test_acc={test_metrics['test_accuracy']:.4f}, run={parent.info.run_id})")


if __name__ == "__main__":
    train()