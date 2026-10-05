import json

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import accuracy_score, f1_score

from common import MIN_VAL_ACCURACY, ROOT

MONITORING_DATA = ROOT / "monitoring_data"
REPORTS = ROOT / "reports"

P_VALUE_THRESHOLD = 0.05
DRIFT_SHARE_THRESHOLD = 0.20


NUMERIC_FEATURES = [
    "brightness_mean",
    "contrast_std",
    "red_mean",
    "green_mean",
    "blue_mean",
    "saturation_mean",
    "feature_mean",
    "feature_std",
    "feature_l2",
    "confidence",
]

CATEGORICAL_FEATURES = [
    "y_pred",
]


def numerical_drift(reference, current, column):
    """
    Kolmogorov-Smirnov test

    H0 = reference และ current
         มาจาก distribution เดียวกัน

    p < 0.05
    = มีหลักฐานว่าการกระจายเปลี่ยน
    """

    result = ks_2samp(
        reference[column].dropna(),
        current[column].dropna(),
    )

    return {
        "test": "Kolmogorov-Smirnov",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "drifted": bool(
            result.pvalue < P_VALUE_THRESHOLD
        ),
    }


def categorical_drift(reference, current, column):
    """
    Chi-square test สำหรับ categorical feature
    เช่น prediction distribution
    """

    categories = sorted(
        set(reference[column].dropna())
        | set(current[column].dropna())
    )

    reference_counts = [
        int((reference[column] == category).sum())
        for category in categories
    ]

    current_counts = [
        int((current[column] == category).sum())
        for category in categories
    ]

    contingency = [
        reference_counts,
        current_counts,
    ]

    statistic, p_value, _, _ = chi2_contingency(
        contingency
    )

    return {
        "test": "Chi-square",
        "categories": categories,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drifted": bool(
            p_value < P_VALUE_THRESHOLD
        ),
    }


def main():

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_path = (
        MONITORING_DATA / "reference.csv"
    )

    current_path = (
        MONITORING_DATA / "current.csv"
    )

    reference = pd.read_csv(
        reference_path
    )

    current = pd.read_csv(
        current_path
    )

    print(
        "Reference:",
        reference.shape,
    )

    print(
        "Current:",
        current.shape,
    )

    # =========================================
    # 1. Evidently HTML Drift Report
    # =========================================

    monitoring_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    data_definition = DataDefinition(
        numerical_columns=NUMERIC_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
    )

    reference_dataset = Dataset.from_pandas(
        reference[monitoring_columns],
        data_definition=data_definition,
    )

    current_dataset = Dataset.from_pandas(
        current[monitoring_columns],
        data_definition=data_definition,
    )

    report = Report(
        [
            DataDriftPreset()
        ]
    )

    snapshot = report.run(
        reference_data=reference_dataset,
        current_data=current_dataset,
    )

    html_path = (
        REPORTS
        / "tomato_evidently_drift.html"
    )

    snapshot.save_html(
        str(html_path)
    )

    # =========================================
    # 2. Statistical Drift Detection
    # =========================================

    drift_results = {}

    print("\n===== NUMERICAL DRIFT =====")

    for column in NUMERIC_FEATURES:

        result = numerical_drift(
            reference,
            current,
            column,
        )

        drift_results[column] = result

        print(
            f"{column:20s}",
            f"p={result['p_value']:.6f}",
            "DRIFT"
            if result["drifted"]
            else "OK",
        )

    print("\n===== CATEGORICAL DRIFT =====")

    for column in CATEGORICAL_FEATURES:

        result = categorical_drift(
            reference,
            current,
            column,
        )

        drift_results[column] = result

        print(
            f"{column:20s}",
            f"p={result['p_value']:.6f}",
            "DRIFT"
            if result["drifted"]
            else "OK",
        )

    drifted_features = [
        column
        for column, result
        in drift_results.items()
        if result["drifted"]
    ]

    total_features = len(
        drift_results
    )

    drift_share = (
        len(drifted_features)
        / total_features
    )

    # =========================================
    # 3. Model Performance Monitoring
    # =========================================

    accuracy = accuracy_score(
        current["y_true"],
        current["y_pred"],
    )

    macro_f1 = f1_score(
        current["y_true"],
        current["y_pred"],
        average="macro",
        zero_division=0,
    )

    mean_confidence = float(
        current["confidence"].mean()
    )

    # =========================================
    # 4. Alert decision
    # =========================================

    data_drift_alert = (
        drift_share
        > DRIFT_SHARE_THRESHOLD
    )

    performance_alert = (
        accuracy
        < MIN_VAL_ACCURACY
    )

    summary = {
        "reference_rows":
            int(len(reference)),

        "current_rows":
            int(len(current)),

        "accuracy":
            round(float(accuracy), 4),

        "macro_f1":
            round(float(macro_f1), 4),

        "mean_confidence":
            round(mean_confidence, 4),

        "drifted_features":
            drifted_features,

        "drifted_feature_count":
            len(drifted_features),

        "total_monitored_features":
            total_features,

        "drift_share":
            round(float(drift_share), 4),

        "drift_share_threshold":
            DRIFT_SHARE_THRESHOLD,

        "minimum_accuracy":
            float(MIN_VAL_ACCURACY),

        "data_drift_alert":
            bool(data_drift_alert),

        "performance_alert":
            bool(performance_alert),

        "drift_results":
            drift_results,
    }

    json_path = (
        REPORTS
        / "tomato_monitoring_summary.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n===== MODEL PERFORMANCE =====")

    print(
        f"Accuracy        : {accuracy:.4f}"
    )

    print(
        f"Macro F1        : {macro_f1:.4f}"
    )

    print(
        f"Mean confidence : {mean_confidence:.4f}"
    )

    print("\n===== DATA DRIFT =====")

    print(
        "Drifted features:",
        drifted_features,
    )

    print(
        "Drift share:",
        f"{drift_share:.4f}",
    )

    print(
        "Drift threshold:",
        DRIFT_SHARE_THRESHOLD,
    )

    print("\n===== ALERT =====")

    print(
        "Data drift alert:",
        data_drift_alert,
    )

    print(
        "Performance alert:",
        performance_alert,
    )

    print("\nSaved:")
    print(html_path)
    print(json_path)


if __name__ == "__main__":
    main()
