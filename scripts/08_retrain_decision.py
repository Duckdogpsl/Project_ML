import argparse
import json
from pathlib import Path


def decide(summary):
    accuracy = summary["accuracy"]
    minimum_accuracy = summary["minimum_accuracy"]

    drift_share = summary["drift_share"]
    drift_threshold = summary["drift_share_threshold"]

    data_drift_alert = summary["data_drift_alert"]
    performance_alert = summary["performance_alert"]

    if performance_alert:
        return {
            "action": "RETRAIN",
            "reason": (
                f"Model performance degraded: "
                f"accuracy={accuracy:.4f} "
                f"< minimum_accuracy={minimum_accuracy:.4f}"
            ),
        }

    if data_drift_alert:
        return {
            "action": "WATCH",
            "reason": (
                f"Data drift detected: "
                f"drift_share={drift_share:.4f} "
                f"> threshold={drift_threshold:.4f}, "
                "but model performance is still acceptable"
            ),
        }

    return {
        "action": "OK",
        "reason": (
            "Model performance and data distribution "
            "are within acceptable thresholds"
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary",
        default="reports/tomato_monitoring_summary.json",
        help="Path to monitoring summary JSON",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary)

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Monitoring summary not found: {summary_path}"
        )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    decision = decide(summary)

    print("===== MONITORING DECISION =====")
    print("Summary :", summary_path)
    print("Accuracy:", summary["accuracy"])
    print(
        "Drift   :",
        summary["drift_share"],
    )

    print("\nACTION :", decision["action"])
    print("REASON :", decision["reason"])


if __name__ == "__main__":
    main()
