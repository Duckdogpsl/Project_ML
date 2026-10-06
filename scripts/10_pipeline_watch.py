"""Local automation: source/data changes run CI+training; timed checks run monitoring.

Keep this controller running on the deployment Mac. It does not merge GitHub PRs.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def inspect_changes(data_dir, previous):
    source_names = subprocess.check_output(
        ["git", "ls-files", "scripts", "tests", "requirements.txt", "requirements-monitoring.txt"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    sources = [ROOT / name for name in source_names if (ROOT / name).is_file()]
    images = [
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not images:
        raise RuntimeError("No input images; refusing to run an empty pipeline")
    current = {"code": fingerprint(sources), "data": fingerprint(images)}
    changed = [key for key in current if current[key] != previous.get(key)]
    return current, changed


def tick(args, execute=None):
    execute = execute or subprocess.run
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.state_dir / "watch.json"
    previous = json.loads(state_file.read_text()) if state_file.exists() else {}
    current, changed = inspect_changes(args.data_dir, previous)
    due = time.time() - previous.get("last_check", 0) >= args.interval_seconds
    if not changed and not due:
        return "IDLE"
    mode = "train" if changed else "monitor"
    event = {
        "time": time.time(),
        "trigger": "change" if changed else "timer",
        "changed": changed,
        "mode": mode,
    }
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    if mode == "train":
        # The local CI gate must pass before running changed code against the model registry.
        checks = execute(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_auto_pipeline.py",
                "tests/test_pipeline_watch.py",
                "-q",
            ],
            cwd=ROOT,
            env=env,
        )
        if checks.returncode != 0:
            event["outcome"] = "CI_FAILED"
            with (args.state_dir / "watch-events.jsonl").open("a") as log:
                log.write(json.dumps(event) + "\n")
            # Remember failure to avoid endless re-runs until source/data changes again.
            state_file.write_text(json.dumps({**current, "last_check": time.time()}))
            return "CI_FAILED"
    result = execute(
        [
            sys.executable,
            str(ROOT / "scripts/09_auto_pipeline.py"),
            "--mode",
            mode,
            "--data-dir",
            str(args.data_dir),
            "--state-dir",
            str(args.state_dir),
            "--models",
            args.models,
            "--retry-models",
            args.retry_models,
            "--port",
            str(args.port),
        ],
        cwd=ROOT,
        env=env,
    )
    event["exit_code"] = result.returncode
    event["outcome"] = "COMPLETED" if result.returncode == 0 else "FAILED_RETAINED_MODEL"
    with (args.state_dir / "watch-events.jsonl").open("a") as log:
        log.write(json.dumps(event) + "\n")
    state_file.write_text(json.dumps({**current, "last_check": time.time()}))
    print(json.dumps(event), flush=True)
    return event["outcome"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/tomato")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "reports/auto_pipeline")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--models", default="logreg,random_forest")
    parser.add_argument("--retry-models", default="logreg_C1.0,rf_n300_depthNone")
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--check-seconds", type=int, default=60)
    parser.add_argument("--run-once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 1 or args.check_seconds < 1:
        parser.error("Intervals must be positive")
    args.data_dir = args.data_dir.resolve()
    args.state_dir = args.state_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "watch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A controller is already running")
        while True:
            try:
                tick(args)
            except Exception as error:
                print(f"Controller error: {error}", file=sys.stderr, flush=True)
                if args.run_once:
                    return 1
            if args.run_once:
                return 0
            time.sleep(args.check_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
