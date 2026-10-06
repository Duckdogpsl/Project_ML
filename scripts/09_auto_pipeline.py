"""Bounded training/retraining with quality gates and a managed local API.

Run with the project's Python environment. Runtime state is ignored under reports/.
"""

import argparse
import fcntl
import importlib.util
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    if path.name == "serving.json":
        temporary.chmod(0o600)
    temporary.replace(path)


def decision(summary):
    # Reuse the team's decision policy; never interpret an API outage as model drift.
    spec = importlib.util.spec_from_file_location(
        "retrain_decision", ROOT / "scripts/08_retrain_decision.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decide(summary)


def orchestrate(mode, attempts, initial_models, retry_models, ops, record):
    """Keep orchestration independent of MLflow/processes for failure-path tests."""
    if attempts < 1 or attempts > 5:
        raise ValueError("attempts must be between 1 and 5")
    if mode == "monitor":
        summary = ops.monitor()
        action = decision(summary)["action"]
        record("monitoring", action=action, accuracy=summary["accuracy"])
        if action != "RETRAIN":
            return action
        if not ops.can_retrain():
            record("cooldown", status="skipped", seconds=3600)
            return "COOLDOWN"
    ops.prepare()
    record("data_validation_and_preprocessing", status="passed")
    for attempt in range(1, attempts + 1):
        models = initial_models if attempt == 1 else retry_models
        record("training", attempt=attempt, models=models)
        result = ops.train(attempt, models)
        record("model_validation", attempt=attempt, **result)
        if not result["approved"]:
            continue
        version = result["version"]
        if mode == "monitor" and not ops.current_data_gate(version):
            record("current_data_gate", attempt=attempt, status="rejected", version=version)
            continue
        # Deployment exceptions are operational failures, not reasons to retrain.
        ops.deploy(version)
        record("deployment", status="passed", version=version)
        return "DEPLOYED"
    record("retry_limit", status="failed", attempts=attempts)
    return "RETRY_LIMIT"


class NativeDeployment:
    """Smoke-test an immutable model version, then restart only our managed API.

    Uses a separate port (8001 by default) so the team's original API is untouched.
    The previous version is restored if the new process fails its startup checks.
    """

    def __init__(self, state_dir, port, env, image):
        self.state_dir, self.port, self.env, self.image = state_dir, port, env, image
        self.state_file = state_dir / "serving.json"

    @staticmethod
    def health(port):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return json.load(response)

    def verify(self, port, version, pid):
        body = self.health(port)
        if body.get("model_version") != str(version) or body.get("process_id") != pid:
            raise RuntimeError("API identity/version does not match the managed process")
        # Exercise real feature extraction and model.predict, not just /health.
        boundary = "tomato-pipeline-boundary"
        mime = "image/png" if self.image.suffix.lower() == ".png" else "image/jpeg"
        payload = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="probe{self.image.suffix}"\r\nContent-Type: {mime}\r\n\r\n'
        ).encode()
        payload += self.image.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/predict",
            data=payload,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            prediction = json.load(response)
        if not prediction.get("disease_class"):
            raise RuntimeError("Prediction smoke test failed")

    def start(self, port, version):
        token = secrets.token_urlsafe(32)
        env = {**self.env, "MODEL_VERSION": str(version), "PIPELINE_CONTROL_TOKEN": token}
        env.pop("RUNNER_TRACKING_ID", None)
        log_path = self.state_dir / f"serving-{port}.log"
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "05_model_serving:app",
                    "--app-dir",
                    "scripts",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Serving exited; inspect {log_path}")
                try:
                    self.verify(port, version, process.pid)
                    return {
                        "pid": process.pid,
                        "port": port,
                        "version": str(version),
                        "control_token": token,
                    }
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    time.sleep(0.5)
            raise RuntimeError(f"Serving startup timed out; inspect {log_path}")
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise

    @staticmethod
    def stop(state):
        # A PID file alone is insufficient: prove identity before sending a signal.
        body = NativeDeployment.health(state["port"])
        if body.get("process_id") != state["pid"] or body.get("model_version") != state["version"]:
            raise RuntimeError("Refusing to stop a service whose identity has changed")
        token = state.get("control_token")
        if not token:
            raise RuntimeError(
                "Legacy service has no authenticated controller; keep it and use a new port"
            )
        request = urllib.request.Request(
            f"http://127.0.0.1:{state['port']}/internal/shutdown",
            data=b"",
            headers={"Authorization": "Bearer " + token},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError("Managed shutdown was not acknowledged")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                NativeDeployment.health(state["port"])
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                return
            time.sleep(0.2)
        raise RuntimeError("Managed API did not stop; refusing to replace it")

    def deploy(self, version):
        # Test the candidate on an unused port while the incumbent is still serving.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            probe_port = sock.getsockname()[1]
        probe = self.start(probe_port, version)
        self.stop(probe)
        previous = json.loads(self.state_file.read_text()) if self.state_file.exists() else None
        if previous:
            if previous["port"] != self.port:
                raise RuntimeError("Managed API port changed; use the port in serving.json")
            self.stop(previous)
        else:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                    raise RuntimeError(
                        "Port is occupied by an unmanaged service; it was not stopped"
                    )
        current = None
        try:
            current = self.start(self.port, version)
            write_json(self.state_file, current)
        except BaseException:
            if current is not None:
                self.stop(current)
            if previous:
                restored = self.start(self.port, previous["version"])
                write_json(self.state_file, restored)
            raise
        return previous


class PipelineOps:
    def __init__(self, args):
        self.args = args
        self.state_dir = args.state_dir.resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Every invocation has new evidence; existing monitoring CSVs/reports remain intact.
        self.run_dir = self.state_dir / f"run-{time.time_ns()}"
        self.run_dir.mkdir()
        self.env = {
            **os.environ,
            "DATA_DIR": str(args.data_dir.resolve()),
            "CLASSES": "all",
            "PROCESSED_DIR": str(self.state_dir / "processed_data"),
            "AUTO_PROMOTE": "0",
            "FORCE_PROMOTE": "0",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(self.state_dir / "matplotlib"),
            "MLFLOW_DISABLE_AGENT_HINT": "1",
        }
        self.events = []

    def record(self, stage, **details):
        event = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            **details,
        }
        self.events.append(event)
        print(json.dumps(event), flush=True)
        write_json(self.run_dir / "events.json", self.events)

    def run(self, script, *arguments, env=None, allowed=(0,)):
        child_env = {**self.env, **(env or {})}
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *arguments],
            cwd=ROOT,
            env=child_env,
            timeout=self.args.step_timeout,
        )
        if result.returncode not in allowed:
            raise RuntimeError(f"{script} failed (exit {result.returncode}); pipeline stopped")

    def can_retrain(self):
        path = self.state_dir / "last_retrain.json"
        return not path.exists() or time.time() - json.loads(path.read_text())["time"] >= 3600

    def prepare(self):
        # Set tracking explicitly: validation/preprocessing otherwise default to ./mlruns.
        from common import MLFLOW_URI

        self.env["MLFLOW_TRACKING_URI"] = MLFLOW_URI
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        artifacts = ROOT / os.getenv("MLFLOW_ARTIFACT_DIR", "mlruns")
        for name, folder in [
            ("Tomato Leaf Disease - Data Validation", "validation"),
            ("Tomato Leaf Disease - Data Preprocessing", "preprocessing"),
        ]:
            if mlflow.get_experiment_by_name(name) is None:
                mlflow.create_experiment(name, artifact_location=(artifacts / folder).as_uri())
        self.run("01.1_data_validation.py", "--data-root", self.env["DATA_DIR"])
        self.run("01.2_preprocess.py")

    def train(self, attempt, models):
        output = self.run_dir / f"train-{attempt}.json"
        self.run(
            "02_train_evaluate_register.py",
            env={"MODELS": models, "TRAIN_RESULT_PATH": str(output)},
            allowed=(0, 2),
        )
        return json.loads(output.read_text())

    def monitor(self, version=None):
        from mlflow.tracking import MlflowClient

        from common import MODEL_ALIAS, MODEL_NAME, setup_mlflow

        setup_mlflow()
        if version is None:
            if not (self.state_dir / "serving.json").exists():
                raise RuntimeError("Bootstrap with --mode train before scheduled monitoring")
            state = json.loads((self.state_dir / "serving.json").read_text())
            body = NativeDeployment.health(state["port"])
            if (
                body.get("model_version") != state["version"]
                or body.get("process_id") != state["pid"]
            ):
                raise RuntimeError("Deployed API identity changed; monitoring stopped")
            version = state["version"]
            alias = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
            if str(alias) != version:
                raise RuntimeError(
                    "Registry and deployed model disagree; reconcile deployment first"
                )
        directory = self.run_dir / f"monitor-{version}"
        env = {
            "MODEL_VERSION": str(version),
            "MONITORING_DATA_DIR": str(directory / "data"),
            "REPORTS_DIR": str(directory / "reports"),
        }
        flags = (
            [
                "--simulate-drift",
                "--brightness-factor",
                str(self.args.brightness_factor),
                "--blur-radius",
                str(self.args.blur_radius),
            ]
            if self.args.simulate_drift
            else []
        )
        self.run("06_prepare_monitoring_data.py", *flags, env=env)
        self.run("07_evidently_drift.py", env=env)
        return json.loads((directory / "reports/tomato_monitoring_summary.json").read_text())

    def current_data_gate(self, version):
        summary = self.monitor(version)
        return not summary["performance_alert"]

    def deploy(self, version):
        from mlflow.tracking import MlflowClient

        from common import MODEL_ALIAS, MODEL_NAME, setup_mlflow

        setup_mlflow()
        client = MlflowClient()
        # Check candidate identity and dimensions by actually serving a JPEG/PNG.
        images = sorted(
            p
            for p in (self.args.data_dir / "val").rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not images:
            raise RuntimeError("No validation image for deployment smoke test")
        adapter = NativeDeployment(self.state_dir, self.args.port, self.env, images[0])
        old_alias = client.get_registered_model(MODEL_NAME).aliases.get(MODEL_ALIAS)
        previous = adapter.deploy(version)
        try:
            client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
            client.set_model_version_tag(MODEL_NAME, version, "status", "champion")
        except BaseException:
            if previous:
                adapter.deploy(previous["version"])
            else:
                state = json.loads(adapter.state_file.read_text())
                adapter.stop(state)
                adapter.state_file.unlink()
            if old_alias is not None:
                client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, old_alias)
            else:
                client.delete_registered_model_alias(MODEL_NAME, MODEL_ALIAS)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "monitor"], default="train")
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / os.getenv("DATA_DIR", "dataset/tomato")
    )
    parser.add_argument("--state-dir", type=Path, default=ROOT / "reports/auto_pipeline")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--models", default="logreg,random_forest")
    parser.add_argument("--retry-models", default="logreg_C1.0,rf_n300_depthNone")
    parser.add_argument("--step-timeout", type=int, default=1800)
    parser.add_argument("--simulate-drift", action="store_true", help="Offline demonstration only")
    parser.add_argument("--brightness-factor", type=float, default=0.55)
    parser.add_argument("--blur-radius", type=float, default=1.2)
    args = parser.parse_args()
    if not 0 < args.brightness_factor <= 1 or args.blur_radius < 0:
        parser.error("brightness must be in (0, 1] and blur radius nonnegative")
    if not 1 <= args.max_attempts <= 5 or not 1 <= args.port <= 65535 or args.step_timeout < 1:
        parser.error("Invalid attempts, port, or timeout")
    args.data_dir = args.data_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    # Serialize scheduled/manual jobs so two pipelines cannot change champion concurrently.
    with (args.state_dir / "pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another pipeline is running; skipped", flush=True)
            return 0
        ops = PipelineOps(args)
        try:
            # Sustained failures have a cooldown across scheduled jobs, not just in one job.
            cooldown = args.state_dir / "last_retrain.json"
            outcome = orchestrate(
                args.mode, args.max_attempts, args.models, args.retry_models, ops, ops.record
            )
            if args.mode == "monitor" and outcome in {"DEPLOYED", "RETRY_LIMIT"}:
                write_json(cooldown, {"time": time.time(), "outcome": outcome})
            write_json(ops.run_dir / "result.json", {"outcome": outcome})
            return 2 if outcome == "RETRY_LIMIT" else 0
        except Exception as error:
            ops.record("pipeline_error", status="failed", message=str(error))
            print(str(error), file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
