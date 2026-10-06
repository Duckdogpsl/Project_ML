"""Failure-path tests for retries, deployment gating, and preserving the incumbent."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/09_auto_pipeline.py"
spec = importlib.util.spec_from_file_location("auto_pipeline", MODULE_PATH)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def summary(performance=False, drift=False):
    return {
        "accuracy": 0.5 if performance else 0.9,
        "minimum_accuracy": 0.7,
        "drift_share": 0.5 if drift else 0.1,
        "drift_share_threshold": 0.2,
        "data_drift_alert": drift,
        "performance_alert": performance,
    }


def operations(results):
    ops = Mock()
    ops.train.side_effect = results
    ops.current_data_gate.return_value = True
    ops.can_retrain.return_value = True
    return ops


def run(ops, mode="train", attempts=2):
    return pipeline.orchestrate(mode, attempts, "baseline", "random_forest", ops, Mock())


def test_failed_gate_retrains_and_only_deploys_passing_version():
    ops = operations([{"approved": False, "version": None}, {"approved": True, "version": "2"}])
    assert run(ops) == "DEPLOYED"
    assert ops.train.call_args_list[0].args == (1, "baseline")
    assert ops.train.call_args_list[1].args == (2, "random_forest")
    ops.deploy.assert_called_once_with("2")


def test_retry_limit_never_deploys_rejected_model():
    ops = operations([{"approved": False, "version": None}] * 2)
    assert run(ops) == "RETRY_LIMIT"
    assert ops.train.call_count == 2
    ops.deploy.assert_not_called()


@pytest.mark.parametrize("drift,expected", [(False, "OK"), (True, "WATCH")])
def test_healthy_performance_does_not_retrain_even_with_drift(drift, expected):
    ops = operations([])
    ops.monitor.return_value = summary(drift=drift)
    assert run(ops, "monitor") == expected
    ops.prepare.assert_not_called()
    ops.train.assert_not_called()
    ops.deploy.assert_not_called()


def test_degraded_performance_triggers_real_training_flow():
    ops = operations([{"approved": True, "version": "4"}])
    ops.monitor.return_value = summary(performance=True)
    assert run(ops, "monitor") == "DEPLOYED"
    ops.prepare.assert_called_once()
    ops.current_data_gate.assert_called_once_with("4")
    ops.deploy.assert_called_once_with("4")


def test_candidate_must_also_pass_current_distribution():
    ops = operations([{"approved": True, "version": "4"}] * 2)
    ops.monitor.return_value = summary(performance=True)
    ops.current_data_gate.return_value = False
    assert run(ops, "monitor") == "RETRY_LIMIT"
    ops.deploy.assert_not_called()


def test_operational_training_error_stops_without_blind_retries():
    ops = operations([])
    ops.train.side_effect = RuntimeError("registry unavailable")
    with pytest.raises(RuntimeError, match="registry unavailable"):
        run(ops)
    assert ops.train.call_count == 1
    ops.deploy.assert_not_called()


def test_invalid_data_stops_before_training():
    ops = operations([])
    ops.prepare.side_effect = RuntimeError("corrupt input")
    with pytest.raises(RuntimeError, match="corrupt input"):
        run(ops)
    ops.train.assert_not_called()


def test_serving_outage_does_not_start_retraining():
    ops = operations([])
    ops.monitor.side_effect = ConnectionError("API down")
    with pytest.raises(ConnectionError):
        run(ops, "monitor")
    ops.train.assert_not_called()


def test_cooldown_still_monitors_but_prevents_repeated_training():
    ops = operations([])
    ops.monitor.return_value = summary(performance=True)
    ops.can_retrain.return_value = False
    assert run(ops, "monitor") == "COOLDOWN"
    ops.monitor.assert_called_once()
    ops.train.assert_not_called()


@pytest.mark.parametrize("attempts", [0, -1, 6])
def test_retries_must_be_bounded(attempts):
    with pytest.raises(ValueError):
        run(operations([]), attempts=attempts)


def test_deployment_failure_restores_previous_running_version(tmp_path, monkeypatch):
    previous = {"pid": 111, "port": 8001, "version": "1"}
    pipeline.write_json(tmp_path / "serving.json", previous)
    adapter = pipeline.NativeDeployment(tmp_path, 8001, {}, Path("unused.jpg"))
    adapter.start = Mock(
        side_effect=[
            {"pid": 222, "port": 12345, "version": "2"},
            RuntimeError("candidate failed on serving port"),
            {"pid": 333, "port": 8001, "version": "1"},
        ]
    )
    adapter.stop = Mock()
    with pytest.raises(RuntimeError, match="candidate failed"):
        adapter.deploy("2")
    import json

    assert json.loads((tmp_path / "serving.json").read_text())["version"] == "1"
    assert adapter.start.call_args_list[-1].args == (8001, "1")


def test_does_not_signal_a_reused_pid(monkeypatch):
    monkeypatch.setattr(pipeline.NativeDeployment, "health", lambda port: {"process_id": 999})
    kill = Mock()
    monkeypatch.setattr(pipeline.os, "kill", kill)
    with pytest.raises(RuntimeError, match="identity"):
        pipeline.NativeDeployment.stop({"pid": 111, "port": 8001, "version": "1"})
    kill.assert_not_called()


def test_preprocess_and_serving_share_feature_vector():
    from PIL import Image

    from common import extract_features

    spec = importlib.util.spec_from_file_location(
        "preprocess", MODULE_PATH.parent / "01.2_preprocess.py"
    )
    preprocess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess)
    assert preprocess.extract_features is extract_features
    assert extract_features(Image.new("RGB", (64, 64))).shape == (2276,)


def test_bad_candidate_never_stops_incumbent(tmp_path):
    pipeline.write_json(tmp_path / "serving.json", {"pid": 111, "port": 8001, "version": "1"})
    adapter = pipeline.NativeDeployment(tmp_path, 8001, {}, Path("unused.jpg"))
    adapter.start = Mock(side_effect=RuntimeError("incompatible features"))
    adapter.stop = Mock()
    with pytest.raises(RuntimeError, match="incompatible features"):
        adapter.deploy("2")
    adapter.stop.assert_not_called()


def test_state_write_failure_stops_candidate_before_restoring(tmp_path, monkeypatch):
    previous = {"pid": 111, "port": 8001, "version": "1"}
    pipeline.write_json(tmp_path / "serving.json", previous)
    candidate = {"pid": 222, "port": 8001, "version": "2"}
    probe = {"pid": 333, "port": 12345, "version": "2"}
    adapter = pipeline.NativeDeployment(tmp_path, 8001, {}, Path("unused.jpg"))
    adapter.start = Mock(side_effect=[probe, candidate, previous])
    adapter.stop = Mock()
    monkeypatch.setattr(pipeline, "write_json", Mock(side_effect=[OSError("disk full"), None]))
    with pytest.raises(OSError, match="disk full"):
        adapter.deploy("2")
    assert adapter.stop.call_args_list[-1].args == (candidate,)
    assert adapter.start.call_args_list[-1].args == (8001, "1")
