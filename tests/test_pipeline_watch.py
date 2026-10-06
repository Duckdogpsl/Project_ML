import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SPEC = importlib.util.spec_from_file_location(
    "watch", Path(__file__).resolve().parents[1] / "scripts/10_pipeline_watch.py"
)
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


def args(tmp_path):
    return SimpleNamespace(
        state_dir=tmp_path,
        data_dir=tmp_path,
        interval_seconds=900,
        models="random_forest",
        retry_models="rf_n300_depthNone",
        port=8001,
    )


def test_changed_code_runs_ci_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(
        watch, "inspect_changes", lambda d, p: ({"code": "new", "data": "same"}, ["code"])
    )
    execute = Mock(return_value=SimpleNamespace(returncode=0))
    assert watch.tick(args(tmp_path), execute) == "COMPLETED"
    assert "pytest" in execute.call_args_list[0].args[0]
    assert "train" in execute.call_args_list[1].args[0]


def test_failed_ci_does_not_train_or_deploy(tmp_path, monkeypatch):
    monkeypatch.setattr(
        watch, "inspect_changes", lambda d, p: ({"code": "new", "data": "same"}, ["code"])
    )
    execute = Mock(return_value=SimpleNamespace(returncode=1))
    assert watch.tick(args(tmp_path), execute) == "CI_FAILED"
    assert execute.call_count == 1


def test_timer_runs_monitoring_without_training(tmp_path, monkeypatch):
    monkeypatch.setattr(
        watch, "inspect_changes", lambda d, p: ({"code": "same", "data": "same"}, [])
    )
    execute = Mock(return_value=SimpleNamespace(returncode=0))
    assert watch.tick(args(tmp_path), execute) == "COMPLETED"
    assert execute.call_count == 1
    assert "monitor" in execute.call_args.args[0]
    event = json.loads((tmp_path / "watch-events.jsonl").read_text())
    assert event["trigger"] == "timer"


def test_no_change_before_interval_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        watch, "inspect_changes", lambda d, p: ({"code": "same", "data": "same"}, [])
    )
    (tmp_path / "watch.json").write_text(json.dumps({"last_check": watch.time.time()}))
    execute = Mock()
    assert watch.tick(args(tmp_path), execute) == "IDLE"
    execute.assert_not_called()
