"""
test_dag.py

Validates that dags/agripulse_dag.py is importable and that its task graph
matches the intended pipeline shape.

Airflow itself is injected as lightweight stubs so this test runs on any
platform. This matters on Windows, where upstream Airflow cannot even be
imported (Airflow 3.x calls os.register_at_fork, which does not exist on
Windows; Airflow only supports Linux/macOS/WSL2). The stubs stand in for the
exact Airflow API surface the DAG uses, so this test detects broken imports,
wrong operator references, and stale task-graph wiring without requiring an
Airflow installation.

Run with:
    pytest tests/test_dag.py -v
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

DAG_PATH = Path(__file__).resolve().parents[1] / "dags" / "agripulse_dag.py"

EXPECTED_TASKS = {
    "ingest_weather",
    "ingest_market",
    "ingest_geo",
    "bronze_to_silver",
    "silver_to_gold",
    "monitor_pipeline_health",
    "prediction_pipeline",
}


class _FakeOperator:
    """Minimal stand-in for BashOperator, supporting the `>>` dependency DSL."""

    def __init__(self, task_id=None, bash_command=None, trigger_rule=None, **kwargs):
        self.task_id = task_id
        self.bash_command = bash_command
        self.trigger_rule = trigger_rule
        self.upstream_task_ids = set()

    def __rshift__(self, other):
        if isinstance(other, (list, tuple)):
            for o in other:
                o.upstream_task_ids.add(self.task_id)
        else:
            other.upstream_task_ids.add(self.task_id)
        return other

    def __rrshift__(self, other):
        if isinstance(other, (list, tuple)):
            self.upstream_task_ids.update(o.task_id for o in other)
        else:
            self.upstream_task_ids.add(other.task_id)
        return self


class _FakeDAG:
    def __init__(self, dag_id=None, schedule_interval=None, **kwargs):
        self.dag_id = dag_id
        self.schedule_interval = schedule_interval

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _TriggerRule:
    ALL_DONE = "all_done"


def _install_airflow_stubs():
    """Register stub airflow modules in sys.modules; returns original entries for restore."""
    saved = {}

    def stub(name):
        saved[name] = sys.modules.get(name)
        mod = types.ModuleType(name)
        mod.__path__ = []  # mark as a package
        sys.modules[name] = mod
        return mod

    airflow = stub("airflow")
    airflow.operators = stub("airflow.operators")
    airflow.utils = stub("airflow.utils")
    stub("airflow.operators.bash").BashOperator = _FakeOperator
    stub("airflow.utils.trigger_rule").TriggerRule = _TriggerRule
    airflow.DAG = _FakeDAG
    return saved


def _restore_modules(saved):
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


@pytest.fixture()
def dag_module():
    saved = _install_airflow_stubs()
    try:
        spec = importlib.util.spec_from_file_location("agripulse_dag_test", str(DAG_PATH))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        _restore_modules(saved)


def test_dag_imports_and_has_expected_name(dag_module):
    assert dag_module.dag.dag_id == "agripulse_external_data_pipeline"
    assert dag_module.dag.schedule_interval == "0 4 * * *"


def _operators_by_task_id(dag_module):
    return {
        obj.task_id: obj
        for obj in vars(dag_module).values()
        if isinstance(obj, _FakeOperator)
    }


def test_dag_has_all_pipeline_tasks(dag_module):
    assert set(_operators_by_task_id(dag_module)) == EXPECTED_TASKS


def test_dag_dependency_chain(dag_module):
    ops = _operators_by_task_id(dag_module)
    ingest = {"ingest_weather", "ingest_market", "ingest_geo"}
    assert ops["bronze_to_silver"].upstream_task_ids == ingest
    assert ops["silver_to_gold"].upstream_task_ids == {"bronze_to_silver"}
    assert ops["monitor_pipeline_health"].upstream_task_ids == {"silver_to_gold"}
    assert ops["prediction_pipeline"].upstream_task_ids == {"monitor_pipeline_health"}


def test_dag_bash_commands_point_at_real_modules(dag_module):
    ops = _operators_by_task_id(dag_module)
    expected_script = {
        "ingest_weather": "src/ingestion/weather_api.py",
        "bronze_to_silver": "src/transform/bronze_to_silver.py",
        "silver_to_gold": "src/transform/silver_to_gold.py",
        "monitor_pipeline_health": "src/quality/monitor.py",
        "prediction_pipeline": "src/ml/prediction_pipeline.py",
    }
    for task_id, script in expected_script.items():
        assert script in ops[task_id].bash_command, task_id


def test_dag_monitor_uses_all_done_trigger(dag_module):
    monitor = _operators_by_task_id(dag_module)["monitor_pipeline_health"]
    assert monitor.trigger_rule == _TriggerRule.ALL_DONE