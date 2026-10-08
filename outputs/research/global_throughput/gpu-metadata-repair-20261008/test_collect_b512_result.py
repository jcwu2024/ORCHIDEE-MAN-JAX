"""Tiny saved-array B512 collector checks; no execution or compilation."""
import copy
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.dev import benchmark_global_batch as base
from scripts.dev.global_server_reference import finite_tolerance_failure

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = load("collector_b512_under_test", HERE / "collect_b512_result.py")
old = load("b512_tiny_fixture", HERE.parent / "gpu-sweetspot-preparation-20261006/test_collect_capacity_result.py")
approval = SimpleNamespace(job_id="99999999", node="synthetic-node", gpu_uuid="GPU-synthetic",
    driver_version="525.85.12", slurm_global_gpu_id="2", job_name="synthetic-b512")


def fixture(directory, tolerance=False):
    old.collector = SimpleNamespace(CONTRACT=collector.CONTRACT, JOB=approval.job_id, CODE_SHA=collector.CODE_SHA,
        sha=collector.sha)
    identity, _, envelope, reference = old.fixture(directory, tolerance)
    name = collector.STAGE
    envelope["name"] = name
    envelope["record"]["provenance"].update(chunk_size=512, chunks_per_collection=1)
    for gate in envelope["record"]["gates"]:
        gate["name"] = gate["name"].replace("capacity_B128", "capacity_B512")
    resources = envelope["record"]["provenance"]["server_resources"]
    resources["hostname"] = approval.node
    resources["Slurm"].update(SLURM_JOB_NODELIST=approval.node, SLURM_JOB_GPUS=approval.slurm_global_gpu_id)
    resources["Slurm"]["ORCHJAX_SLURM_ALLOC_TRES"] += ",node=1"
    envelope["record"]["execution_origin"]["environment"].update(hostname=approval.node,
        physical_gpu={"uuid": approval.gpu_uuid, "driver_version": approval.driver_version})
    return identity, name, envelope, reference


def check(directory, identity, name, envelope, reference, *, default_leaf_count=False):
    record = old.save(directory, envelope)
    kwargs = {} if default_leaf_count else {"leaf_count": 4}
    return collector.validate_stage(directory, name, identity, record, reference, base,
        finite_tolerance_failure, approval, **kwargs)


def test_complete_new_B512_keeps_full_collection_and_separate_cost_boundaries(tmp_path):
    result = check(tmp_path, *fixture(tmp_path))
    assert result["stage"] == collector.STAGE and result["batch_shape"] == 512
    assert len(result["actual_saved_output_reference_comparisons"]) == 512
    assert result["scientific_acceptance"] is False
    assert len(result["core"]["samples_seconds"]) == len(result["wrapper"]["samples_seconds"]) == 5
    assert result["startup"]["compile_seconds"] == 1 and result["checkpoint_IO"]["available"]


@pytest.mark.parametrize("fault", ["B256", "job", "code", "node", "gpu", "samples", "dtype", "days", "gates", "NaN", "default760"])
def test_wrong_or_incomplete_B512_evidence_is_withheld(tmp_path, fault):
    identity, name, envelope, reference = fixture(tmp_path)
    if fault == "B256":
        name = "capacity_B256_fixed512_v1"
        envelope["name"] = name
    elif fault == "job":
        envelope["record"]["provenance"]["server_resources"]["Slurm"]["SLURM_JOB_ID"] = "15643868"
    elif fault == "code":
        envelope["record"]["execution_origin"]["environment"]["git_head"] = collector.prior.CODE_SHA
    elif fault == "node":
        envelope["record"]["execution_origin"]["environment"]["hostname"] = "other-node"
    elif fault == "gpu":
        envelope["record"]["execution_origin"]["environment"]["physical_gpu"]["uuid"] = "other"
    elif fault == "samples":
        envelope["value"]["measurements"]["core"]["samples_seconds"] = [1] * 4
    elif fault == "dtype":
        state, daily = envelope["value"]["outputs"][0]
        envelope["value"]["outputs"] = ((state.astype("float32"), daily), envelope["value"]["outputs"][1])
    elif fault == "days":
        state, daily = envelope["value"]["outputs"][0]
        envelope["value"]["outputs"] = ((state, daily[:, :6]), envelope["value"]["outputs"][1])
    elif fault == "gates":
        envelope["record"]["gates"].pop()
    elif fault == "NaN":
        envelope["value"]["outputs"][0][0][0, 0] = float("nan")
    with pytest.raises(ValueError):
        check(tmp_path, identity, name, envelope, reference, default_leaf_count=fault == "default760")


def test_even_finite_tolerance_failed_stage_cannot_be_promoted(tmp_path):
    with pytest.raises(ValueError, match="tolerance-failed"):
        check(tmp_path, *fixture(tmp_path, tolerance=True))


@pytest.mark.parametrize("state", ["COMPLETED", "FAILED"])
def test_terminal_accounting_charges_parent_once_and_preserves_actual_failure(state):
    row = f"{approval.job_id}|{approval.job_name}|gnall|{state}|2700|4|cpu=4,gres/gpu=1,node=1|{approval.node}|0:0\n"
    result = collector.parent_accounting(row + row.replace(approval.job_id, approval.job_id + ".batch"), approval)
    assert result["state"] == state
    assert result["whole_job_cost"]["primary"]["CNY"] == pytest.approx(1.8)
    assert result["whole_job_cost"]["conservative"]["CNY"] == pytest.approx(1.86)


@pytest.mark.parametrize("fault", ["unfinished", "duplicate", "oldjob", "wall", "node", "allocation", "partition", "jobname", "nodecount", "schema"])
def test_accounting_rejects_unapproved_or_unfinished_parent(fault):
    row = f"{approval.job_id}|{approval.job_name}|gnall|COMPLETED|2700|4|cpu=4,gres/gpu=1,node=1|{approval.node}|0:0\n"
    if fault == "unfinished":
        row = row.replace("COMPLETED", "RUNNING")
    elif fault == "duplicate":
        row += row
    elif fault == "oldjob":
        row = row.replace(approval.job_id, "15643868")
    elif fault == "wall":
        row = row.replace("2700", "2701")
    elif fault == "node":
        row = row.replace(approval.node, "other-node")
    elif fault == "partition":
        row = row.replace("gnall", "gln01")
    elif fault == "jobname":
        row = row.replace(approval.job_name, "other-name")
    elif fault == "nodecount":
        row = row.replace("node=1", "node=2")
    elif fault == "schema":
        row = row.strip() + "|start|end\n"
    else:
        row = row.replace("cpu=4", "cpu=8")
    with pytest.raises(ValueError):
        collector.parent_accounting(row, approval)


def report():
    return {"execution_source_identity": collector.execution_identity(), "scientific_acceptance": False,
        "status": "completed_exploratory_capacity_screen", "completed_shapes": [512],
        "comparison_shapes_including_imported_baseline": [128, 256, 512],
        "conditional_512_decision": {"permitted": True, "compiler_memory_origin": {"shape": 256, "sha256": "sealed256"}}}


@pytest.mark.parametrize("fault", ["certificate", "incomplete", "oldB256", "admission", "memory"])
def test_report_certificate_completion_and_B256_memory_are_required(fault):
    value = report()
    identity = {"continuation_B256_sha256": "sealed256"}
    collector.report_checks(value, identity)
    if fault == "certificate":
        value["execution_source_identity"]["execution_source_compatibility"]["active"] = False
    elif fault == "incomplete":
        value["status"] = "failed"
    elif fault == "oldB256":
        value["completed_shapes"] = [256, 512]
    elif fault == "admission":
        value["conditional_512_decision"]["permitted"] = False
    else:
        value["conditional_512_decision"]["compiler_memory_origin"]["sha256"] = "wrong"
    with pytest.raises(ValueError):
        collector.report_checks(value, identity)


@pytest.mark.parametrize("size", [128, 256])
def test_imported_origins_revalidated_without_new_stage_or_relabel(tmp_path, size):
    origin = {"source_identity": {}, "timing": {"core": {"samples_seconds": [1]*5}},
        "provenance": {"server_resources": {"Slurm": {"SLURM_JOB_ID": "15600405" if size == 128 else "15643868"}}},
        "execution_origin": {"environment": {"loadavg": (1., 2., 3.)}}}
    recorded = copy.deepcopy(origin)
    recorded["execution_origin"]["environment"]["loadavg"] = [1., 2., 3.]
    calls = []
    def importer(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs["reference"] == "native-reference"
        return {}, origin
    capacity = SimpleNamespace(CONTRACT="old128", CONTINUATION_CONTRACT="old256",
        import_b128_baseline=importer, import_b256_baseline=importer)
    collector.validate_baseline(recorded, {}, tmp_path / "old", tmp_path / "new", size,
        capacity, base.jax, "native-reference")
    assert len(calls) == 1 and not (tmp_path / "new").exists()
    recorded["timing"]["core"]["samples_seconds"][0] = 99
    with pytest.raises(ValueError, match="original evidence mismatch"):
        collector.validate_baseline(recorded, {}, tmp_path / "old", tmp_path / "new", size,
            capacity, base.jax, "native-reference")


def test_approval_parameters_are_mandatory_and_no_job_was_invented():
    with pytest.raises(SystemExit):
        collector.parser().parse_args([])
    assert not hasattr(collector, "JOB") and not hasattr(collector, "NODE")


def stage_report(record):
    value = report()
    value.update(environment=copy.deepcopy(record["execution_origin"]["environment"]),
        server_resources=copy.deepcopy(record["provenance"]["server_resources"]),
        stages={collector.STAGE: copy.deepcopy(record)}, gates=copy.deepcopy(record["gates"]),
        timing={f"capacity_B512_{key}_seconds": record["startup"][key + "_seconds"] for key in ("lower", "compile")})
    return value


def test_denied_empty_stage_manifest_is_reported_without_promotion(tmp_path):
    value = report()
    value.update(status="failed", completed_shapes=[])
    value["conditional_512_decision"].update(permitted=False, reasons=["resource denial"])
    accepted, withheld = collector.collect_stage(tmp_path, {"continuation_B256_sha256": "sealed256"}, {},
        value, None, None, None, approval)
    assert accepted == [] and len(withheld) == 1


@pytest.mark.parametrize("fault", ["environment", "resources", "stage", "gates", "timing"])
def test_material_report_stage_disagreement_is_withheld(tmp_path, fault):
    identity, _, envelope, reference = fixture(tmp_path)
    record = old.save(tmp_path, envelope)
    value = stage_report(record)
    collector.report_stage_checks(value, record)
    if fault == "environment":
        value["environment"]["git_head"] = "wrong"
    elif fault == "resources":
        value["server_resources"]["allocated_CPU_request_count"] = 8
    elif fault == "stage":
        value["stages"][collector.STAGE]["sha256"] = "wrong"
    elif fault == "gates":
        value["gates"].pop()
    else:
        value["timing"]["capacity_B512_compile_seconds"] = 99
    accepted, withheld = collector.collect_stage(tmp_path, {**identity, "continuation_B256_sha256": "sealed256"},
        {collector.STAGE: record}, value, reference, base, finite_tolerance_failure, approval, leaf_count=4)
    assert accepted == [] and len(withheld) == 1


def test_dirty_code_checkout_stops_main_before_execution_imports(monkeypatch):
    args = SimpleNamespace(job_id=approval.job_id, code_dir=Path("declared-checkout"))
    monkeypatch.setattr(collector, "parser", lambda: SimpleNamespace(parse_args=lambda: args))
    commands = []
    def git_output(command, **kwargs):
        commands.append(command)
        return collector.CODE_SHA if command[1] == "rev-parse" else " M scripts/dev/benchmark_global_gpu_capacity.py\n"
    monkeypatch.setattr(collector.subprocess, "check_output", git_output)
    with pytest.raises(ValueError, match="clean before any execution imports"):
        collector.main()
    assert commands[-1] == ["git", "status", "--porcelain", "--untracked-files=all"]
    assert sys.dont_write_bytecode is True


def test_actual_git_dirty_scripts_rejected_at_pinned_HEAD_without_model_import(tmp_path):
    root = HERE.parents[3]
    objects = subprocess.check_output(["git", "rev-parse", "--git-path", "objects"], cwd=root, text=True).strip()
    objects = (root / objects).resolve()
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    alternatives = tmp_path / ".git/objects/info/alternates"
    alternatives.write_bytes((str(objects).replace("\\", "/") + "\n").encode())
    subprocess.run(["git", "update-ref", "HEAD", collector.CODE_SHA], cwd=tmp_path, check=True)
    dirty = tmp_path / "scripts/dev/benchmark_global_batch.py"
    dirty.parent.mkdir(parents=True)
    dirty.write_text("raise RuntimeError('this file must never be imported')\n")
    with pytest.raises(ValueError, match="clean before any execution imports"):
        collector.admit_code_checkout(tmp_path)
    command = [sys.executable, str(HERE / "collect_b512_result.py"), "--code-dir", str(tmp_path)]
    for name in ("stage-dir", "admission", "sacct-file", "b128-baseline-dir", "b256-baseline-dir", "output"):
        command += ["--" + name, str(tmp_path / "absent" / name)]
    command += ["--job-id", approval.job_id, "--job-name", approval.job_name, "--node", approval.node,
        "--gpu-uuid", approval.gpu_uuid, "--slurm-global-gpu-id", approval.slurm_global_gpu_id,
        "--driver-version", approval.driver_version, "--approved-nodes", "1", "--approved-tasks", "1",
        "--approved-cpus", "4", "--approved-gpus", "1", "--approved-wall-seconds", "2700"]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0 and "clean before any execution imports" in result.stderr
    assert "this file must never be imported" not in result.stderr
    assert not (tmp_path / "absent").exists()
