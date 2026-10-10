"""Prepare or run one independently budgeted paid B4 T7 follow-up."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX")
CODE = ROOT / "runtime/checkouts/shared-split-width-v1"
OVERLAY = ROOT / "runtime/transfer/shared-latent-reader-20261004"
DIAG = ROOT / "runtime/outputs/diagnostics"
PREFIX = os.environ.get("T7_B4_PAID_PREFIX", "latent-reader-T7-batch-20261009-paid")
SCOPE = "latent_reader_T7_B4_paid_followup_20261009"
NODE = "ibc13b03n07"
EXECUTION_NODE = "ibc13b03n05"
JOB_ID = "15666049"
REPLACEMENT_PREFIX = "latent-reader-T7-batch-20261009-paid.replacement"
REPLACEMENT_SCHEMA = "latent_T7_B4_paid_followup_replacement_v1"
REPLACEMENT_WALL_SECONDS = 10680
PRIOR_CHARGE_SECONDS = 120
AGGREGATE_WALL_SECONDS = 10800
AGGREGATE_COST_CNY = 8.28
ELIGIBLE_NODES = ("ibc13b03n01", "ibc13b03n04", "ibc13b03n05")
EXCLUDED_NODES = "ibc13b02n[01-03],ibc13b03n[02-03,06-07]"
SLURM_SCRIPT = OVERLAY / "slurm_shared_latent_T7_batch_paid.sh"
APPROVAL = OVERLAY / (PREFIX + ".approval.json")
PLACEMENT = OVERLAY / (PREFIX + ".placement.json")
LEDGER = DIAG / (PREFIX + ".gpu-budget.json")
CPU_READINESS = DIAG / "latent-reader-T7-batch-20261008.cpu-readiness.json"
RUNTIME_PREFLIGHT = DIAG / (PREFIX + ".runtime-preflight.json")
PROBE_REPORT = DIAG / (PREFIX + ".compile-and-compare.json")
BATCH_EXECUTABLE = DIAG / (PREFIX + ".executable.pkl")
B1_EXECUTABLE = DIAG / "latent-reader-T7-20261005-paid-replacement-compile-candidate.executable.pkl"
B1_MEMORY_REPORT = DIAG / "latent-reader-T7-20261005-paid-replacement-compile-candidate.json"
FREE_PREFIX = "latent-reader-T7-batch-20261008"
FREE_LEDGER = DIAG / (FREE_PREFIX + ".occupation.json")
FREE_REPORT = DIAG / (FREE_PREFIX + ".compile-and-compare.json")
FREE_ATTEMPT = DIAG / (FREE_PREFIX + ".executable.pkl.compile-attempt.json")
WALL_SECONDS = 10800
CLEANUP_RESERVE = 360
PREFLIGHT_CAP = 240
PROBE_CAP = 10200
PROBE_BACKEND_CAP = 9000
POST_COMPILE_RESERVE = 600
CPU_RATE = 0.07
GPU_RATE = 2.20
MEMORY_GIB = 64
EXPECTED_CPU = 8
PRIOR_CHARGE_CNY = round(PRIOR_CHARGE_SECONDS / 3600 * (EXPECTED_CPU * CPU_RATE + GPU_RATE), 3)
APPROVAL_SCHEMA = "latent_T7_B4_paid_followup_approval_v1"
REPLACEMENT_APPROVAL_SCHEMA = REPLACEMENT_SCHEMA
PLACEMENT_SCHEMA = "latent_T7_B4_paid_followup_placement_v1"
HEAD = "c599236bcf57b745e9bafed678298010f4c3b58f"


class PaidResourceError(RuntimeError):
    pass


class PaidTermination(RuntimeError):
    pass


class PaidDeadline(RuntimeError):
    pass


class PaidLaunchFailure(RuntimeError):
    pass


class PaidRuntimeFailure(RuntimeError):
    pass


class PaidNumericalFailure(RuntimeError):
    pass


class PaidIncomplete(RuntimeError):
    pass


def expected_cost(wall_seconds=WALL_SECONDS):
    hours = wall_seconds / 3600
    return round(EXPECTED_CPU * CPU_RATE * hours + GPU_RATE * hours, 2)


def submit_command(node=NODE):
    return f"sbatch --nodelist={node} --time=03:00:00 {cluster_path(SLURM_SCRIPT)}"


def cluster_path(path):
    return str(path).replace("\\", "/")


def replacement_submit_command():
    approval_path = OVERLAY / (REPLACEMENT_PREFIX + ".approval.json")
    stdout = DIAG / (REPLACEMENT_PREFIX + "-%j.out")
    stderr = DIAG / (REPLACEMENT_PREFIX + "-%j.err")
    exported = (f"ALL,T7_B4_PAID_APPROVAL={cluster_path(approval_path)},"
                f"T7_B4_PAID_PREFIX={REPLACEMENT_PREFIX}")
    return (f"sbatch --time=02:58:00 --exclude={EXCLUDED_NODES} --export={exported} "
            f"--output={cluster_path(stdout)} --error={cluster_path(stderr)} "
            f"{cluster_path(SLURM_SCRIPT)}")


def phase_caps(wall_seconds=WALL_SECONDS):
    return {"runtime_preflight": PREFLIGHT_CAP,
            "probe": min(PROBE_CAP, wall_seconds - PREFLIGHT_CAP - CLEANUP_RESERVE),
            "cleanup_reserve": CLEANUP_RESERVE}


def approval_template():
    return {
        "schema_version": APPROVAL_SCHEMA, "approved": False,
        "approval_origin": None, "approval_note": None, "approved_at": None,
        "partition": "gnall", "nodes": 1, "tasks": 1,
        "allocated_CPUs": EXPECTED_CPU, "GPUs": 1, "memory_GiB": MEMORY_GIB,
        "wall_seconds": WALL_SECONDS, "worst_case_CNY": expected_cost(),
        "CPU_rate_CNY_per_hour": CPU_RATE, "GPU_rate_CNY_per_hour": GPU_RATE,
        "nonexclusive": True, "working_directory": cluster_path(CODE),
        "funding_scope": SCOPE, "historical_and_free_grants_excluded": True,
        "free_ledger_path": cluster_path(FREE_LEDGER),
        "free_compile_attempt_path": cluster_path(FREE_ATTEMPT),
        "candidate_preexisting_B1_recompile_allowed": False,
        "new_B4_graph_compile_cap": 1, "paid_job_cap": 1,
        "B4_backend_seconds": PROBE_BACKEND_CAP,
        "post_compile_reserve_seconds": POST_COMPILE_RESERVE,
        "training_updates": 0, "T30_graph": False, "daily_graph": False,
        "selected_node": NODE, "submit_command": submit_command(NODE),
    }


def replacement_approval_template():
    data = approval_template()
    data.update(schema_version=REPLACEMENT_APPROVAL_SCHEMA,
        wall_seconds=REPLACEMENT_WALL_SECONDS,
        worst_case_CNY=expected_cost(REPLACEMENT_WALL_SECONDS),
        selected_node=None, eligible_nodes=list(ELIGIBLE_NODES),
        submit_command=replacement_submit_command(), prior_job_id=JOB_ID,
        prior_charge_seconds=PRIOR_CHARGE_SECONDS,
        prior_charge_ceiling_CNY=PRIOR_CHARGE_CNY,
        aggregate_wall_seconds=AGGREGATE_WALL_SECONDS,
        aggregate_cost_ceiling_CNY=AGGREGATE_COST_CNY)
    return data


def initial_paid_ledger_fields(approval, node, job_id):
    return {
        "schema_version": "latent_T7_B4_paid_followup_v1",
        "approved": True,
        "funding_scope": approval["funding_scope"],
        "SLURM_job_id": str(job_id),
        "node": node,
        "execution_node": node,
        "allocated_CPUs": EXPECTED_CPU,
        "GPUs": 1,
        "wall_seconds": approval["wall_seconds"],
        "probe_wall_seconds": phase_caps(approval["wall_seconds"])["probe"],
        "new_B4_graph_compile_cap": approval["new_B4_graph_compile_cap"],
        "new_B4_compile_attempt_count": 0,
    }


def validate_plan(data, *, approved_required=True, node=NODE):
    expected = {
        "schema_version": APPROVAL_SCHEMA,
        "approved": True if approved_required else False,
        "approval_origin": "explicit_human_user" if approved_required else None,
        "partition": "gnall", "nodes": 1, "tasks": 1, "allocated_CPUs": EXPECTED_CPU,
        "GPUs": 1, "memory_GiB": MEMORY_GIB, "wall_seconds": WALL_SECONDS,
        "worst_case_CNY": expected_cost(), "CPU_rate_CNY_per_hour": CPU_RATE,
        "GPU_rate_CNY_per_hour": GPU_RATE, "nonexclusive": True,
        "working_directory": cluster_path(CODE), "funding_scope": SCOPE,
        "historical_and_free_grants_excluded": True,
        "free_ledger_path": cluster_path(FREE_LEDGER),
        "free_compile_attempt_path": cluster_path(FREE_ATTEMPT),
        "candidate_preexisting_B1_recompile_allowed": False,
        "new_B4_graph_compile_cap": 1, "paid_job_cap": 1,
        "B4_backend_seconds": PROBE_BACKEND_CAP,
        "post_compile_reserve_seconds": POST_COMPILE_RESERVE,
        "training_updates": 0, "T30_graph": False, "daily_graph": False,
        "selected_node": node,
        "submit_command": submit_command(node),
    }
    if not isinstance(data, dict):
        raise ValueError("paid approval receipt must be a JSON object")
    if data.get("schema_version") == REPLACEMENT_APPROVAL_SCHEMA:
        expected.update(schema_version=REPLACEMENT_APPROVAL_SCHEMA,
            wall_seconds=REPLACEMENT_WALL_SECONDS,
            worst_case_CNY=expected_cost(REPLACEMENT_WALL_SECONDS),
            selected_node=None, eligible_nodes=list(ELIGIBLE_NODES),
            submit_command=replacement_submit_command(), prior_job_id=JOB_ID,
            prior_charge_seconds=PRIOR_CHARGE_SECONDS,
            prior_charge_ceiling_CNY=PRIOR_CHARGE_CNY,
            aggregate_wall_seconds=AGGREGATE_WALL_SECONDS,
            aggregate_cost_ceiling_CNY=AGGREGATE_COST_CNY)
    for key, value in expected.items():
        if type(data.get(key)) is not type(value) or data.get(key) != value:
            raise ValueError("paid follow-up approval differs: " + key)
    if approved_required and (not data.get("approved_at") or not data.get("approval_note")):
        raise ValueError("approved receipt requires human approval time and note")
    if not approved_required and (data.get("approved_at") or data.get("approval_note")):
        raise ValueError("unapproved template cannot contain approval metadata")


def placement_scope(approval):
    keys = ("partition", "nodes", "tasks", "allocated_CPUs", "GPUs", "memory_GiB",
            "wall_seconds", "worst_case_CNY", "CPU_rate_CNY_per_hour",
            "GPU_rate_CNY_per_hour", "nonexclusive", "working_directory", "funding_scope")
    return {key: approval[key] for key in keys}


def validate_placement(data, approval, *, approval_path=None, expected_job_id=None):
    if not isinstance(data, dict):
        raise ValueError("placement receipt must be a JSON object")
    expected = {
        "schema_version": PLACEMENT_SCHEMA,
        "job_id": JOB_ID,
        "requested_node": NODE,
        "execution_node": EXECUTION_NODE,
        "submit_command": submit_command(NODE),
        "resource_scope": placement_scope(approval),
    }
    if approval.get("selected_node") != NODE or approval.get("submit_command") != submit_command(NODE):
        raise ValueError("placement requires the original n07 approval and submit command")
    for key, value in expected.items():
        if type(data.get(key)) is not type(value) or data.get(key) != value:
            raise ValueError("paid placement receipt differs: " + key)
    if approval_path is not None:
        digest = sha256(approval_path)
        if data.get("approval_receipt_sha256") != digest:
            raise ValueError("paid placement receipt differs: approval_receipt_sha256")
    if expected_job_id is not None and str(expected_job_id) != data["job_id"]:
        raise ValueError("paid placement receipt job_id differs from this allocation")


def select_probe_stage(executable, attempt, sidecar):
    """Choose one compile, a saved-only comparison, or fail closed."""
    executable, attempt, sidecar = map(Path, (executable, attempt, sidecar))
    if executable.is_file():
        if not sidecar.is_file():
            raise ValueError("saved B4 executable lacks its identity sidecar")
        return "saved-compare"
    if sidecar.exists():
        raise ValueError("B4 identity exists without the saved executable")
    if attempt.exists():
        raise ValueError("B4 compile attempt exists without a saved executable; recompilation forbidden")
    return "compile-and-compare"


def verify_free_failure(report_path, ledger_path, attempt_path):
    """Bind this paid continuation to the failed free attempt without changing it."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    attempt = json.loads(Path(attempt_path).read_text(encoding="utf-8"))
    if (report.get("status") != "incomplete_timeout"
            or report.get("deadline_phase") != "backend_B4"
            or report.get("model_compiles") != {"B1": 0, "B4": 1}
            or report.get("B4_cold") is not None):
        raise ValueError("free B4 attempt evidence is not the expected unsaved backend timeout")
    if (ledger.get("limit_gpu_seconds") != 7200
            or ledger.get("independent_funding_scope") != "latent_B4_T7_20261008"
            or not ledger.get("sessions")):
        raise ValueError("free occupation ledger identity differs")
    if (attempt.get("full_B4_compile_attempts") != 1
            or not isinstance(attempt.get("mathematical_identity"), dict)):
        raise ValueError("free B4 attempt marker differs")
    return {"free_ledger": str(ledger_path), "free_report": str(report_path),
            "free_compile_attempt": str(attempt_path),
            "free_attempt_identity": attempt["mathematical_identity"],
            "free_ledger_sha256": sha256(ledger_path),
            "free_report_sha256": sha256(report_path),
            "free_attempt_sha256": sha256(attempt_path),
            "free_ledger_limit_preserved": 7200}


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_command(args, stage, output, wall_seconds):
    return [str(OVERLAY / "probe_shared_latent_T7_batch.py"),
        "--stage", stage, "--horizon", "7",
        "--manifest", str(OVERLAY / "latent-reader-T7-20261005.manifest.json"),
        "--prepared-context", str(DIAG / "shared-expert-real-daily-context-c599236.pkl"),
        "--parameters", str(DIAG / "shared-latent-reader-20261004-T7-assembled-input.parameters.pkl"),
        "--field-adapter", str(ROOT / "runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py"),
        "--backbone-adapter", str(ROOT / "runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"),
        "--transformer-module", str(OVERLAY / "shared_transformer.py"),
        "--latent-module", str(OVERLAY / "shared_latent_reader.py"),
        "--turnover-overlay", str(OVERLAY / "turnover_leaf_age_fall.guarded-source.py"),
        "--runtime-preflight", str(RUNTIME_PREFLIGHT),
        "--candidate-executable", str(B1_EXECUTABLE),
        "--batch-executable", str(BATCH_EXECUTABLE),
        "--training-prepared-manifest", str(ROOT / "runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json"),
        "--dataset-manifest", str(ROOT / "runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json"),
        "--cpu-readiness-report", str(CPU_READINESS),
        "--b1-memory-report", str(B1_MEMORY_REPORT),
        "--paid-ledger", str(getattr(args, "ledger", LEDGER)),
        "--wall-seconds", str(wall_seconds), "--lower-seconds", "600",
        "--backend-seconds", str(PROBE_BACKEND_CAP), "--load-seconds", "360",
        "--post-compile-reserve-seconds", str(POST_COMPILE_RESERVE),
        "--call-seconds", "180", "--pairs", "3", "--output", str(output)]


def container_base(gpu, affinity, output, *, dry_run=False):
    toolchain = ROOT / "runtime/toolchains/cuda-nvlink-12.6.85"
    cuda = Path("<resolved-libcuda>") if dry_run else Path("/usr/lib64/libcuda.so.1").resolve()
    nvml = Path("<resolved-libnvidia-ml>") if dry_run else Path("/usr/lib64/libnvidia-ml.so.1").resolve()
    binds = f"/WORK:/WORK,{ROOT}/runtime/data:{ROOT}/data:ro,{toolchain}:{toolchain}:ro,{cuda}:/usr/lib/x86_64-linux-gnu/libcuda.so.1,{nvml}:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    return ["taskset", "-c", affinity, "env", "-u", "LD_LIBRARY_PATH",
        "SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs",
        "SINGULARITYENV_CUDA_VISIBLE_DEVICES=" + gpu,
        "SINGULARITYENV_JAX_PLATFORMS=cuda,cpu", "SINGULARITYENV_JAX_ENABLE_X64=true",
        "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "SINGULARITYENV_ORCHIDEE_REPO_ROOT=" + str(CODE),
        "SINGULARITYENV_PYTHONPATH=" + str(CODE) + ":" + str(OVERLAY),
        "SINGULARITYENV_PREPEND_PATH=" + str(OVERLAY / "nvlink-bin") + ":" + str(ROOT / "runtime/toolchains/cuda-nvlink-12.6.85/bin"),
        "SINGULARITYENV_ORCHIDEE_NVLINK_LOG=" + str(output) + ".nvlink.log",
        "SINGULARITYENV_ORCHIDEE_NVLINK_STAGE=" + PREFIX,
        "SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1",
        "SINGULARITYENV_OPENBLAS_NUM_THREADS=1", "SINGULARITYENV_OMP_NUM_THREADS=8",
        "singularity", "exec", "--nv", "--bind", binds, "--pwd", str(CODE),
        "/apps/soft/sif/foundationpose", str(ROOT / ".venvs/orcjax_gpu/bin/python"), "-B"]


def phase_command(command, *, gpu, memory_path):
    return ["bash", str(OVERLAY / "shared_latent_T7_batch_paid_memory.sh"),
        str(gpu), str(memory_path), "--"] + list(command)


def summarize_probe(report, data, *, executable, sidecar):
    pairs = report.get("pairs", [])
    compile_counts = report.get("model_compiles")
    status = report.get("status")
    data.update(probe_status=status, probe_error_type=report.get("error_type"),
        probe_pairs_completed=len(pairs) if isinstance(pairs, list) else None,
        probe_pairs_required=3,
        probe_pairs_complete=isinstance(pairs, list) and len(pairs) >= 3,
        probe_model_compiles=compile_counts,
        probe_model_saved=Path(executable).is_file() and Path(sidecar).is_file())
    if status == "numerical_failed":
        return "numerical_failed"
    if status in {"incomplete", "incomplete_timeout", "timeout"}:
        return "incomplete"
    if status == "passed":
        return "passed"
    return "runtime_failed"


def is_resource_error(error, paid_helpers=None):
    types = [PaidResourceError]
    external_type = getattr(paid_helpers, "PaidResourceError", None)
    if isinstance(external_type, type) and external_type not in types:
        types.append(external_type)
    return isinstance(error, tuple(types))


def import_paid_helpers():
    path = OVERLAY / "shared_latent_T7_paid_budget.py"
    spec = importlib.util.spec_from_file_location("shared_latent_T7_paid_budget_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError("existing paid helper module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def reserve_ledger(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_allocation(node, *, allowed_nodes=None, placement=None, approval=None, approval_path=None):
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise ValueError("paid execution requires one actual Slurm allocation")
    expected = {"SLURM_JOB_PARTITION": "gnall", "SLURM_CPUS_PER_TASK": "8",
        "SLURM_NTASKS": "1", "SLURM_GPUS_ON_NODE": "1", "SLURM_MEM_PER_NODE": "65536"}
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise ValueError("Slurm resource mismatch: " + key)
    if os.environ.get("SLURM_JOB_NUM_NODES", os.environ.get("SLURM_NNODES")) != "1":
        raise ValueError("one node required")
    actual_node = os.environ.get("SLURMD_NODENAME", "")
    expected_node = node
    if placement is not None:
        if approval is None:
            raise ValueError("paid placement receipt requires its approval receipt")
        validate_placement(placement, approval, approval_path=approval_path,
                           expected_job_id=job_id)
        expected_node = placement["execution_node"]
    if allowed_nodes is not None:
        if allowed_nodes != list(ELIGIBLE_NODES) or actual_node not in allowed_nodes:
            raise ValueError("actual node is outside the approved eligible node set")
    elif actual_node != expected_node:
        raise ValueError("actual node differs from approved placement")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu or gpu.strip() != gpu:
        raise ValueError("one scheduler-assigned physical GPU required")
    return actual_node, gpu


def resource_snapshot(gpu, node):
    return {"node": node, "host_load": list(os.getloadavg()),
        "SLURM_job_id": os.environ.get("SLURM_JOB_ID"),
        "allocated_CPUs": int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
        "available_affinity": sorted(os.sched_getaffinity(0)),
        "visible_GPU": gpu,
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "-i", gpu], text=True, timeout=15),
        "compute_GPU_processes": subprocess.check_output(["nvidia-smi", "-i", gpu,
            "--query-compute-apps=pid,used_gpu_memory,process_name", "--format=csv,noheader,nounits"],
            text=True, timeout=15).strip()}


@contextmanager
def signals(state):
    def stop(signum, _frame):
        if state["signal"] is None:
            state["signal"] = signum
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_phase(command, *, stage, cap, gpu, affinity, ledger, data, paid_helpers, started, termination):
    if termination["signal"] is not None:
        raise PaidTermination(signal.Signals(termination["signal"]).name)
    allowed = data["wall_seconds"] - CLEANUP_RESERVE - (time.monotonic() - started)
    cap = min(cap, int(allowed))
    if cap <= 0:
        raise PaidDeadline("cleanup reserve reached before " + stage)
    memory = DIAG / (PREFIX + "." + stage + ".memory.csv")
    row = {"stage": stage, "cap_seconds": cap, "status": "running",
        "started_epoch": time.time(), "host_load": list(os.getloadavg()),
        "GPU_snapshot": subprocess.check_output(["nvidia-smi", "-i", gpu], text=True, timeout=15)}
    data.setdefault("phases", []).append(row)
    atomic(ledger, data)
    sampled = phase_command(command, gpu=gpu, memory_path=memory)
    before = time.monotonic()
    try:
        process = subprocess.Popen(sampled, cwd=CODE, start_new_session=True)
    except OSError as error:
        row.update(status="launch_failed", launch_error_type=type(error).__name__,
            launch_error=str(error), ended_epoch=time.time())
        atomic(ledger, data)
        raise PaidLaunchFailure(stage + " helper could not start") from error
    row["pid"] = process.pid
    row["memory_receipt"] = str(memory)
    row["process_receipt"] = str(memory) + ".processes.csv"
    atomic(ledger, data)
    status = None
    try:
        while True:
            if termination["signal"] is not None:
                raise PaidTermination(signal.Signals(termination["signal"]).name)
            remaining = cap - (time.monotonic() - before)
            if remaining <= 0:
                raise PaidDeadline(stage + " phase cap reached")
            try:
                status = process.wait(timeout=min(1.0, remaining))
                break
            except subprocess.TimeoutExpired:
                pass
    finally:
        paid_helpers.shutdown_phase(process)
        row.update(status="finished" if process.poll() is not None else "cleanup_failed",
            child_release_confirmed=process.poll() is not None,
            exit_status=status if status is not None else process.poll(),
            occupied_seconds=time.monotonic() - before,
            ended_epoch=time.time())
        atomic(ledger, data)
    if status:
        raise PaidRuntimeFailure(stage + " exited " + str(status))
    if stage == "runtime-preflight":
        report = json.loads(Path(data["runtime_preflight_path"]).read_text(encoding="utf-8"))
        if report.get("status") != "passed" or report.get("passed") is not True or report.get("actual_backend") != "gpu":
            raise PaidRuntimeFailure("fresh paid runtime preflight did not pass on GPU")
    return row


def dry_run(args):
    receipt = None
    if args.approval.is_file():
        receipt = json.loads(args.approval.read_text(encoding="utf-8"))
        if receipt.get("approved") is True:
            validate_plan(receipt, approved_required=True)
        else:
            validate_plan(receipt, approved_required=False)
    placement = None
    if args.placement.is_file():
        if receipt is None:
            raise ValueError("placement receipt requires its approved approval receipt")
        placement = json.loads(args.placement.read_text(encoding="utf-8"))
        validate_placement(placement, receipt, approval_path=args.approval)
    wall_seconds = receipt.get("wall_seconds", WALL_SECONDS) if receipt else WALL_SECONDS
    cost = receipt.get("worst_case_CNY", expected_cost(wall_seconds)) if receipt else expected_cost()
    print("PREPARED ONLY: no approval, submission, GPU initialization, or job execution")
    print(f"Partition gnall; 1 node, 1 task, 8 CPU, 1 V100, 64 GiB, nonexclusive; wall {wall_seconds}s")
    hours = wall_seconds / 3600
    print(f"Worst-case cost: CPU 8*0.07*{hours:g}h + GPU 1*2.20*{hours:g}h = CNY{cost:.2f}")
    print("Working directory: " + CODE.as_posix())
    target_node = placement["execution_node"] if placement else NODE
    if receipt and receipt.get("eligible_nodes"):
        print("Eligible nodes: " + ",".join(receipt["eligible_nodes"]))
        print("Execution node: selected by Slurm from the eligible set")
    else:
        print("Requested node: " + NODE)
        print("Execution node: " + target_node)
    if placement:
        print("Placement receipt: " + str(args.placement))
    print("Probe budget: up to 150 min backend; at least 10 min post-compile reserve; 3 alternating pairs; completion is not guaranteed")
    print("Independent paid prefix/ledger: " + PREFIX + " / " + LEDGER.as_posix())
    print("Preserved free ledger and compile-attempt: " + FREE_LEDGER.as_posix() + " ; " + FREE_ATTEMPT.as_posix())
    print("Submission command: " + (receipt.get("submit_command", submit_command()) if receipt else submit_command()))
    print("Approval receipt present: " + str(args.approval.is_file()))
    print("Approved: " + str(bool(receipt and receipt.get("approved") is True)))
    dry_gpu, dry_affinity = "0", "0-7"
    preflight = container_base(dry_gpu, dry_affinity, RUNTIME_PREFLIGHT, dry_run=True)
    preflight += [str(CODE / "scripts/hpc/check_daily_training_runtime.py"),
        "--backend", "gpu", "--output", str(RUNTIME_PREFLIGHT)]
    preflight_memory = DIAG / (PREFIX + ".runtime-preflight.memory.csv")
    print("Runtime-preflight Popen argv: " + shlex.join(phase_command(preflight,
        gpu=dry_gpu, memory_path=preflight_memory)))
    probe_window = phase_caps(wall_seconds)["probe"]
    probe = container_base(dry_gpu, dry_affinity, PROBE_REPORT, dry_run=True)
    probe += candidate_command(args, "compile-and-compare", PROBE_REPORT, probe_window)
    probe_memory = DIAG / (PREFIX + ".probe.memory.csv")
    print("Probe Popen argv: " + shlex.join(phase_command(probe,
        gpu=dry_gpu, memory_path=probe_memory)))


def run(args):
    started = time.monotonic()
    if args.dry_run:
        dry_run(args)
        return 0
    approval = json.loads(args.approval.read_text(encoding="utf-8"))
    validate_plan(approval, approved_required=True,
                  node=approval.get("selected_node") or NODE)
    placement = None
    if args.placement.is_file():
        placement = json.loads(args.placement.read_text(encoding="utf-8"))
    node, gpu = validate_allocation(approval.get("selected_node") or NODE,
        allowed_nodes=approval.get("eligible_nodes"), placement=placement,
        approval=approval if placement is not None else None,
        approval_path=args.approval if placement is not None else None)
    if args.ledger.exists():
        raise FileExistsError("paid ledger exists; one new paid job only, no reset or retry")
    if not CODE.is_dir() or not OVERLAY.is_dir() or not DIAG.is_dir():
        raise FileNotFoundError("declared project runtime directory missing")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE, text=True).strip() != HEAD:
        raise ValueError("declared immutable code checkout differs")
    free_evidence = verify_free_failure(FREE_REPORT, FREE_LEDGER, FREE_ATTEMPT)
    for path in (CPU_READINESS, B1_EXECUTABLE,
                 B1_EXECUTABLE.with_suffix(B1_EXECUTABLE.suffix + ".identity.json"),
                 B1_MEMORY_REPORT,
                 OVERLAY / "probe_shared_latent_T7_batch.py",
                 OVERLAY / "shared_latent_T7_batch_ad.py",
                 OVERLAY / "shared_latent_T7_batch_assets.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if CPU_READINESS.stat().st_size <= 0:
        raise ValueError("CPU readiness report is empty")
    attempt = BATCH_EXECUTABLE.with_suffix(BATCH_EXECUTABLE.suffix + ".compile-attempt.json")
    sidecar = BATCH_EXECUTABLE.with_suffix(BATCH_EXECUTABLE.suffix + ".identity.json")
    stage = select_probe_stage(BATCH_EXECUTABLE, attempt, sidecar)
    if stage == "compile-and-compare" and not args.allow_new_compile:
        raise ValueError("new B4 compile requires the explicit approved one-graph flag")
    if stage == "saved-compare" and args.allow_new_compile:
        # A saved graph always selects load-only, regardless of the requested approval scope.
        pass
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < EXPECTED_CPU:
        raise PaidResourceError("allocated process affinity exposes fewer than 8 CPUs")
    chosen = ",".join(str(cpu) for cpu in affinity[:EXPECTED_CPU])
    if args.ledger.exists():
        raise FileExistsError("paid ledger appeared concurrently; refusing a second run")
    for output in (RUNTIME_PREFLIGHT, PROBE_REPORT):
        if output.exists():
            raise FileExistsError("paid output already exists: " + str(output))
    data = {**initial_paid_ledger_fields(approval, node, os.environ["SLURM_JOB_ID"]),
        "status": "running", "approval_receipt": str(args.approval),
        "approval_receipt_sha256": sha256(args.approval),
        "placement_receipt": str(args.placement) if placement is not None else None,
        "placement_receipt_sha256": sha256(args.placement) if placement is not None else None,
        "requested_node": approval["selected_node"],
        "eligible_nodes": approval.get("eligible_nodes"),
        "independent_of_free_7200s_ledger": True, "free_evidence": free_evidence,
        "available_affinity": affinity, "used_affinity": chosen,
        "CUDA_VISIBLE_DEVICES": gpu, "host_load": list(os.getloadavg()),
        "cost_ceiling_CNY": approval["worst_case_CNY"],
        "runtime_preflight_path": str(RUNTIME_PREFLIGHT),
        "timing_basis": "runner_entry_monotonic; scheduler launch and epilogue excluded",
        "new_B4_compile_attempt_count": int(attempt.exists()),
        "probe_stage": stage,
        "candidate_B1_executable": str(B1_EXECUTABLE), "candidate_B1_recompiles": 0,
        "phase_caps_seconds": phase_caps(approval["wall_seconds"]),
        "backend_seconds_cap": PROBE_BACKEND_CAP,
        "post_compile_reserve_seconds": POST_COMPILE_RESERVE,
        "probe_report_path": str(PROBE_REPORT),
        "optimizer_updates": 0, "T30_graph": False, "daily_graph": False,
        "phases": []}
    initial_resources = resource_snapshot(gpu, node)
    data["initial_resources"] = initial_resources
    reserve_ledger(args.ledger, data)
    paid_helpers = import_paid_helpers()
    termination = {"signal": None}
    final_status = "failed"
    with signals(termination):
        try:
            idle = paid_helpers.assigned_gpu_idle(gpu, "runtime-preflight", args.ledger, data)
            if not idle.get("passed"):
                raise PaidResourceError("assigned physical GPU is not idle before preflight")
            atomic(args.ledger, data)
            base = container_base(gpu, chosen, RUNTIME_PREFLIGHT)
            preflight_command = base + [str(CODE / "scripts/hpc/check_daily_training_runtime.py"),
                "--backend", "gpu", "--output", str(RUNTIME_PREFLIGHT)]
            run_phase(preflight_command, stage="runtime-preflight", cap=PREFLIGHT_CAP,
                gpu=gpu, affinity=chosen, ledger=args.ledger, data=data,
                paid_helpers=paid_helpers, started=started, termination=termination)
            idle = paid_helpers.assigned_gpu_idle(gpu, "B4_model", args.ledger, data)
            if not idle.get("passed"):
                raise PaidResourceError("assigned physical GPU is not idle before B4 model work")
            atomic(args.ledger, data)
            available = approval["wall_seconds"] - CLEANUP_RESERVE - (time.monotonic() - started)
            probe_wall = min(phase_caps(approval["wall_seconds"])["probe"], int(available))
            if probe_wall <= POST_COMPILE_RESERVE + PROBE_BACKEND_CAP:
                raise PaidDeadline("less than the planned backend plus post-compile reserve remains")
            output = PROBE_REPORT
            base = container_base(gpu, chosen, output)
            cmd = base + candidate_command(args, stage, output, probe_wall)
            data["probe_stage"] = stage
            data["probe_wall_seconds"] = probe_wall
            atomic(args.ledger, data)
            run_phase(cmd, stage="probe", cap=probe_wall,
                gpu=gpu, affinity=chosen, ledger=args.ledger, data=data,
                paid_helpers=paid_helpers, started=started, termination=termination)
            report = json.loads(PROBE_REPORT.read_text(encoding="utf-8"))
            outcome = summarize_probe(report, data, executable=BATCH_EXECUTABLE,
                sidecar=sidecar)
            atomic(args.ledger, data)
            if outcome == "numerical_failed":
                raise PaidNumericalFailure("strict full-result numerical/state gates failed")
            if outcome == "incomplete":
                raise PaidIncomplete("inner B4 probe reported an incomplete timeout")
            if outcome != "passed":
                raise PaidRuntimeFailure("B4 probe ended with status " + str(report.get("status")))
            if report.get("model_compiles") != {"B1": 0, "B4": (1 if stage == "compile-and-compare" else 0)}:
                raise PaidRuntimeFailure("model compile count differs from selected one-shot stage")
            if len(report.get("pairs", [])) < 3:
                raise PaidRuntimeFailure("three alternating full-result pairs are required")
            final_status = "passed"
        except Exception as error:
            if PROBE_REPORT.is_file() and "probe_status" not in data:
                try:
                    partial_report = json.loads(PROBE_REPORT.read_text(encoding="utf-8"))
                    summarize_probe(partial_report, data, executable=BATCH_EXECUTABLE,
                        sidecar=sidecar)
                except (OSError, ValueError, TypeError):
                    pass
            probe_status = data.get("probe_status")
            failure_status = ("numerical_failed" if isinstance(error, PaidNumericalFailure)
                or probe_status == "numerical_failed"
                else "incomplete" if isinstance(error, (PaidDeadline, PaidIncomplete, subprocess.TimeoutExpired))
                or probe_status in {"incomplete", "incomplete_timeout", "timeout"}
                else "terminated" if isinstance(error, PaidTermination)
                else "resource_failed" if is_resource_error(error, paid_helpers)
                else "launch_failed" if isinstance(error, PaidLaunchFailure)
                else "runtime_failed" if isinstance(error, PaidRuntimeFailure)
                else "failed")
            data.update(status=failure_status, error_type=type(error).__name__, error=str(error),
                termination_signal=termination["signal"])
            raise
        finally:
            try:
                snapshot = subprocess.check_output(["nvidia-smi", "-i", gpu], text=True, timeout=15)
            except Exception as error:
                snapshot = {"unavailable": type(error).__name__, "message": str(error)}
            for path, digest_key in ((FREE_LEDGER, "free_ledger_unchanged"),
                                     (FREE_ATTEMPT, "free_attempt_unchanged"),
                                     (FREE_REPORT, "free_report_unchanged")):
                data[digest_key] = path.is_file() and sha256(path) == free_evidence[digest_key.replace("_unchanged", "_sha256")]
            if not all(data.get(key) is True for key in ("free_ledger_unchanged", "free_attempt_unchanged", "free_report_unchanged")):
                data["status"] = "preservation_conflict"
            elapsed = time.monotonic() - started
            helper_occupied = sum(float(phase.get("occupied_seconds", 0.0))
                for phase in data.get("phases", []))
            final_ledger_status = ("preservation_conflict"
                if data.get("status") == "preservation_conflict"
                else final_status if final_status == "passed" else data.get("status", "failed"))
            data.update(status=final_ledger_status,
                runner_elapsed_seconds=elapsed,
                scheduler_billed_seconds_lower_bound=elapsed,
                helper_occupied_seconds=helper_occupied,
                helper_GPU_seconds=helper_occupied,
                estimated_cost_CNY=round(elapsed / 3600 * (EXPECTED_CPU * CPU_RATE + GPU_RATE), 4),
                final_GPU_snapshot=snapshot, original_B1_recompiled=False,
                new_B4_compile_attempt_count=int(attempt.exists()),
                within_wall=elapsed <= data["wall_seconds"])
            atomic(args.ledger, data)
    return 0 if final_status == "passed" else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--approval", type=Path, default=APPROVAL)
    p.add_argument("--placement", type=Path, default=PLACEMENT)
    p.add_argument("--ledger", type=Path, default=LEDGER)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-new-compile", action="store_true",
        help="must be present in the approved one-job configuration; compile cap remains one")
    return p


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
