"""Saved-only paid T30 matched comparison; requires a fresh paid approval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time


ROOT = Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX")
CODE = ROOT / "runtime/checkouts/shared-split-width-v1"
OVERLAY = ROOT / "runtime/transfer/shared-latent-reader-20261004"
DIAG = ROOT / "runtime/outputs/diagnostics"
PREFIX = "latent-reader-T30-20261006-paid-matched"
SCOPE = "latent_reader_T30_saved_matched_paid_1200s_20261006"
HEAD = "c599236bcf57b745e9bafed678298010f4c3b58f"
BASE_SHA = "bb8674f3d2ceadb28df907346d18f6300dd87f46d09ff6010c795f72dbee6485"
TURNOVER_SHA = "f6240cf5ce64a47b9bbe0647b5e5715305e9a9230ea77f2bd9f42981ae35a93c"
WALL_SECONDS = 1200
CLEANUP_RESERVE_SECONDS = 120
PHASE_CAP_SECONDS = 1020
CALL_CAP_SECONDS = 100
RATE_CPU_CNY_PER_HOUR = 0.07
RATE_GPU_CNY_PER_HOUR = 2.20
ALLOCATED_CPUS = 8
ALLOCATED_GPUS = 1
WORST_CASE_CNY = 0.92

MANIFEST = OVERLAY / "latent-reader-T30-20261006.manifest.json"
PREPARED_CONTEXT = DIAG / "shared-expert-real-daily-context-c599236.pkl"
PARAMETERS = DIAG / "shared-latent-reader-20261004-T7-assembled-input.parameters.pkl"
FIELD_ADAPTER = ROOT / "runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py"
BACKBONE_ADAPTER = ROOT / "runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
TRANSFORMER = OVERLAY / "shared_transformer.py"
LATENT = OVERLAY / "shared_latent_reader.py"
TURNOVER = OVERLAY / "turnover_leaf_age_fall.guarded-source.py"
MATCHED_PROBE = OVERLAY / "probe_shared_latent_reader_T7_pair.py"
MATCHED_PROBE_SHA256 = "58c73984d64219b20a6ead6d165092969cdb3d67506f2701726ae1193b8b1968"
ORIGINAL_PREFLIGHT = DIAG / (
    "latent-reader-T30-20261006-compile-original-30610006-0000-4000-8000-000000000001.runtime-preflight.json"
)
CANDIDATE_PREFLIGHT = DIAG / (
    "latent-reader-T30-20261006-compile-candidate-30610006-0000-4000-8000-000000000002.runtime-preflight.json"
)
ORIGINAL = DIAG / (
    "latent-reader-T30-20261006-compile-original-30610006-0000-4000-8000-000000000001.executable.pkl"
)
CANDIDATE = DIAG / (
    "latent-reader-T30-20261006-compile-candidate-30610006-0000-4000-8000-000000000002.executable.pkl"
)
APPROVAL = OVERLAY / (PREFIX + "-approval.json")
LEDGER = DIAG / (PREFIX + ".gpu-budget.json")


class PaidResourceError(RuntimeError):
    """The allocated GPU or its release could not be verified safely."""


def worst_case_cost_cny() -> float:
    return (ALLOCATED_CPUS * RATE_CPU_CNY_PER_HOUR + ALLOCATED_GPUS * RATE_GPU_CNY_PER_HOUR) * WALL_SECONDS / 3600.0


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(binary: Path) -> Path:
    return binary.with_suffix(binary.suffix + ".identity.json")


def report_path(binary: Path) -> Path:
    return binary.with_name(binary.name.removesuffix(".executable.pkl") + ".json")


def expected_compile_identity(binary: Path, architecture: str, manifest: dict) -> dict:
    sidecar = sidecar_path(binary)
    if not binary.is_file():
        raise FileNotFoundError("saved-only graph missing; paid path cannot compile: " + str(binary))
    if not sidecar.is_file():
        raise FileNotFoundError("saved graph identity sidecar missing: " + str(sidecar))
    saved = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    identity = saved.get("identity")
    architecture_id = (
        "original" if architecture == "original" else "shared_latent32x384_reader64x2"
    )
    expected = {
        "architecture": architecture_id,
        "source_git_head": HEAD,
        "artifact_id": 742,
        "horizon": 30,
        "native_AD_argnums": [0, 1],
        "precision": "FP32_neural_FP64_physics",
        "turnover_base_sha256": BASE_SHA,
        "turnover_sha256": TURNOVER_SHA,
        "training_updates": 0,
        "jax_version": "0.4.38",
        "manifest_sha256": sha256_file(MANIFEST),
        "input_bundle_sha256": sha256_file(Path(manifest["input_bundle"])),
        "prepared_context_sha256": sha256_file(PREPARED_CONTEXT),
        "compiler_options": {
            "xla_gpu_enable_llvm_module_compilation_parallelism": True,
            "xla_gpu_force_compilation_parallelism": 4,
        },
    }
    expected_count, expected_leaves = (
        (64884481, 300) if architecture == "original" else (66123713, 355)
    )
    expected["parameter_count"] = expected_count
    expected["parameter_leaves"] = expected_leaves
    if saved.get("component") != "T30" or not isinstance(identity, dict):
        raise ValueError("saved graph sidecar is not a T30 identity object")
    for key, value in expected.items():
        if type(identity.get(key)) is not type(value) or identity.get(key) != value:
            raise ValueError(f"saved {architecture} T30 identity differs: {key}")
    if architecture == "original":
        provenance = identity.get("parameter_provenance", {})
        if provenance.get("completed_updates") != 0 or provenance.get("checkpoint") != manifest["original_checkpoint"]:
            raise ValueError("original T30 graph is not the declared update0 checkpoint")
    else:
        provenance = identity.get("parameter_provenance", {})
        if provenance.get("initialization") != "fresh_public_whole_tree" or provenance.get("artifact") != manifest["candidate_parameters"]:
            raise ValueError("candidate T30 graph is not the declared fresh candidate initialization")

    report = report_path(binary)
    if not report.is_file():
        raise FileNotFoundError("saved compile report missing: " + str(report))
    result = json.loads(report.read_text(encoding="utf-8-sig"))
    other = "candidate" if architecture == "original" else "original"
    if (result.get("component") != "T30"
            or result.get("requested_stage") != "compile-" + architecture
            or result.get("training_updates") != 0
            or result.get("status") not in {"passed", "numerical_failed"}
            or result.get("model_compiles", {}).get(architecture) != 1
            or result.get("model_compiles", {}).get(other) != 0
            or result.get("identities", {}).get(architecture) != identity
            or result.get("cold_measurements", {}).get(architecture, {}).get("executable") != str(binary)):
        raise ValueError("saved T30 compile report, identity or load-only graph path differs")
    if not result.get("results"):
        raise ValueError("saved T30 compile report has no preserved first full-AD result")
    return identity


def validate_runtime_preflights(original_path: Path, candidate_path: Path, jax_version: str) -> dict:
    receipts = []
    for path in (original_path, candidate_path):
        if not path.is_file():
            raise FileNotFoundError(f"passed T30 compile runtime preflight missing: {path}")
        receipt = json.loads(path.read_text(encoding="utf-8-sig"))
        if (receipt.get("status") != "passed" or receipt.get("passed") is not True
                or receipt.get("actual_backend") != "gpu" or receipt.get("jax_version") != jax_version
                or receipt.get("requested_backend") != "gpu"):
            raise ValueError("T30 compile runtime preflight is failed or incompatible")
        receipts.append(receipt)
    if receipts[0].get("devices") != receipts[1].get("devices"):
        raise ValueError("T30 compile runtime preflights do not share the same visible device identity")
    return receipts[1]


def validate_saved_assets(*, manifest_path=MANIFEST, original=ORIGINAL, candidate=CANDIDATE,
                          original_preflight=ORIGINAL_PREFLIGHT, candidate_preflight=CANDIDATE_PREFLIGHT):
    if manifest_path != MANIFEST or original != ORIGINAL or candidate != CANDIDATE:
        raise ValueError("only the declared T30 UUID0001/UUID0002 paths are admitted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    expected_manifest = {
        "schema_version": "shared_corrected_latent_reader_recurrent_pair_manifest_v1",
        "prepared_artifact_index": 0,
        "horizon": 30,
        "native_AD_argnums": [0, 1],
        "precision": "FP32_neural_FP64_physics",
        "turnover_base_sha256": BASE_SHA,
        "turnover_sha256": TURNOVER_SHA,
        "training_updates": 0,
        "T30_graph": True,
        "prepared_artifact_id": 742,
        "execution_authorization_is_external": True,
    }
    for key, value in expected_manifest.items():
        if type(manifest.get(key)) is not type(value) or manifest.get(key) != value:
            raise ValueError(f"prepared T30 manifest differs: {key}")
    for key in ("source_profile", "input_bundle", "original_checkpoint", "candidate_parameters", "historical_original_admission"):
        path = Path(manifest[key])
        if not path.is_file():
            raise FileNotFoundError(f"prepared T30 asset missing: {path}")
    for path in (PREPARED_CONTEXT, PARAMETERS, FIELD_ADAPTER, BACKBONE_ADAPTER, TRANSFORMER, LATENT, TURNOVER):
        if not path.is_file():
            raise FileNotFoundError(f"required prepared matched asset missing: {path}")
    if sha256_file(TURNOVER) != TURNOVER_SHA:
        raise ValueError("guarded turnover source pin differs")
    if not MATCHED_PROBE.is_file() or sha256_file(MATCHED_PROBE) != MATCHED_PROBE_SHA256:
        raise ValueError("matched T30 overlay probe is missing or its source pin differs")
    original_identity = expected_compile_identity(original, "original", manifest)
    candidate_identity = expected_compile_identity(candidate, "candidate", manifest)
    common = (
        "source_git_head", "manifest_sha256", "input_bundle_sha256", "prepared_context_sha256",
        "compiler_options", "neural_module_sha256", "scientific_functions", "historical_source_admission",
        "input_case_sha256", "artifact_id", "horizon", "method", "prepared_context_identity", "sources",
        "turnover_base_sha256", "turnover_sha256", "native_AD_argnums", "precision", "training_updates", "jax_version",
    )
    if any(original_identity.get(key) != candidate_identity.get(key) for key in common):
        raise ValueError("original and candidate saved graphs do not share the same T30 case/source identity")
    preflight_data = validate_runtime_preflights(original_preflight, candidate_preflight, original_identity["jax_version"])
    if original_identity.get("device_kind") != candidate_identity.get("device_kind"):
        raise ValueError("saved original/candidate executable device kinds differ")
    return {
        "manifest": manifest,
        "identities": {"original": original_identity, "candidate": candidate_identity},
        "preflight": preflight_data,
        "binary_sha256": {"original": sha256_file(original), "candidate": sha256_file(candidate)},
        "sidecar_sha256": {"original": sha256_file(sidecar_path(original)), "candidate": sha256_file(sidecar_path(candidate))},
        "matched_probe_sha256": MATCHED_PROBE_SHA256,
    }


def submit_command(node: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", node):
        raise ValueError("approved selected node name is invalid")
    return f"sbatch --nodelist={node} {OVERLAY}/slurm_shared_latent_reader_T30_matched_paid.sh"


def validate_approval(data: dict, assets: dict) -> str:
    if data.get("approved") is not True or data.get("approval_origin") != "explicit_human_user":
        raise ValueError("a fresh explicit human paid approval is required")
    node = str(data.get("selected_node", ""))
    expected = {
        "schema_version": "latent_T30_matched_paid_approval_v1",
        "approved": True,
        "approval_origin": "explicit_human_user",
        "funding_scope": SCOPE,
        "partition": "gnall",
        "nodes": 1,
        "tasks": 1,
        "allocated_CPUs": 8,
        "GPUs": 1,
        "memory_GiB": 64,
        "wall_seconds": WALL_SECONDS,
        "nonexclusive": True,
        "working_directory": CODE.as_posix(),
        "worst_case_CNY": WORST_CASE_CNY,
        "CPU_rate_CNY_per_hour": RATE_CPU_CNY_PER_HOUR,
        "GPU_rate_CNY_per_hour": RATE_GPU_CNY_PER_HOUR,
        "training_updates": 0,
        "original_compiles": 0,
        "candidate_compiles": 0,
        "saved_asset_sha256": {
            "original_binary": assets["binary_sha256"]["original"],
            "candidate_binary": assets["binary_sha256"]["candidate"],
            "original_identity_sidecar": assets["sidecar_sha256"]["original"],
            "candidate_identity_sidecar": assets["sidecar_sha256"]["candidate"],
        },
        "matched_probe_sha256": assets["matched_probe_sha256"],
        "submit_command": submit_command(node),
    }
    if not re.fullmatch(r"[A-Za-z0-9_-]+", node):
        raise ValueError("fresh paid approval must identify the selected gnall node")
    for key, value in expected.items():
        if type(data.get(key)) is not type(value) or data.get(key) != value:
            raise ValueError(f"explicit T30 paid approval differs: {key}")
    if not data.get("approval_note") or not data.get("approved_at_utc") or not data.get("idle_node_snapshot"):
        raise ValueError("human approval, timestamp and fresh idle-node snapshot are required")
    return node


def allocation_gpu() -> str:
    expected = {
        "SLURM_JOB_PARTITION": "gnall",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_NTASKS": "1",
        "SLURM_GPUS_ON_NODE": "1",
        "SLURM_MEM_PER_NODE": "65536",
    }
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("true execution requires a paid Slurm allocation")
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise ValueError("actual Slurm allocation differs: " + key)
    if os.environ.get("SLURM_JOB_NUM_NODES", os.environ.get("SLURM_NNODES")) != "1":
        raise ValueError("exactly one gnall node is required")
    node = os.environ.get("SLURMD_NODENAME", "")
    if not node:
        raise ValueError("Slurm did not report the actual node")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu or gpu.strip() != gpu:
        raise ValueError("exactly one scheduler-assigned visible GPU is required")
    return gpu


def assigned_gpu_snapshot(gpu: str) -> dict:
    query = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=10,
    ).strip()
    process_output = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid,used_gpu_memory,process_name", "--format=csv,noheader,nounits"],
        text=True, timeout=10,
    ).strip()
    fields = [part.strip() for part in query.split(",")]
    if len(fields) != 5:
        raise ValueError("ambiguous assigned-GPU observation")
    memory, utilization = int(fields[3]), int(fields[4])
    processes = []
    if process_output and "no running processes" not in process_output.lower():
        for row in process_output.splitlines():
            parts = [part.strip() for part in row.split(",", 2)]
            if len(parts) != 3 or not parts[0].isdigit():
                raise ValueError("ambiguous assigned-GPU compute-process list")
            processes.append({"pid": int(parts[0]), "used_gpu_memory": parts[1], "process_name": parts[2]})
    return {
        "epoch_seconds": time.time(),
        "scheduler_visible_gpu": gpu,
        "gpu_query": query,
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": processes,
        "idle": memory <= 100 and utilization == 0 and not processes,
    }


def classify_gpu_processes(snapshot: dict, own_pids: set[int], *, empty_classification: str = "external") -> dict:
    for item in snapshot.get("compute_processes", []):
        item["classification"] = (
            "owned_process_group" if item["pid"] in own_pids else empty_classification
        )
    return snapshot


def current_process_group_pids(pgid: int) -> set[int]:
    output = subprocess.check_output(["ps", "-eo", "pid=,pgid="], text=True, timeout=10)
    found = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit() and int(fields[1]) == pgid:
            found.add(int(fields[0]))
    return found


def cleanup_process_group(pgid: int | None, *, process=None, term_timeout: float = 20.0, kill_timeout: float = 5.0) -> dict:
    if pgid is None:
        return {"release_confirmed": True, "remaining_pids": [], "signals_sent": [], "errors": []}
    signals_sent = []
    errors = []

    def members():
        try:
            return current_process_group_pids(pgid)
        except Exception as error:
            errors.append({"type": type(error).__name__, "message": str(error)})
            return {pgid}

    remaining = members()
    if remaining:
        try:
            os.killpg(pgid, signal.SIGTERM)
            signals_sent.append("SIGTERM")
        except ProcessLookupError:
            pass
        except Exception as error:
            errors.append({"type": type(error).__name__, "message": str(error)})

        deadline = time.monotonic() + term_timeout
        while remaining and time.monotonic() < deadline:
            if process is not None:
                process.poll()
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
            remaining = members()
        if remaining:
            try:
                os.killpg(pgid, signal.SIGKILL)
                signals_sent.append("SIGKILL")
            except ProcessLookupError:
                pass
            except Exception as error:
                errors.append({"type": type(error).__name__, "message": str(error)})
            deadline = time.monotonic() + kill_timeout
            while remaining and time.monotonic() < deadline:
                if process is not None:
                    process.poll()
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
                remaining = members()
    return {
        "release_confirmed": not remaining and not errors,
        "remaining_pids": sorted(remaining),
        "signals_sent": signals_sent,
        "errors": errors,
    }


def launch_command(gpu: str, affinity: str, report: Path) -> list[str]:
    toolchain = ROOT / "runtime/toolchains/cuda-nvlink-12.6.85"
    cuda = Path("/usr/lib64/libcuda.so.1").resolve()
    nvml = Path("/usr/lib64/libnvidia-ml.so.1").resolve()
    binds = f"/WORK:/WORK,{ROOT}/runtime/data:{ROOT}/data:ro,{toolchain}:{toolchain}:ro,{cuda}:/usr/lib/x86_64-linux-gnu/libcuda.so.1,{nvml}:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    env = [
        "SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs",
        "SINGULARITYENV_CUDA_VISIBLE_DEVICES=" + gpu,
        "SINGULARITYENV_JAX_PLATFORMS=cuda,cpu",
        "SINGULARITYENV_JAX_ENABLE_X64=true",
        "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "SINGULARITYENV_ORCHIDEE_REPO_ROOT=" + str(CODE),
        "SINGULARITYENV_PYTHONPATH=" + str(CODE) + ":" + str(OVERLAY),
        "SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=" + str(ROOT / "runtime/cache/jax/orcjax_gpu" / PREFIX),
        "SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0",
        "SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0",
        "SINGULARITYENV_PREPEND_PATH=" + str(OVERLAY / "nvlink-bin") + ":" + str(toolchain / "bin"),
        "SINGULARITYENV_ORCHIDEE_NVLINK_STAGE=" + PREFIX,
        "SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1",
        "SINGULARITYENV_OPENBLAS_NUM_THREADS=1",
        "SINGULARITYENV_OMP_NUM_THREADS=8",
    ]
    python = ROOT / ".venvs/orcjax_gpu/bin/python"
    probe = MATCHED_PROBE
    args = [
        str(probe), "--stage", "matched", "--horizon", "30",
        "--manifest", str(MANIFEST), "--prepared-context", str(PREPARED_CONTEXT),
        "--parameters", str(PARAMETERS), "--field-adapter", str(FIELD_ADAPTER),
        "--backbone-adapter", str(BACKBONE_ADAPTER), "--transformer-module", str(TRANSFORMER),
        "--latent-module", str(LATENT), "--turnover-overlay", str(TURNOVER),
        "--runtime-preflight", str(CANDIDATE_PREFLIGHT), "--wall-seconds", str(PHASE_CAP_SECONDS),
        "--lower-seconds", "1", "--backend-seconds", "1", "--call-seconds", str(CALL_CAP_SECONDS),
        "--pairs", "3", "--force-compilation-parallelism", "4", "--output", str(report),
        "--original-executable", str(ORIGINAL), "--candidate-executable", str(CANDIDATE),
    ]
    return [
        "taskset", "-c", affinity, "env", "-u", "LD_LIBRARY_PATH", *env,
        "singularity", "exec", "--nv", "--bind", binds, "--pwd", str(CODE),
        "/apps/soft/sif/foundationpose", str(python), "-B", *args,
    ]


def dry_run() -> int:
    report = DIAG / (PREFIX + "-<SLURM_JOB_ID>.json")
    cmd = launch_command("<scheduler-assigned-GPU>", "<8 allocated CPU IDs>", report)
    print("PREPARED ONLY: no paid approval, allocation, GPU observation, graph load or model call")
    print("Resources: gnall, 1 node, 1 task, 8 CPUs, 1 V100, 64GiB, nonexclusive, 00:20:00")
    print(f"Cost ceiling: 8*0.07 + 1*2.20 = CNY2.76/hour; 2.76*(1200/3600) = CNY{worst_case_cost_cny():.2f}")
    print("Phase budget: setup/identity/idle checks <=60s + matched <=1020s + cleanup reserve120s = 1200s")
    print(f"Matched probe: {MATCHED_PROBE} sha256={MATCHED_PROBE_SHA256}; T30, UUID0001 original + UUID0002 candidate, 3 alternating pairs, 8 full-AD calls, call cap100s, compiles0/0, updates0")
    print("False/unapproved template must not execute; paid ledger is independent and never reads or resets the free15600-second ledger")
    print("External GPU PIDs are recorded as external and never terminated or attributed to this job")
    print("COMMAND " + shlex.join(cmd))
    return 0


def validate_approval_file(path: Path, assets: dict) -> str:
    if not path.is_file():
        raise FileNotFoundError("fresh paid approval receipt missing; no implicit authorization")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return validate_approval(data, assets)


def run(args) -> int:
    started = time.monotonic()
    if args.validate_assets:
        assets = validate_saved_assets()
        print(json.dumps({
            "status": "assets_validated",
            "component": "T30",
            "model_compiles": {"original": 0, "candidate": 0},
            "training_updates": 0,
            "saved_graph_sha256": assets["binary_sha256"],
            "matched_probe_sha256": assets["matched_probe_sha256"],
            "runtime_preflights": {
                "original": str(ORIGINAL_PREFLIGHT),
                "candidate": str(CANDIDATE_PREFLIGHT),
                "status": "passed",
                "jax_version": assets["preflight"]["jax_version"],
                "devices": assets["preflight"].get("devices"),
            },
        }, indent=2, sort_keys=True))
        return 0
    if args.dry_run:
        if args.approval != APPROVAL or args.ledger != LEDGER:
            raise ValueError("paid approval and independent paid ledger paths are fixed")
        return dry_run()
    if args.approval != APPROVAL or args.ledger != LEDGER:
        raise ValueError("paid approval and independent paid ledger paths are fixed")
    if args.ledger.exists():
        raise FileExistsError("independent paid ledger already exists; no reset or retry is inferred")
    assets = validate_saved_assets()
    approved_node = validate_approval_file(args.approval, assets)
    gpu = allocation_gpu()
    node = os.environ["SLURMD_NODENAME"]
    if node != approved_node:
        raise ValueError("actual allocated node differs from fresh approval")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < ALLOCATED_CPUS:
        raise ValueError("Slurm CPU affinity contains fewer than eight allocated CPUs")
    chosen_affinity = ",".join(str(value) for value in affinity[:ALLOCATED_CPUS])
    job_id = os.environ["SLURM_JOB_ID"]
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("invalid Slurm job id")
    report = DIAG / f"{PREFIX}-{job_id}.json"
    memory_log = DIAG / f"{PREFIX}-{job_id}.memory.csv"
    process_log = DIAG / f"{PREFIX}-{job_id}.processes.csv"
    if any(path.exists() for path in (report, memory_log, process_log)):
        raise FileExistsError("paid result/monitor output exists; no overwrite or retry")
    scheduler_snapshot = subprocess.check_output(["scontrol", "show", "job", job_id], text=True, timeout=10)
    if "TimeLimit=00:20:00" not in scheduler_snapshot:
        raise ValueError("actual Slurm wall differs from the 20-minute approved envelope")
    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError):
        load = None
    ledger = {
        "schema_version": "latent_T30_matched_paid_allocation_v1",
        "funding_scope": SCOPE,
        "approval_receipt": str(args.approval),
        "free_15600_second_grant_excluded": True,
        "historical_T7_paid_grants_excluded": True,
        "worst_case_CNY": WORST_CASE_CNY,
        "wall_seconds": WALL_SECONDS,
        "phase_cap_seconds": PHASE_CAP_SECONDS,
        "cleanup_reserve_seconds": CLEANUP_RESERVE_SECONDS,
        "SLURM_job_id": job_id,
        "node": node,
        "scheduler_allocation": scheduler_snapshot,
        "allocated_CPUs": ALLOCATED_CPUS,
        "available_affinity": affinity,
        "used_affinity": chosen_affinity,
        "GPUs": ALLOCATED_GPUS,
        "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS"),
        "CUDA_VISIBLE_DEVICES": gpu,
        "host_load_at_start": load,
        "original_saved_binary": str(ORIGINAL),
        "candidate_saved_binary": str(CANDIDATE),
        "model_compiles": {"original": 0, "candidate": 0},
        "training_updates": 0,
        "planned_full_AD_calls": 8,
        "planned_pairs": 3,
        "saved_asset_sha256": assets["binary_sha256"],
        "phases": [],
        "status": "starting",
    }
    initial = assigned_gpu_snapshot(gpu)
    classify_gpu_processes(initial, set(), empty_classification="external_before_launch")
    ledger["assigned_GPU_idle_checks"] = [initial]
    atomic_json(args.ledger, ledger)
    if not initial["idle"]:
        external = initial["compute_processes"]
        ledger.update(status="resource_failed", resource_failure="assigned GPU not idle before model process",
                      external_GPU_processes_before_start=external, own_process_group_started=False,
                      paid_model_calls=0)
        atomic_json(args.ledger, ledger)
        raise PaidResourceError("assigned GPU busy/unverified; external processes were recorded and left untouched")

    phase_budget = min(PHASE_CAP_SECONDS, int(WALL_SECONDS - CLEANUP_RESERVE_SECONDS - (time.monotonic() - started)))
    if phase_budget <= 0:
        ledger.update(status="resource_failed", resource_failure="setup exhausted cleanup reserve", paid_model_calls=0)
        atomic_json(args.ledger, ledger)
        raise TimeoutError("no matched-phase budget remains")
    command = launch_command(gpu, chosen_affinity, report)
    pgid = None
    process = None
    stop = threading.Event()
    samples = []
    monitor_errors = []
    monitor_lock = threading.Lock()
    phase = {"name": "saved-only-T30-matched", "cap_seconds": phase_budget, "call_cap_seconds": CALL_CAP_SECONDS,
             "command": command, "report": str(report), "status": "running"}
    ledger["phases"].append(phase)
    ledger.update(status="running", start_epoch=time.time(), matched_phase_cap_seconds=phase_budget)
    atomic_json(args.ledger, ledger)
    old_handlers = {}
    termination = {"signal": None}

    def on_signal(signum, frame):
        if termination["signal"] is None:
            termination["signal"] = signum

    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, on_signal)

    def sample_gpu():
        while not stop.wait(2.0):
            try:
                snapshot = assigned_gpu_snapshot(gpu)
                own = current_process_group_pids(pgid) if pgid is not None else set()
                classify_gpu_processes(snapshot, own)
                with monitor_lock:
                    samples.append(snapshot)
            except Exception as error:
                with monitor_lock:
                    monitor_errors.append({"epoch_seconds": time.time(), "type": type(error).__name__, "message": str(error)})

    monitor = threading.Thread(target=sample_gpu, name="paid-t30-gpu-monitor", daemon=True)
    exit_status = None
    try:
        monitor.start()
        began = time.monotonic()
        process = subprocess.Popen(command, cwd=CODE, start_new_session=True)
        pgid = process.pid
        phase["process_group_id"] = pgid
        atomic_json(args.ledger, ledger)
        while True:
            if termination["signal"] is not None:
                raise TimeoutError("Slurm termination requested: " + signal.Signals(termination["signal"]).name)
            remaining = phase_budget - (time.monotonic() - began)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, phase_budget)
            try:
                exit_status = process.wait(timeout=min(1.0, remaining))
                break
            except subprocess.TimeoutExpired:
                pass
        phase["status"] = "finished" if exit_status == 0 else "failed"
        phase["exit_status"] = exit_status
    except subprocess.TimeoutExpired:
        exit_status = 124
        phase.update(status="timed_out", exit_status=exit_status)
    except Exception as error:
        exit_status = 143 if termination["signal"] else 1
        phase.update(status="interrupted" if termination["signal"] else "failed",
                     exit_status=exit_status, error_type=type(error).__name__, error=str(error))
    finally:
        cleanup = cleanup_process_group(pgid, process=process)
        if not cleanup["release_confirmed"]:
            phase["cleanup_failed"] = True
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                phase["cleanup_failed"] = True
        stop.set()
        if monitor.ident is not None:
            monitor.join(timeout=5)
        try:
            final_gpu = assigned_gpu_snapshot(gpu)
        except Exception as error:
            final_gpu = {"observation_error_type": type(error).__name__, "observation_error": str(error)}
        own_pids = current_process_group_pids(pgid) if pgid is not None else set()
        gpu_pids = {item["pid"] for item in final_gpu.get("compute_processes", [])}
        own_gpu_pids = sorted(own_pids & gpu_pids)
        external_gpu_pids = sorted(gpu_pids - set(own_pids))
        release_confirmed = cleanup["release_confirmed"] and not own_pids
        elapsed = time.monotonic() - started
        if report.is_file():
            try:
                matched = json.loads(report.read_text(encoding="utf-8-sig"))
            except Exception as error:
                matched = {"status": "unreadable", "error_type": type(error).__name__, "error": str(error)}
        else:
            matched = None
        ledger.update(
            status=("measurement_complete" if matched and matched.get("status") in {"passed", "numerical_failed"}
                    and matched.get("model_compiles") == {"original": 0, "candidate": 0}
                    and matched.get("completed_recurrent_calls") == 8 and len(matched.get("pairs", [])) == 3
                    and matched.get("training_updates") == 0 else "failed"),
            matched_report=str(report) if report.exists() else None,
            matched_status=None if matched is None else matched.get("status"),
            matched_completed_recurrent_calls=None if matched is None else matched.get("completed_recurrent_calls"),
            matched_model_compiles=None if matched is None else matched.get("model_compiles"),
            matched_pairs=None if matched is None else len(matched.get("pairs", [])),
            helper_exit_status=exit_status,
            termination_signal=termination["signal"],
            occupied_helper_seconds=elapsed,
            runtime_within_wall=elapsed <= WALL_SECONDS,
            helper_cost_estimate_CNY=elapsed / 3600.0 * (ALLOCATED_CPUS * RATE_CPU_CNY_PER_HOUR + ALLOCATED_GPUS * RATE_GPU_CNY_PER_HOUR),
            cost_scope="helper elapsed estimate; scheduler charge pending; worst allocation ceiling CNY0.92",
            assigned_GPU_idle_checks=[initial, *samples],
            GPU_monitor_errors=monitor_errors,
            final_GPU_snapshot=final_gpu,
            own_process_group_id=pgid,
            own_process_group_release_confirmed=release_confirmed,
            own_process_group_cleanup=cleanup,
            own_GPU_processes_after_cleanup=own_gpu_pids,
            external_GPU_processes_after_cleanup=external_gpu_pids,
            assigned_GPU_idle_after_cleanup=final_gpu.get("idle") if isinstance(final_gpu, dict) else False,
            release_note="External PIDs are not helper children; they are recorded and never killed or used to negate this process-group release.",
        )
        if own_pids:
            ledger["status"] = "cleanup_failed"
        if elapsed > WALL_SECONDS:
            ledger["status"] = "wall_exceeded"
        if memory_log.exists() or process_log.exists():
            ledger["status"] = "failed"
        else:
            memory_log.parent.mkdir(parents=True, exist_ok=True)
            with memory_log.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["epoch_seconds", "scheduler_gpu", "memory_used_mib", "utilization_percent", "compute_pids"])
                for item in [initial, *samples]:
                    writer.writerow([item.get("epoch_seconds"), gpu, item.get("memory_used_mib"), item.get("utilization_percent"),
                                     ";".join(str(proc["pid"]) for proc in item.get("compute_processes", []))])
            with process_log.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["epoch_seconds", "pid", "used_gpu_memory", "process_name", "classification"])
                for item in [initial, *samples]:
                    if not item.get("compute_processes"):
                        writer.writerow([item.get("epoch_seconds"), "none", 0, "none", "none"])
                    for proc in item.get("compute_processes", []):
                        writer.writerow([item.get("epoch_seconds"), proc["pid"], proc["used_gpu_memory"], proc["process_name"],
                                         proc.get("classification", "unclassified")])
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        atomic_json(args.ledger, ledger)
    if ledger.get("status") == "measurement_complete":
        return 0
    return int(exit_status or 1)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--approval", type=Path, default=APPROVAL)
    result.add_argument("--ledger", type=Path, default=LEDGER)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--validate-assets", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    try:
        sys.exit(run(arguments))
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
