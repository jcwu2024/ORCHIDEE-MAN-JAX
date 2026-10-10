"""Prepared-only paid T7 fallback; explicit approval and Slurm are mandatory.

Independent allocation ledger, unchanged scientific manifest/graph identity.
Load the original success; compile the candidate at most once globally.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX")
CODE = ROOT/"runtime/checkouts/shared-split-width-v1"
OVERLAY = ROOT/"runtime/transfer/shared-latent-reader-20261004"
DIAG = ROOT/"runtime/outputs/diagnostics"
PREFIX = "latent-reader-T7-20261005"
SCOPE = "latent_reader_T7_paid_fallback_2h_552_20261005"
CAPS = {"runtime-preflight": 180, "candidate": 6240, "matched": 600, "setup_margin": 60, "cleanup": 120}
WALL_SECONDS = 7200
REPLACEMENT_SCOPE = 'latent_reader_T7_paid_replacement_7140_552_total_20261006'
REPLACEMENT_PREFIX = PREFIX+'-paid-replacement'
HEAD = "c599236bcf57b745e9bafed678298010f4c3b58f"


class PaidResourceError(RuntimeError):
    """Assigned-card availability failed; no other user's process is touched."""


class PaidTermination(RuntimeError):
    """Sticky signal request requiring bounded phase cleanup and settlement."""


def shutdown_phase(process):
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    try: process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait(timeout=5)
    # Clear any remaining descendants in this helper-owned session only.
    try: os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError: pass


def posix_join(values):
    """Render Linux commands faithfully when preparing them on Windows."""
    return shlex.join([value.replace("\\", "/") for value in values])


def atomic(path, value):
    temporary = path.with_name("."+path.name+f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def assigned_gpu_idle(gpu, stage, ledger, data):
    started = time.monotonic()
    wait_cap = 20
    while True:
        receipt = assigned_gpu_idle_observation(gpu, stage, ledger, data, deadline=started+wait_cap)
        if receipt['passed']:
            return receipt
        # A completed own preflight can leave a utilization sampling tail.
        # Wait only with no CUDA PID and low memory; the idle thresholds stay exact.
        remaining = wait_cap-(time.monotonic()-started)
        transient = ('observation_error' not in receipt and not receipt.get('compute_PID_query')
                     and 0 <= receipt.get('memory_used_mib', 101) <= 100
                     and receipt.get('utilization_percent', 0) > 0)
        receipt.update(idle_wait_cap_seconds=wait_cap,
                       idle_wait_elapsed_seconds=time.monotonic()-started,
                       waiting_for_utilization_zero=bool(transient and remaining > 0))
        atomic(ledger, data)
        if not transient or remaining <= 0:
            raise PaidResourceError('assigned physical GPU remains occupied or unverified before '+stage+' after bounded idle checks; no model work allowed')
        time.sleep(min(2, remaining))


def assigned_gpu_idle_observation(gpu, stage, ledger, data, *, deadline=None):
    receipt = {"stage": stage, "physical_GPU": gpu, "epoch_seconds": time.time(),
        "scope": "assigned-card total only; not own architecture memory or 24GiB admission",
        "thresholds": {"memory_used_mib_max": 100, "utilization_percent": 0, "compute_processes": 0}}
    data.setdefault("assigned_GPU_idle_checks", []).append(receipt)
    def query_timeout():
        remaining = 10 if deadline is None else deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('whole assigned-card idle-check deadline reached')
        return min(10, remaining)
    try:
        receipt["gpu_query"] = subprocess.check_output(["nvidia-smi", "-i", gpu,
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=query_timeout()).strip()
        receipt["compute_PID_query"] = subprocess.check_output(["nvidia-smi", "-i", gpu,
            "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=query_timeout()).strip()
        fields = receipt["gpu_query"].split(",")
        if len(fields) != 2: raise ValueError("ambiguous assigned GPU observation")
        memory, utilization = map(int, fields)
        receipt.update(memory_used_mib=memory, utilization_percent=utilization,
            passed=0 <= memory <= 100 and utilization == 0 and not receipt["compute_PID_query"])
    except Exception as error:
        receipt.update(passed=False, observation_error_type=type(error).__name__, observation_error=str(error))
    atomic(ledger, data)
    return receipt


def validate_approval(data, node, *, replacement=False):
    expected = {"schema_version": "latent_T7_paid_fallback_approval_v1", "approved": True,
        "approval_origin": "explicit_human_user", "partition": "gnall", "nodes": 1, "tasks": 1,
        "allocated_CPUs": 8, "GPUs": 1, "memory_GiB": 64, "wall_seconds": 7200,
        "worst_case_CNY": 5.52, "CPU_rate_CNY_per_hour": 0.07, "GPU_rate_CNY_per_hour": 2.20,
        "nonexclusive": True, "working_directory": CODE.as_posix(), "funding_scope": SCOPE,
        "historical_and_free_grants_excluded": True, "original_recompile_allowed": False,
        "candidate_graph_cap": 1, "training_updates": 0, "T30_graph": False}
    if replacement:
        expected.update(schema_version='latent_T7_paid_replacement_approval_v1', wall_seconds=7140,
                        worst_case_CNY=5.474, funding_scope=REPLACEMENT_SCOPE,
                        replaces_job_id='15580715', prior_allocation_reserved_seconds=60,
                        cumulative_wall_seconds_cap=7200, cumulative_cost_CNY_cap=5.52)
    for key, value in expected.items():
        if type(data.get(key)) is not type(value) or data[key] != value:
            raise ValueError("explicit paid approval differs: "+key)
    if not data.get("approval_note") or not data.get("approved_at"):
        raise ValueError("human approval note and recorded time required")
    submitted_node = str(data.get('selected_node', ''))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", submitted_node):
        raise ValueError('original submitted node identity invalid')
    if submitted_node != node and not replacement:
        node_receipt = DIAG/(PREFIX+'-paid.accounting.json')
        adjustment = json.loads(node_receipt.read_text(encoding='utf-8-sig')).get('execution_node_adjustment', {}) if node_receipt.is_file() else {}
        if (adjustment.get('original_node') != submitted_node or adjustment.get('selected_node') != node
                or adjustment.get('job_id') != os.environ.get('SLURM_JOB_ID')
                or adjustment.get('unchanged_approved_resource_and_cost_envelope') is not True
                or not adjustment.get('authority_note') or not adjustment.get('idle_node_snapshot')):
            raise ValueError('actual node differs without a recorded same-job execution adjustment')
    elif submitted_node != node:
        raise ValueError('replacement node differs from actual replacement approval')
    expected_command = f"sbatch --nodelist={submitted_node} {OVERLAY.as_posix()}/slurm_shared_latent_reader_T7_pair_paid.sh"
    if replacement:
        expected_command = replacement_submit_command(submitted_node)
        actual_job = os.environ.get('SLURM_JOB_ID', '')
        if not actual_job.isdigit() or data.get('authorized_SLURM_job_id') != actual_job:
            raise ValueError('replacement must bind its actual newly approved held job ID')
    if data.get("submit_command") != expected_command:
        raise ValueError("approved submission command differs")
    if not data.get("idle_node_snapshot") or not data.get("saved_original_device_kind"):
        raise ValueError("fresh idle-node evidence and exact saved original device kind required")


def replacement_submit_command(node):
    export = 'ALL,T7_PAID_REPLACEMENT_OF=15580715'
    return (f'sbatch --hold --nodelist={node} --time=01:59:00 --export={export} '
            f'--output={DIAG.as_posix()}/{REPLACEMENT_PREFIX}-%j.out '
            f'--error={DIAG.as_posix()}/{REPLACEMENT_PREFIX}-%j.err '
            f'{OVERLAY.as_posix()}/slurm_shared_latent_reader_T7_pair_paid.sh')


def configure_route(args):
    global WALL_SECONDS, CAPS
    if args.replacement_of_job is None:
        return
    if (args.replacement_of_job != '15580715' or args.prefix != REPLACEMENT_PREFIX
            or args.approval != OVERLAY/(REPLACEMENT_PREFIX+'.approval.json')
            or args.ledger != DIAG/(REPLACEMENT_PREFIX+'.gpu-budget.json')
            or args.aggregate_budget != DIAG/(PREFIX+'-paid.aggregate-budget.json')):
        raise ValueError('replacement requires its isolated explicit receipt/ledger/report paths')
    WALL_SECONDS = 7140
    CAPS = {**CAPS, 'runtime-preflight': 120}


def aggregate_replacement(args, approval, *, final=None):
    """One conservative same-envelope replacement; preserve the first allocation."""
    import fcntl
    with args.aggregate_budget.with_suffix('.json.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(args.aggregate_budget.read_text(encoding='utf-8-sig'))
        expected = {'schema_version': 'latent_T7_paid_aggregate_budget_v1',
                    'original_job_id': '15580715', 'wall_seconds_cap': 7200,
                    'cost_CNY_cap': 5.52, 'rate_CNY_per_hour': 2.76,
                    'original_allocation_reserved_seconds': 60,
                    'replacement_wall_seconds_cap': 7140}
        for key, value in expected.items():
            if type(data.get(key)) is not type(value) or data[key] != value:
                raise ValueError('aggregate replacement budget differs: '+key)
        original_path = DIAG/(PREFIX+'-paid.gpu-budget.first-attempt.json')
        if data.get('original_helper_ledger') != str(original_path):
            raise ValueError('aggregate must retain exact immutable first-attempt source')
        original = json.loads(original_path.read_text())
        if (original.get('SLURM_job_id') != '15580715' or original.get('funding_scope') != SCOPE
                or original.get('status') != 'resource_failed' or original.get('allocated_CPUs') != 8
                or original.get('GPUs') != 1 or original.get('occupied_gpu_seconds', 61) > 60
                or not original.get('phases') or any(p.get('status') != 'finished'
                or p.get('child_release_confirmed') is not True for p in original['phases'])
                or data.get('original_scheduler_elapsed_seconds') != 28):
            raise ValueError('original paid allocation lacks settled <=60s evidence')
        if (data.get('original_helper_occupied_gpu_seconds') != original['occupied_gpu_seconds']
                or data.get('original_allocation_cost_CNY_reserve') != 0.046):
            raise ValueError('original charges/reservation changed')
        attempts = data.setdefault('replacement_attempts', [])
        job_id = os.environ['SLURM_JOB_ID']
        if final is None:
            if attempts:
                raise ValueError('one replacement already reserved; no budget reset/retry allowed')
            attempts.append({'job_id': job_id, 'approval_receipt': str(args.approval),
                'ledger': str(args.ledger), 'reserved_allocation_seconds': 7140,
                'reserved_cost_CNY': 5.474, 'status': 'reserved', 'started_epoch': time.time()})
        else:
            if len(attempts) != 1 or attempts[0].get('job_id') != job_id:
                raise ValueError('replacement settlement job does not match reservation')
            attempts[0].update(status=final['status'],
                helper_occupied_gpu_seconds=final['occupied_gpu_seconds'],
                helper_runtime_within_wall=final['runtime_within_wall'], ended_epoch=time.time(),
                scheduler_charge_pending=True)
        data.update(total_reserved_allocation_seconds=60+sum(a['reserved_allocation_seconds'] for a in attempts),
                    total_reserved_cost_CNY=0.046+sum(a['reserved_cost_CNY'] for a in attempts))
        if data['total_reserved_allocation_seconds'] > 7200 or data['total_reserved_cost_CNY'] > 5.52+1e-12:
            raise ValueError('cumulative original+replacement paid envelope exceeded')
        atomic(args.aggregate_budget, data)


def validate_allocation():
    if not os.environ.get("SLURM_JOB_ID"): raise ValueError("actual execution requires a paid Slurm allocation")
    expected = {"SLURM_JOB_PARTITION": "gnall", "SLURM_CPUS_PER_TASK": "8", "SLURM_NTASKS": "1", "SLURM_GPUS_ON_NODE": "1", "SLURM_MEM_PER_NODE": "65536"}
    for key, value in expected.items():
        if os.environ.get(key) != value: raise ValueError("Slurm resource mismatch: "+key)
    if os.environ.get("SLURM_JOB_NUM_NODES", os.environ.get("SLURM_NNODES")) != "1": raise ValueError("one node required")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu or gpu.strip() != gpu: raise ValueError("exactly one scheduler-selected visible GPU required")
    return gpu


def saved_identity(path, architecture):
    if not path.is_file(): raise FileNotFoundError("required existing success binary: "+str(path))
    sidecar = path.with_suffix(path.suffix+".identity.json")
    data = json.loads(sidecar.read_text())
    identity = data["identity"]
    expected_architecture = "original" if architecture == "original" else "shared_latent32x384_reader64x2"
    if (data.get("component") != "T7" or identity.get("architecture") != expected_architecture
            or identity.get("source_git_head") != HEAD or identity.get("horizon") != 7
            or identity.get("native_AD_argnums") != [0, 1] or identity.get("jax_version") != "0.4.38"
            or identity.get("turnover_sha256") != "f6240cf5ce64a47b9bbe0647b5e5715305e9a9230ea77f2bd9f42981ae35a93c"
            or identity.get("compiler_options") != {"xla_gpu_enable_llvm_module_compilation_parallelism": True, "xla_gpu_force_compilation_parallelism": 4}):
        raise ValueError("existing corrected graph identity incompatible")
    return identity


def candidate_path(paid_path):
    """Global guard across free and paid outputs: any success is load-only."""
    paths = sorted(DIAG.glob(PREFIX+"-compile-candidate*.executable.pkl"))
    paths += sorted(DIAG.glob(PREFIX+"-compile-candidate*.executable.pkl.identity.json"))
    paths += sorted(DIAG.glob(PREFIX+"-paid*-compile-candidate*.executable.pkl"))
    paths += sorted(DIAG.glob(PREFIX+"-paid*-compile-candidate*.executable.pkl.identity.json"))
    binaries = {path if path.name.endswith(".pkl") else path.with_name(path.name.removesuffix(".identity.json")) for path in paths}
    binaries.add(paid_path)
    selected = None
    for path in sorted(binaries):
        if path.is_file():
            saved_identity(path, "candidate")
            if selected is not None:
                raise ValueError("multiple preserved candidate graphs; explicit reconciliation required")
            selected = path
        elif path.with_suffix(path.suffix+".identity.json").exists():
            raise ValueError("incomplete preserved candidate assets; no compilation fallback")
    return selected


def container_command(args, gpu, affinity, output):
    toolchain = ROOT/"runtime/toolchains/cuda-nvlink-12.6.85"
    cuda = Path("/usr/lib64/libcuda.so.1").resolve() if not args.dry_run else Path("<resolved libcuda.so.1>")
    nvml = Path("/usr/lib64/libnvidia-ml.so.1").resolve() if not args.dry_run else Path("<resolved libnvidia-ml.so.1>")
    binds = f"/WORK:/WORK,{ROOT}/runtime/data:{ROOT}/data:ro,{toolchain}:{toolchain}:ro,{cuda}:/usr/lib/x86_64-linux-gnu/libcuda.so.1,{nvml}:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    return ["taskset", "-c", affinity, "env", "-u", "LD_LIBRARY_PATH",
        "SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs", "SINGULARITYENV_CUDA_VISIBLE_DEVICES="+gpu,
        "SINGULARITYENV_JAX_PLATFORMS=cuda,cpu", "SINGULARITYENV_JAX_ENABLE_X64=true", "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "SINGULARITYENV_ORCHIDEE_REPO_ROOT="+str(CODE), "SINGULARITYENV_PYTHONPATH="+str(CODE)+":"+str(OVERLAY),
        "SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="+str(ROOT/"runtime/cache/jax/orcjax_gpu"/args.prefix),
        "SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0", "SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0",
        "SINGULARITYENV_PREPEND_PATH="+str(OVERLAY/"nvlink-bin")+":"+str(toolchain/"bin"),
        "SINGULARITYENV_ORCHIDEE_NVLINK_LOG="+str(output)+".nvlink.log", "SINGULARITYENV_ORCHIDEE_NVLINK_STAGE="+args.prefix,
        "SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1", "SINGULARITYENV_OPENBLAS_NUM_THREADS=1", "SINGULARITYENV_OMP_NUM_THREADS=8",
        "singularity", "exec", "--nv", "--bind", binds, "--pwd", str(CODE), "/apps/soft/sif/foundationpose", str(ROOT/".venvs/orcjax_gpu/bin/python"), "-B"]


def probe_args(args, stage, output, preflight):
    return [str(OVERLAY/"probe_shared_latent_reader_T7_pair.py"), "--stage", stage,
        "--manifest", str(OVERLAY/(PREFIX+".manifest.json")), "--prepared-context", str(DIAG/"shared-expert-real-daily-context-c599236.pkl"),
        "--parameters", str(DIAG/"shared-latent-reader-20261004-T7-assembled-input.parameters.pkl"),
        "--field-adapter", str(ROOT/"runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py"),
        "--backbone-adapter", str(ROOT/"runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"),
        "--transformer-module", str(OVERLAY/"shared_transformer.py"), "--latent-module", str(OVERLAY/"shared_latent_reader.py"),
        "--turnover-overlay", str(OVERLAY/"turnover_leaf_age_fall.guarded-source.py"), "--runtime-preflight", str(preflight),
        "--wall-seconds", "6210" if stage == "compile-candidate" else "570", "--lower-seconds", "420", "--backend-seconds", "5700",
        "--call-seconds", "180", "--pairs", "3", "--force-compilation-parallelism", "4", "--output", str(output)]


def reusable_preflight(previous, original_report, original, path, binding):
    linked = original_report.get("runtime_preflight", {})
    old_driver = re.search(r"Driver Version:\s*([0-9.]+)", original_report.get("resources", {}).get("nvidia_smi", ""))
    return bool(previous.get("schema_version") == "daily_training_runtime_check_v1" and previous.get("passed") is True
        and previous.get("status") == "passed" and previous.get("actual_backend") == "gpu"
        and previous.get("jax_version") == binding["jax_version"] == binding["jaxlib_version"]
        and previous.get("numpy_version") == binding["numpy_version"] and linked.get("status") == "passed"
        and original_report.get("status") == "passed" and original_report.get("identities", {}).get("original") == original
        and Path(linked.get("artifact", "")) == path and old_driver and old_driver.group(1) == binding["driver_version"]
        and binding["x64"] and binding["python"] == str(ROOT/".venvs/orcjax_gpu/bin/python")
        and binding["code_root"] == str(CODE) and binding["container"] == "/apps/soft/sif/foundationpose"
        and binding["device_kind"] == original["device_kind"])


def runtime_check(args):
    # Runs only inside the approved paid job; no model graph is compiled here.
    gpu = validate_allocation()
    approval = json.loads(args.approval.read_text())
    validate_approval(approval, os.environ.get("SLURMD_NODENAME", ""), replacement=args.replacement_of_job is not None)
    original = saved_identity(args.original, "original")
    sys.path.insert(0, str(CODE))
    spec = importlib.util.spec_from_file_location("scripts.hpc.orcjax_gpu_runtime", CODE/"scripts/hpc/orcjax_gpu_runtime.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    jax, devices = module.initialize_gpu_backend()
    if len(devices) != 1 or jax.__version__ != original["jax_version"] or devices[0].device_kind != original["device_kind"] or devices[0].device_kind != approval["saved_original_device_kind"]:
        raise ValueError("paid device/JAX does not match the exact saved original; original recompilation forbidden")
    import jaxlib
    import numpy as np
    driver = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=driver_version", "--format=csv,noheader"], text=True, timeout=10).strip()
    binding = {"python": sys.executable, "container": "/apps/soft/sif/foundationpose", "code_root": os.environ.get("ORCHIDEE_REPO_ROOT"),
        "jax_version": jax.__version__, "jaxlib_version": jaxlib.__version__, "numpy_version": np.__version__,
        "device_kind": devices[0].device_kind, "driver_version": driver, "x64": bool(jax.config.jax_enable_x64)}
    reused = None
    if args.runtime_preflight and args.runtime_preflight.is_file():
        previous = json.loads(args.runtime_preflight.read_text())
        report_path = args.original.with_name(args.original.name.removesuffix(".executable.pkl")+".json")
        original_report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        if reusable_preflight(previous, original_report, original, args.runtime_preflight, binding):
            reused = previous
    atomic(args.runtime_report.with_suffix(".compatibility.json"), {"passed": True, "device_kind": devices[0].device_kind,
        "jax_version": jax.__version__, "visible_GPU": gpu, "original_binary": str(args.original), "model_compiles": 0,
        "actual_environment_binding": binding, "existing_preflight_reused": reused is not None,
        "existing_preflight": str(args.runtime_preflight) if args.runtime_preflight else None})
    if reused is not None:
        atomic(args.runtime_report, {**reused, "reused_from": str(args.runtime_preflight),
            "paid_environment_binding_verified": binding, "fresh_synthetic_checks_executed": False})
        return
    import runpy
    sys.argv = [str(CODE/"scripts/hpc/check_daily_training_runtime.py"), "--backend", "gpu", "--output", str(args.runtime_report)]
    runpy.run_path(sys.argv[0], run_name="__main__")


def run(args):
    preflight = DIAG/(args.prefix+"-runtime-preflight.json")
    candidate_output, matched_output = DIAG/(args.prefix+"-compile-candidate.json"), DIAG/(args.prefix+"-matched.json")
    paid_candidate = candidate_output.with_suffix(".executable.pkl")
    if args.dry_run:
        base = container_command(args, "<scheduler single CUDA_VISIBLE_DEVICES>", "<8 CPUs from allocated affinity>", candidate_output)
        print("PREPARED ONLY: requires explicit paid approval receipt + gnall Slurm allocation; no execution")
        print(f'gnall:1 node/1 task/8 CPU/1 V100/64GiB/nonexclusive/{WALL_SECONDS}s; rate2.76CNY/hour')
        print(f'{CAPS["runtime-preflight"]} preflight +6240 candidate +600 matched +60 setup margin +120 cleanup ={WALL_SECONDS} seconds')
        if args.replacement_of_job:
            print('PROPOSAL ONLY: original job15580715 reserved60s/CNY0.046 + replacement7140s/CNY5.474(display5.48) = cumulative7200s/CNY5.52; fresh human approval and actual held newjob binding mandatory')
            print('Replacement isolated approval='+str(args.approval)+' ledger='+str(args.ledger)+' aggregate='+str(args.aggregate_budget))
        print(f'Candidate lower cap420s (observed320.579s), backend5700s, whole candidate6210s; matched570s preserves all8 full-AD calls/3 pairs. Whole{WALL_SECONDS}s and cleanup reserve remain controlling limits; completion is not guaranteed.')
        print("Original required load-only: "+args.original.as_posix()+"; actual device must match its exact saved kind/JAX0.4.38/options")
        print("Candidate global load-only guard: "+DIAG.as_posix()+"/"+PREFIX+"-compile-candidate*.executable.pkl OR "+paid_candidate.as_posix())
        print("EACH PHASE SAMPLER: bash "+(OVERLAY/"shared_latent_T7_paid_memory.sh").as_posix()+" <single physical scheduler GPU> "+DIAG.as_posix()+"/"+args.prefix+"-<phase>.memory.csv -- <phase command below>; process CSV suffix .processes.csv")
        print("ASSIGNED GPU GUARDS: before preflight and candidate model; zero compute PIDs, memory<=100MiB, utilization0; resource failure stops model work")
        base = [value.replace("\\", "/") for value in base]
        reuse_args = ["--runtime-preflight", str(args.runtime_preflight)] if args.runtime_preflight else []
        if args.replacement_of_job:
            reuse_args += ['--replacement-of-job', args.replacement_of_job, '--aggregate-budget', str(args.aggregate_budget),
                           '--ledger', str(args.ledger)]
        print("PRECHECK/PREFLIGHT "+posix_join(base+[str(OVERLAY/Path(__file__).name), "--runtime-check", "--approval", str(args.approval), "--prefix", args.prefix, "--original", str(args.original), "--runtime-report", str(preflight)]+reuse_args))
        print("CANDIDATE "+posix_join(base+probe_args(args, "compile-candidate", candidate_output, preflight))+" [--candidate-executable <existing free/recovery/paid success if present>]")
        print("MATCHED "+posix_join(base+probe_args(args, "matched", matched_output, preflight)+["--original-executable", args.original.as_posix(), "--candidate-executable", "<exact selected existing free/recovery/paid candidate binary>"]))
        return
    started = time.monotonic()
    gpu = validate_allocation()
    approval = json.loads(args.approval.read_text())
    node = os.environ.get("SLURMD_NODENAME", "")
    validate_approval(approval, node, replacement=args.replacement_of_job is not None)
    if args.replacement_of_job:
        scheduler = subprocess.check_output(['/rmprog/slurm/v22.05.7/bin/scontrol', 'show', 'job', os.environ['SLURM_JOB_ID']], text=True)
        if 'TimeLimit=01:59:00' not in scheduler:
            raise ValueError('replacement actual Slurm wall must be7140s')
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE, text=True).strip() != HEAD: raise ValueError("immutable checkout head differs")
    original = saved_identity(args.original, "original")
    if original["device_kind"] != approval["saved_original_device_kind"]: raise ValueError("approved device kind differs from saved original")
    existing = candidate_path(paid_candidate)
    if args.ledger.exists(): raise FileExistsError("paid ledger already exists; a new submission/approval is not inferred")
    if any(path.exists() for path in (preflight, candidate_output, matched_output)): raise FileExistsError("paid outputs already exist")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 8: raise ValueError("allocated affinity contains fewer than8 CPUs")
    chosen = ",".join(map(str, affinity[:8]))
    if args.replacement_of_job:
        aggregate_replacement(args, approval)
    data = {"schema_version": "latent_T7_independent_paid_allocation_v1", "funding_scope": REPLACEMENT_SCOPE if args.replacement_of_job else SCOPE, "approval_receipt": str(args.approval),
        "replaces_job_id": args.replacement_of_job,
        "free_14400_and_historical_grants_excluded": True, "worst_case_CNY": 5.474 if args.replacement_of_job else 5.52, "wall_seconds": WALL_SECONDS,
        "SLURM_job_id": os.environ["SLURM_JOB_ID"], "node": node, "allocated_CPUs": 8, "available_affinity": affinity,
        "used_affinity": chosen, "GPUs": 1, "CUDA_VISIBLE_DEVICES": gpu, "host_load": list(os.getloadavg()),
        "original_saved_binary": str(args.original), "original_compiles": 0, "phase_caps": CAPS, "phases": [], "status": "running"}
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    atomic(args.ledger, data)
    termination = {"signal": None}
    def requested_stop(signum, frame):
        if termination["signal"] is None:
            termination["signal"] = signum
    old_handlers = {signum: signal.signal(signum, requested_stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    def execute(stage, cap, cmd):
        if termination["signal"] is not None:
            raise PaidTermination("received "+signal.Signals(termination["signal"]).name)
        available = WALL_SECONDS-CAPS["cleanup"]-(time.monotonic()-started)
        cap = min(cap, int(available))
        if cap <= 0: raise TimeoutError("cleanup reserve reached")
        before = time.monotonic()
        row = {"stage": stage, "cap_seconds": cap, "command": cmd, "status": "running",
            "GPU_snapshot": subprocess.check_output(["nvidia-smi"], text=True, timeout=10), "host_load": list(os.getloadavg())}
        data["phases"].append(row); atomic(args.ledger, data)
        memory = DIAG/(args.prefix+"-"+stage+".memory.csv")
        sampled_cmd = ["bash", str(OVERLAY/"shared_latent_T7_paid_memory.sh"), gpu, str(memory), "--"]+cmd
        row.update(command=sampled_cmd, memory_receipt=str(memory), process_receipt=str(memory)+".processes.csv")
        process = subprocess.Popen(sampled_cmd, cwd=CODE, start_new_session=True)
        row["pid"] = process.pid
        atomic(args.ledger, data)
        code = None
        try:
            while True:
                if termination["signal"] is not None:
                    raise PaidTermination("received "+signal.Signals(termination["signal"]).name)
                remaining = cap-(time.monotonic()-before)
                if remaining <= 0: raise subprocess.TimeoutExpired(sampled_cmd, cap)
                try:
                    code = process.wait(timeout=min(1, remaining))
                    break
                except subprocess.TimeoutExpired: pass
        except subprocess.TimeoutExpired:
            code = 124
        finally:
            try: shutdown_phase(process)
            finally:
                row.update(status="finished" if process.poll() is not None else "cleanup_failed", child_release_confirmed=process.poll() is not None,
                    exit_status=code if code is not None else process.poll(),
                    termination_signal=termination["signal"], occupied_gpu_seconds=time.monotonic()-before)
                atomic(args.ledger, data)
        return code
    try:
        assigned_gpu_idle(gpu, "runtime-preflight", args.ledger, data)
        prebase = container_command(args, gpu, chosen, preflight)
        reuse_args = ["--runtime-preflight", str(args.runtime_preflight)] if args.runtime_preflight else []
        if args.replacement_of_job:
            reuse_args += ['--replacement-of-job', args.replacement_of_job, '--aggregate-budget', str(args.aggregate_budget),
                           '--ledger', str(args.ledger)]
        code = execute("runtime-preflight", CAPS["runtime-preflight"], prebase+[str(OVERLAY/Path(__file__).name), "--runtime-check", "--approval", str(args.approval), "--prefix", args.prefix, "--original", str(args.original), "--runtime-report", str(preflight)]+reuse_args)
        if code: raise RuntimeError("paid runtime/device preflight failed; no candidate compile")
        # Recheck immediately before model work, not merely at job startup.
        existing = candidate_path(paid_candidate)
        assigned_gpu_idle(gpu, "candidate-model", args.ledger, data)
        cmd = container_command(args, gpu, chosen, candidate_output)+probe_args(args, "compile-candidate", candidate_output, preflight)
        if existing: cmd += ["--candidate-executable", str(existing)]
        code = execute("candidate-load" if existing else "candidate-compile", CAPS["candidate"], cmd)
        chosen_candidate = existing or paid_candidate
        if code or not chosen_candidate.is_file(): raise RuntimeError("candidate phase incomplete/failed; saved graph retained, no fabricated matched ratio")
        saved_identity(chosen_candidate, "candidate")
        cmd = container_command(args, gpu, chosen, matched_output)+probe_args(args, "matched", matched_output, preflight)+["--original-executable", str(args.original), "--candidate-executable", str(chosen_candidate)]
        code = execute("matched-load-only", CAPS["matched"], cmd)
        data.update(status="finished" if code == 0 else "failed", matched_exit_status=code, candidate_saved_binary=str(chosen_candidate))
    except Exception as error:
        data.update(status="terminated" if isinstance(error, PaidTermination) else "resource_failed" if isinstance(error, PaidResourceError) else "failed",
            termination_signal=termination["signal"], error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            snapshot = subprocess.check_output(["nvidia-smi"], text=True, timeout=10)
        except Exception as error:
            snapshot = {"unavailable": type(error).__name__, "message": str(error)}
        elapsed = time.monotonic()-started
        if termination["signal"] is not None:
            data.update(status="terminated", termination_signal=termination["signal"])
        data.update(occupied_gpu_seconds=elapsed, observed_helper_runtime_estimate_CNY=elapsed/3600*(8*.07+2.20),
            cost_estimate_scope=f'helper wall only; scheduler startup excluded; allocation cap{WALL_SECONDS}s; cumulative approved/proposed ceilingCNY5.52',
            runtime_within_wall=elapsed <= WALL_SECONDS, final_GPU_snapshot=snapshot,
            original_compiles=0, candidate_recompile_fallback=False)
        atomic(args.ledger, data)
        if args.replacement_of_job:
            aggregate_replacement(args, approval, final=data)
        for signum, handler in old_handlers.items(): signal.signal(signum, handler)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--approval", type=Path, default=OVERLAY/(PREFIX+"-paid.approval.json"))
    p.add_argument("--ledger", type=Path, default=DIAG/(PREFIX+"-paid.gpu-budget.json"))
    p.add_argument("--prefix", default=PREFIX+"-paid")
    p.add_argument("--original", type=Path, default=DIAG/(PREFIX+"-compile-original.executable.pkl"))
    p.add_argument("--runtime-report", type=Path)
    p.add_argument("--runtime-preflight", type=Path, help="reuse this passed receipt only after actual same-environment binding verification")
    p.add_argument("--runtime-check", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument('--replacement-of-job')
    p.add_argument('--aggregate-budget', type=Path)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    configure_route(arguments)
    if not re.fullmatch(PREFIX+r"-paid(?:-[A-Za-z0-9_-]+)?", arguments.prefix): raise ValueError("isolated paid prefix required")
    if arguments.runtime_check: runtime_check(arguments)
    else: run(arguments)
