"""Stdlib Slurm admission and physical-core binding before importing JAX.

This changes execution resources only, never model physics or numerical dtypes.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path


def physical_cores(allowed, sysfs=Path("/sys/devices/system/cpu")):
    records = []
    for logical in sorted(allowed):
        topology = sysfs / f"cpu{logical}" / "topology"
        package = int((topology / "physical_package_id").read_text().strip())
        core = int((topology / "core_id").read_text().strip())
        siblings = (topology / "thread_siblings_list").read_text().strip()
        records.append({"logical_cpu": logical, "package_id": package, "physical_core_id": core,
                        "SMT_siblings": siblings})
    return records


def admit(mode, budget, *, environ=None, allowed=None, sysfs=Path("/sys/devices/system/cpu"), hostname=None,
          partitions=("cnall", "cnmix")):
    env = os.environ if environ is None else environ
    hostname = platform.node() if hostname is None else hostname
    if mode not in {"scalar", "pool", "batch"} or budget not in {1, 4}:
        raise ValueError("CPU modes scalar/pool/batch require fixed budgets 1 or 4")
    if mode == "scalar" and budget != 1 or mode == "pool" and budget != 4:
        raise ValueError("scalar requires one core; persistent pool requires four cores")
    if not env.get("SLURM_JOB_ID") or env.get("SLURM_JOB_PARTITION") not in partitions:
        raise RuntimeError("physical performance requires an actual declared paid Slurm partition allocation")
    if int(env.get("SLURM_CPUS_PER_TASK", "0")) != budget or int(env.get("SLURM_NTASKS", "1")) != 1:
        raise RuntimeError("Slurm allocation must be one task with exactly the declared CPU budget")
    if hostname.startswith(("cln", "test", "ibtest", "gln")):
        raise RuntimeError("login and diagnostic nodes are prohibited for physical CPU performance")
    if allowed is None:
        if not hasattr(os, "sched_getaffinity"):
            raise RuntimeError("Linux sched affinity is required for physical performance")
        allowed = sorted(os.sched_getaffinity(0))
    topology = physical_cores(allowed, sysfs)
    selected, seen = [], set()
    for record in topology:
        identity = record["package_id"], record["physical_core_id"]
        if identity not in seen:
            selected.append(record["logical_cpu"])
            seen.add(identity)
    if len(selected) != budget:
        raise RuntimeError("allocation affinity must contain exactly the declared unique physical cores; require Slurm core binding, SMT is not another core")
    return {"mode": mode, "allocated_CPU_cores": budget, "allocated_CPU_request_count": budget,
            "selected_logical_cpus": selected[:budget],
            "original_affinity": list(allowed), "topology": topology, "hostname": hostname,
            "Slurm": {name: env.get(name) for name in (
                "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK", "SLURM_NTASKS",
                "SLURM_JOB_NODELIST", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "ORCHJAX_SLURM_ALLOC_TRES",
                "ORCHJAX_SLURM_ALLOC_TRES_SOURCE")},
            "pool_controller_policy": "shares worker0 allocated core; no fifth CPU" if mode == "pool" else None,
            "thread_count_claim": "affinity is a resource bound; vmap is not a thread-count claim"}


def admit_gpu(*, environ=None, allowed=None, sysfs=Path("/sys/devices/system/cpu"), hostname=None):
    """GPU auxiliary allocation: retain inherited mask and disclose SMT topology."""
    env = os.environ if environ is None else environ
    hostname = platform.node() if hostname is None else hostname
    request = int(env.get("SLURM_CPUS_PER_TASK", "0"))
    if (not env.get("SLURM_JOB_ID") or env.get("SLURM_JOB_PARTITION") != "gnall"
            or request <= 0 or int(env.get("SLURM_NTASKS", "1")) != 1):
        raise RuntimeError("candidate-first requires one real gnall Slurm task and declared auxiliary CPU allocation")
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or visible == "-1" or len(visible.split(",")) != 1:
        raise RuntimeError("candidate-first must retain one Slurm-selected visible GPU")
    if not env.get("SLURM_JOB_GPUS") and not env.get("SLURM_STEP_GPUS"):
        raise RuntimeError("Slurm allocated GPU identity required")
    if hostname.startswith(("cln", "test", "ibtest", "gln")):
        raise RuntimeError("candidate-first performance cannot run on login/diagnostic nodes")
    allowed = sorted(os.sched_getaffinity(0)) if allowed is None else sorted(allowed)
    topology = physical_cores(allowed, sysfs)
    if not allowed:
        raise RuntimeError("empty GPU auxiliary affinity mask")
    physical = {(record["package_id"], record["physical_core_id"]) for record in topology}
    return {"mode": "GPU_auxiliary", "allocated_CPU_request_count": request,
            "observed_unique_physical_cores": len(physical), "original_affinity": allowed,
            "selected_logical_cpus": allowed, "topology": topology, "hostname": hostname,
            "CUDA_VISIBLE_DEVICES_preserved": visible,
            "Slurm": {name: env.get(name) for name in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK",
                "SLURM_NTASKS", "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "ORCHJAX_SLURM_ALLOC_TRES",
                "ORCHJAX_SLURM_ALLOC_TRES_SOURCE")},
            "thread_count_claim": "inherited auxiliary mask preserved; allocated CPU count is not a physical-core equivalence claim"}


def bind(cpus, *, one_thread):
    """Must run in each spawned process before its first model/JAX import."""
    if "jax" in __import__("sys").modules:
        raise RuntimeError("resource binding must precede JAX import")
    flags = os.environ.get("XLA_FLAGS", "")
    if not one_thread and len(cpus) > 1 and (
            "xla_cpu_multi_thread_eigen=false" in flags or "intra_op_parallelism_threads=1" in flags):
        raise RuntimeError("batch4 cannot inherit pool/global single-thread XLA flags; declare a four-core execution policy")
    os.sched_setaffinity(0, set(cpus))
    if sorted(os.sched_getaffinity(0)) != sorted(cpus):
        raise RuntimeError("failed to establish exact CPU affinity")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1" if one_thread else str(len(cpus))


def parse_cpu_list(value):
    result = set()
    for part in value.split(","):
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            result.update(range(first, last + 1))
        else:
            result.add(int(part))
    return result


def thread_affinity(cpus, task_root=Path("/proc/self/task")):
    """Audit actual masks of all presently observed backend/runtime threads."""
    expected, records = set(cpus), []
    for status in task_root.glob("*/status"):
        try:
            lines = status.read_text().splitlines()
        except FileNotFoundError:
            continue  # Thread ended during this census.
        value = next(line.split(":", 1)[1].strip() for line in lines if line.startswith("Cpus_allowed_list:"))
        allowed = parse_cpu_list(value)
        if not allowed or not allowed.issubset(expected):
            raise RuntimeError(f"thread affinity widened outside declared resource mask: TID{status.parent.name} {value}")
        records.append({"TID": int(status.parent.name), "allowed_logical_cpus": sorted(allowed)})
    if not records:
        raise RuntimeError("Linux backend thread affinity evidence unavailable")
    return {"observed_thread_count": len(records), "expected_logical_cpus": sorted(expected),
            "all_observed_thread_masks_within_declared_cores": True, "threads": records,
            "scope": "post-backend/compile census; cgroup and process affinity remain enclosing allocation bounds"}
