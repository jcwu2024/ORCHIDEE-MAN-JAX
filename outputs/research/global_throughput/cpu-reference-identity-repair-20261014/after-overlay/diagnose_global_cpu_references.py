"""Free one-CPU saved-reference preparation; exits before physical runner creation.

Uses the production input reader, stage identity producer and reference loader.
This entry cannot execute a physical model and never admits a CPU48 allocation.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from scripts.dev import benchmark_global_server_cpu as cpu
    from scripts.dev.global_cpu_candidate_identity import candidate_identity

    args = cpu.parser().parse_args()
    affinity = sorted(os.sched_getaffinity(0))
    if (os.environ.get("SLURM_JOB_PARTITION") != "cntest"
            or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
            or os.environ.get("SLURM_NTASKS") != "1" or len(affinity) != 1
            or not socket.gethostname().startswith("ibtest")):
        raise RuntimeError("reference diagnostic requires actual cntest one-CPU test-node allocation")
    if (args.mode != "batch" or args.cpu_budget != 1 or args.sizes != [64]
            or args.validation_mode != "exploratory" or args.cpu_reference_role != "cpu-internal"
            or args.prepared_input is None or args.gpu_reference_dir is None
            or args.cpu_reference_dir is None or args.cpu_reference_admission_record is None
            or not args.expected_cpu_reference_admission_sha256 or not args.expected_gpu_reference_sha256
            or args.dry_run or args.synthetic_smoke or args.cache_mode != "off" or args.warm_repeats != 5):
        raise ValueError("diagnostic requires unchanged FULL64 exploratory science/reference arguments and budget1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"diagnostic single-CPU thread policy differs at {key}")
    os.environ.setdefault("ORCHIDEE_REPO_ROOT", str(ROOT))
    os.environ.setdefault("ORCHIDEE_DATA_ROOT", str(args.truth_root / "data"))
    os.environ.setdefault("ORCHIDEE_REFERENCE_ROOT", str(args.truth_root / "reference"))
    os.environ["ORCHIDEE_OUTPUT_ROOT"] = str(args.output_dir.resolve())
    from scripts.dev import benchmark_global_scale as scale

    scale.jax.config.update("jax_enable_x64", True)
    recorder = scale.base.Recorder(args.output_dir.resolve())
    code = 0
    try:
        environment = scale.base.metadata(args)
        recorder.result["environment"] = environment
        if (environment["jax"] != "0.4.38" or environment["jaxlib"] != "0.4.38"
                or environment["gpu_count"] != 0 or len(environment["device_details"]) != 1):
            raise RuntimeError("accepted single CPU JAX/jaxlib0.4.38 required")
        resources = {"candidate_provenance": candidate_identity(args, ROOT),
                     "parent_allocated_CPU_request_count": 1, "billing_CPU_request_count": 1}
        lanes, second = scale.read_prepared(args.prepared_input, args.config)
        prepared = cpu.prepare_references(args, recorder, lanes, second, scale, resources)
        identity, screen, gpu, origin, scalar, _, _ = prepared
        recorder.result.update(status="reference_preparation_passed", workload_points=len(lanes),
            compatibility_audit=screen, output_leaf_counts={"GPU": len(scale.jax.tree_util.tree_leaves(gpu)),
                "CPU_scalar": len(scale.jax.tree_util.tree_leaves(scalar[64][0]))},
            gpu_checkpoint_sha256=origin["sha256"], scalar_checkpoint_sha256=scalar[64][1]["sha256"],
            candidate_provenance=resources["candidate_provenance"],
            actual_stage_runtime_git_head=identity["runtime"].get("git_head"))
    except Exception as exc:
        code = 1
        recorder.result.update(status="reference_preparation_failed", message=str(exc), traceback=traceback.format_exc())
    finally:
        recorder.result.update(diagnostic_allocation={"job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"), "node": socket.gethostname(),
            "actual_cpus": 1, "affinity": affinity}, JAX_initialized=True,
            physical_runner_created=False, physical_lower_compile_run=False, scientific_acceptance=False,
            exit_boundary="immediately after shared production prepare_references; before StageStore/SingleRunner/PoolRunner")
        recorder.stop.set()
        recorder.flush()
        print(json.dumps({"status":recorder.result["status"], "message":recorder.result.get("message"),
            "identity":recorder.result.get("reference_preparation_identity"), "physical_lower_compile_run":False}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
