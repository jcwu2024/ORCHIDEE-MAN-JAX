"""Declared immutable fixed2048 interfaces only; no model or GPU entry execution."""
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT = Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX")
PREP = PROJECT / "runtime/outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009"
CODE = PROJECT / "runtime/worktrees/global-throughput-gpu-fixed2048-v1"
EXPECTED = json.loads((PREP / "final-code-pin.json").read_text())["execution_code_commit"]


def main():
    started = time.perf_counter()
    checks = []

    def command(name, args):
        result = subprocess.run(args, cwd=CODE, text=True, capture_output=True)
        checks.append({"name": name, "command": [str(arg) for arg in args],
            "exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise AssertionError(f"{name}: {result.stderr}")
        return result.stdout

    if command("commit", ["git", "rev-parse", "HEAD"]).strip() != EXPECTED:
        raise AssertionError("exact execution pin required")
    if command("clean_before", ["git", "status", "--porcelain", "--untracked-files=all"]).strip():
        raise AssertionError("immutable checkout dirty before interfaces")
    pin = json.loads((PREP / "final-code-pin.json").read_text())
    source_files = {}
    for name, expected in pin["commit_blob_sha256"].items():
        actual = hashlib.sha256((CODE / name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"committed executed interface source bytes differ: {name}")
        source_files[name] = actual
    sys.path.insert(0, str(CODE))
    from scripts.dev.global_source_compatibility import source_identity

    science = source_identity(CODE)
    if science["actual_science_source_sha256"] != "b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8" or not science["execution_source_compatibility"]["active"]:
        raise AssertionError("inherited exact scientific compatibility mismatch")
    paths = {
        "prepared": PROJECT / "runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl",
        "reference": PROJECT / "runtime/outputs/research/global_throughput/gpu-round4/model",
        "b1024": PROJECT / "runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1/capacity",
        "memory_baseline": PROJECT / "runtime/outputs/research/global_throughput/server-benchmark/global-throughput-v2-2b14506/gpu-b64/candidate-first",
        "admission": PROJECT / "runtime/outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/target-input-admission.json",
        "assets": PROJECT / "runtime/outputs/research/global_throughput/round4-inputs/selected64-assets-v2",
        "config": CODE / "configs/orchidee_man_250919.yaml",
        "cpu_python": PROJECT / ".venvs/orcjax_cpu/bin/python",
        "gpu_python": PROJECT / ".venvs/orcjax_gpu/bin/python",
        "image": Path("/apps/soft/sif/foundationpose"),
        "identity_helper": CODE / "scripts/hpc/global_gpu_identity.py"}
    if any(not path.exists() for path in paths.values()):
        raise AssertionError("required interface/runtime path absent")
    capacity = CODE / "scripts/dev/benchmark_global_gpu_capacity.py"
    command("capacity_help", [sys.executable, "-B", capacity, "--help"])
    args = ["--prepared-input", paths["prepared"], "--full-reference-dir", paths["reference"],
        "--memory-baseline-dir", paths["memory_baseline"],
        "--output-dir", PROJECT / "runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed2048-v1/capacity",
        "--config", paths["config"], "--asset-root", paths["assets"], "--warm-repeats", "5",
        "--budget-seconds", "7200", "--fixed-2048", "--b1024-evidence-dir", paths["b1024"], "--stage-shape", "1024", "--dry-run"]
    dry = json.loads(command("capacity_actual_dry_run", [sys.executable, "-B", capacity, *args]))
    if dry["contract"] != "GPU-fixed2048-repeated-FULL64-B1024-B2048-fresh-process-capacity-v1" or dry["candidate_shapes"] != [1024, 2048] or dry["lane_days"] != 28672 or dry["JAX_imported"] or dry["model_execution"]:
        raise AssertionError("declared fixed2048 no-JAX interface drift")
    candidate_args = list(args)
    candidate_args[candidate_args.index("--stage-shape") + 1] = "2048"
    candidate_args += ["--control-stage-dir", PROJECT / "runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed2048-v1/capacity/B1024"]
    candidate = json.loads(command("capacity_actual_candidate_dry_run", [sys.executable, "-B", capacity, *candidate_args]))
    if candidate["contract"] != dry["contract"] or candidate["lane_days"] != 28672 or candidate["JAX_imported"] or candidate["model_execution"]:
        raise AssertionError("declared fresh candidate no-JAX interface drift")
    for name in ("slurm_global_gpu_capacity.sh", "slurm_global_gpu_capacity_2048.sh"):
        command("syntax_" + name, ["bash", "-n", CODE / "scripts/hpc" / name])
    os.environ["CODE_DIR"] = str(CODE)
    os.environ["RUN_ROOT"] = str(PROJECT / "runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed2048-v1")
    wrapper = CODE / "scripts/hpc/slurm_global_gpu_capacity_2048.sh"
    spool = PREP / "actual-deployed-spool"
    spool.mkdir(exist_ok=True)
    spooled = spool / "slurm_script"
    shutil.copyfile(wrapper, spooled)
    commands = []
    for name, path in (("direct", wrapper), ("spooled", spooled)):
        output = command(name, ["bash", path, "--dry-run"])
        tokens = shlex.split(next(line for line in output.splitlines() if line.startswith("command=")).split("=", 1)[1])
        if str(CODE / "scripts/hpc/global_batch_gpu_entry.py") not in tokens or tokens[tokens.index("--config") + 1] != str(paths["config"]) or "--fixed-2048" not in tokens:
            raise AssertionError("actual CODE_DIR helper/config resolution drift")
        commands.append(tokens)
    if commands[0] != commands[1]:
        raise AssertionError("actual direct/spool commands differ")
    coordinator = CODE / "scripts/hpc/global_gpu_capacity_2048_coordinator.py"
    if commands[0][1] != str(coordinator) or commands[0][0] != str(paths["cpu_python"]):
        raise AssertionError("coordinator must resolve through canonical CPU Python")
    plan = json.loads(command("coordinator_actual_plan", [sys.executable, "-B", coordinator,
        "--dry-run", *commands[0][2:]]))
    if plan["contract"] != "GPU-fixed2048-repeated-FULL64-B1024-B2048-fresh-process-capacity-v1":
        raise AssertionError("fresh2048 coordinator contract drift")
    for size in (1024, 2048):
        tokens = plan["commands"][str(size)]
        if tokens[tokens.index("--stage-shape") + 1] != str(size):
            raise AssertionError("coordinator child shape drift")
        if tokens[tokens.index("--output-dir") + 1] != str(Path(os.environ["RUN_ROOT"]) / "capacity" / f"B{size}"):
            raise AssertionError("coordinator child output lifetime/layout drift")
        if str(CODE / "scripts/hpc/global_batch_gpu_entry.py") not in tokens:
            raise AssertionError("coordinator child GPU entry drift")
    if plan["commands"]["2048"][plan["commands"]["2048"].index("--control-stage-dir") + 1] != str(Path(os.environ["RUN_ROOT"]) / "capacity/B1024"):
        raise AssertionError("candidate control stage drift")
    collector = CODE / "outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/collect_fixed2048_result.py"
    command("collector_help", [sys.executable, "-B", collector, "--help"])
    spec = importlib.util.spec_from_file_location("fixed2048_deployed_collector", collector)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.admit_code_checkout(CODE, EXPECTED)
    if "jax" in sys.modules:
        raise AssertionError("interface probe imported JAX")
    if command("clean_after", ["git", "status", "--porcelain", "--untracked-files=all"]).strip():
        raise AssertionError("immutable checkout dirty after interfaces")
    if Path(os.environ["RUN_ROOT"]).exists():
        raise AssertionError("dry interfaces created paid output root")
    result = {"status": "passed", "code_checkout": str(CODE), "expected_commit": EXPECTED,
        "clean_before_after": True, "independent_objects_no_alternates": not (CODE / ".git/objects/info/alternates").exists(),
        "committed_interface_source_sha256": source_files, "scientific_source_identity": science,
        "required_paths_exist": {key: str(value) for key, value in paths.items()},
        "checks": checks, "collector_explicit_clean_code_pin_admission": True,
        "direct_spooled_commands_equal": True, "actual_coordinator_plan": plan, "fixture_PROJECT_relocation": False,
        "affinity": sorted(os.sched_getaffinity(0)), "wall_seconds": time.perf_counter() - started,
        "JAX_imported": False, "model_preparation_lower_compile_run": False, "GPU_initialization": False,
        "fixed2048_model_acceptance": False, "paid_submission": False}
    (PREP / "deployment-interface-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "checks": len(checks), "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
