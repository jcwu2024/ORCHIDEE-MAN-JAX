"""Actual target Bash expansion in a PROJECT-only relocated diagnostic fixture."""
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path


def main():
    started = time.perf_counter()
    prep = Path(__file__).resolve().parent
    project = prep / "interface-project"
    declared = project / "runtime/worktrees/declared-code/scripts/hpc"
    declared.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in ("slurm_global_gpu_capacity.sh", "slurm_global_gpu_capacity_1024.sh"):
        source = (prep / "launcher-sources" / name).read_bytes()
        sources[name] = hashlib.sha256(source).hexdigest()
        source = source.decode().replace("\r\n", "\n").replace(
            "PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX", f"PROJECT={shlex.quote(str(project))}")
        (declared / name).write_text(source)
        subprocess.run(["bash", "-n", str(declared / name)], check=True)
    spool = prep / "diagnostic-spool"
    spool.mkdir(exist_ok=True)
    spooled = spool / "slurm_script"
    shutil.copyfile(declared / "slurm_global_gpu_capacity_1024.sh", spooled)
    env = {key: value for key, value in os.environ.items() if key not in (
        "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES",
        "CONTINUE_512", "CONTINUE_256_512", "FIXED_1024", "OPTIONAL_B64_BASELINE", "RUN_ROOT")}
    env["CODE_DIR"] = str(declared.parents[1])
    evidence = []
    for name, wrapper in (("declared", declared / "slurm_global_gpu_capacity_1024.sh"), ("spooled", spooled)):
        run = subprocess.run(["bash", str(wrapper), "--dry-run"], env=env, text=True, capture_output=True)
        if run.returncode:
            raise AssertionError(run.stderr)
        tokens = shlex.split(next(line for line in run.stdout.splitlines() if line.startswith("command=")).split("=", 1)[1])
        if "--fixed-1024" not in tokens or "--b512-evidence-dir" not in tokens:
            raise AssertionError("fixed1024 arguments absent")
        if any(flag in tokens for flag in ("--continue-512", "--continue-256-512", "--optional-b64-baseline", "--conditional-512")):
            raise AssertionError("historical mode leaked into fixed1024")
        if "wall=01:30:00" not in run.stdout:
            raise AssertionError("wrong hard-wall expansion")
        evidence.append({"case": name, "exit": run.returncode, "stdout": run.stdout,
            "stderr": run.stderr, "command": tokens})
    if evidence[0]["command"] != evidence[1]["command"]:
        raise AssertionError("spooled wrapper changes actual command")
    if (project / "runtime/outputs").exists():
        raise AssertionError("dry-run created runtime output")
    result = {"status": "passed", "source_sha256": sources, "checks": evidence,
        "direct_spooled_commands_equal": True, "wall_seconds": time.perf_counter() - started,
        "scope": "target Bash syntax/runtime expansion; only PROJECT literal relocated into preparation fixture; no paid checkout deployment",
        "model_preparation_lower_compile_run": False, "JAX_imported": False, "GPU_initialization": False,
        "eventual_immutable_declared_checkout_interface_check": "pending final code SHA/deployment"}
    (prep / "canonical-launcher-interfaces.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "direct_spooled_equal": True, "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
