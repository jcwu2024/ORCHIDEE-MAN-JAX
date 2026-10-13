import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


root = Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX")
code = root / "runtime/worktrees/global-throughput-cpu48-b64-v3"
prepared = root / "runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl"
reference_dir = root / "runtime/outputs/research/global_throughput/gpu-round4/model"
config = code / "configs/orchidee_man_250919.yaml"
expected_commit = "03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8"
expected_prepared = "be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7"
expected_reference = "a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=code, text=True).strip()


assert git("rev-parse", "HEAD") == expected_commit
assert git("status", "--porcelain", "--untracked-files=all") == ""
assert not (code / ".git/objects/info/alternates").exists()
assert (code / "jax_orchidee/__init__.py").is_file()
assert config.is_file()
assert prepared.is_file()
prepared_sha = sha256(prepared)
assert prepared_sha == expected_prepared, prepared_sha

manifest_path = reference_dir / "stage_manifest.json"
manifest = json.loads(manifest_path.read_text())
record = manifest["stages"]["B4_group_references"]
reference_path = (reference_dir / record["path"]).resolve()
assert reference_path.is_relative_to(reference_dir.resolve())
reference_sha = sha256(reference_path)
assert record["sha256"] == reference_sha == expected_reference, reference_sha

sys.path.insert(0, str(code))

import jax
import jaxlib
import jax_orchidee

result = {
    "status": "passed",
    "hostname": subprocess.check_output(["hostname"], text=True).strip(),
    "cpu_affinity": sorted(os.sched_getaffinity(0)),
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "jax": jax.__version__,
    "jaxlib": jaxlib.__version__,
    "jax_enable_x64": jax.config.read("jax_enable_x64"),
    "jax_orchidee_module": str(Path(jax_orchidee.__file__).resolve()),
    "code_dir": str(code),
    "commit": git("rev-parse", "HEAD"),
    "worktree_clean": not bool(git("status", "--porcelain", "--untracked-files=all")),
    "alternates": (code / ".git/objects/info/alternates").exists(),
    "science_tree": git("rev-parse", "HEAD:jax_orchidee"),
    "prepared_input": str(prepared),
    "prepared_input_sha256": prepared_sha,
    "gpu_reference_stage": str(reference_path),
    "gpu_reference_sha256": reference_sha,
    "config": str(config),
    "model_execution": False,
    "lower_or_compile": False,
}
assert result["python"] == "3.11.15"
assert result["jax"] == result["jaxlib"] == "0.4.38"
assert result["jax_enable_x64"] is True
assert result["science_tree"] == "75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba"
assert len(result["cpu_affinity"]) == 1
print(json.dumps(result, indent=2, sort_keys=True))
