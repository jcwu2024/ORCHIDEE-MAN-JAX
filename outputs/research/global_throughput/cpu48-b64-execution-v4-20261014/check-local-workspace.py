"""Record the fixed local checkout and actual package import before execution."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().resolve()))
import jax_orchidee

root = Path.cwd().resolve()
expected = Path("E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput").resolve()
top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
module = Path(jax_orchidee.__file__).resolve()
assert root == expected and Path(top).resolve() == expected
assert branch == "research/global-throughput"
assert module == expected / "jax_orchidee/__init__.py"
result = {"status": "passed", "cwd": str(root), "git_top_level": top,
          "branch": branch, "actual_import": str(module),
          "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "physical_model_execution": False}
Path(__file__).with_name("local-workspace-check.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
