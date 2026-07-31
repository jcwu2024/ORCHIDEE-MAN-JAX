from __future__ import annotations

import ast
from pathlib import Path


def test_causal_adapter_launcher_initializes_gpu_before_runner_import():
    path = Path(__file__).resolve().parents[2] / "scripts" / "hpc" / "run_causal_carbon_adapter_gpu.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    assert isinstance(main.body[0], ast.Expr)
    assert ast.unparse(main.body[0].value) == "initialize_gpu_backend()"
    assert isinstance(main.body[1], ast.ImportFrom)
    assert main.body[1].module == ("research.daily_coarse_graining.causal_carbon_adapter_run")


def test_causal_adapter_screening_initializes_gpu_before_runner_import():
    path = Path(__file__).resolve().parents[2] / "scripts" / "hpc" / "run_causal_carbon_adapter_screening_gpu.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    assert isinstance(main.body[0], ast.Expr)
    assert ast.unparse(main.body[0].value) == "initialize_gpu_backend()"
    assert isinstance(main.body[1], ast.ImportFrom)
    assert main.body[1].module == ("research.daily_coarse_graining.causal_carbon_adapter_screening")
