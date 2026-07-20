from pathlib import Path


def run_oracle(output_dir: Path, compiler: Path) -> dict[str, object]:
    return {"family": output_dir.name, "status": "not_run", "compiler": str(compiler)}
