"""Run a versioned matrix of canonical neural rollouts in one GPU process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend

MATRIX_SCHEMA_VERSION = "canonical_rollout_matrix_v1"
REPORT_SCHEMA_VERSION = "canonical_rollout_matrix_report_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_matrix(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise ValueError(f"matrix schema must be {MATRIX_SCHEMA_VERSION!r}")
    if int(payload.get("start_day", 0)) < 1 or int(payload.get("days", 0)) < 1:
        raise ValueError("matrix day window must be positive")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("rollout matrix must contain cases")
    ids = set()
    for case in cases:
        required = {
            "id",
            "landpoint_id",
            "year",
            "allow_training_split_diagnostic",
        }
        if set(case) != required:
            raise ValueError("rollout matrix case fields do not match the schema")
        case_id = str(case["id"])
        if (
            not case_id
            or case_id in ids
            or "/" in case_id
            or "\\" in case_id
            or not isinstance(case["allow_training_split_diagnostic"], bool)
        ):
            raise ValueError("rollout matrix case ID or diagnostic flag is invalid")
        ids.add(case_id)
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_matrix(args: argparse.Namespace) -> Mapping[str, Any]:
    matrix_path = args.matrix.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError("rollout matrix output directory already exists")
    matrix = load_matrix(matrix_path)
    output_dir.mkdir(parents=True)

    initialize_gpu_backend()
    from research.daily_coarse_graining.canonical_rollout import run_rollout

    records = []
    for case in matrix["cases"]:
        output = output_dir / f"{case['id']}.json"
        rollout_args = argparse.Namespace(
            dataset=args.dataset,
            statistics=args.statistics,
            acceptance=args.acceptance,
            plan=args.plan,
            checkpoint=args.checkpoint,
            landpoint_id=str(case["landpoint_id"]),
            year=int(case["year"]),
            start_day=int(matrix["start_day"]),
            days=int(matrix["days"]),
            allow_training_split_diagnostic=bool(
                case["allow_training_split_diagnostic"]
            ),
            mode="neural",
            teacher_replay_tolerance=1.0e-10,
            state_feedback="free",
            output=output,
        )
        result = run_rollout(rollout_args)
        if result["status"] != "completed":
            raise RuntimeError(f"rollout matrix case {case['id']} did not complete")
        _atomic_json(output, result)
        records.append(
            {
                "id": case["id"],
                "output": output.name,
                "output_sha256": _sha256_file(output),
                "selection": result["selection"],
                "summary": result["summary"],
            }
        )
        print(json.dumps(records[-1], indent=2), flush=True)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "completed",
        "matrix": str(matrix_path),
        "matrix_sha256": _sha256_file(matrix_path),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(args.checkpoint.resolve()),
        "cases": records,
    }
    _atomic_json(output_dir / "matrix_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_matrix(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
