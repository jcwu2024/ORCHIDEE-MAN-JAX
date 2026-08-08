"""GPU-only launcher for one frozen first-process arm and seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-git-head", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    jax, _devices = initialize_gpu_backend()
    jax.config.update("jax_enable_x64", True)
    from research.daily_coarse_graining.first_neural_process_training_run import (
        prepare_process_training,
        train_first_process_arm,
    )

    prepared = prepare_process_training(
        dataset_manifest=args.dataset_manifest,
        acceptance_report=args.acceptance_report,
        base_statistics=args.base_statistics,
        process_statistics=args.process_statistics,
        protocol_path=args.protocol,
    )
    report = train_first_process_arm(
        prepared,
        arm=args.arm,
        seed=args.seed,
        source_git_head=args.source_git_head,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
