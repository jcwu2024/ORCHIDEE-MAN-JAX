"""GPU-only launcher for the formal rollout-stability screening evaluator."""

from __future__ import annotations

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def main() -> int:
    initialize_gpu_backend()

    from research.daily_coarse_graining.rollout_stability_screening import (
        main as run_screening,
    )

    return run_screening()


if __name__ == "__main__":
    raise SystemExit(main())
