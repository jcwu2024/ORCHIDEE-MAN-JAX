"""GPU-only launcher for the frozen rollout-stability experiment."""

from __future__ import annotations

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def main() -> int:
    initialize_gpu_backend()

    from research.daily_coarse_graining.rollout_stability_run import (
        main as run_rollout_stability,
    )

    return run_rollout_stability()


if __name__ == "__main__":
    raise SystemExit(main())
