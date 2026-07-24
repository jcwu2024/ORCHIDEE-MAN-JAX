"""GPU-only launcher for differentiable multistep canonical training."""

from __future__ import annotations

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def main() -> int:
    initialize_gpu_backend()

    from research.daily_coarse_graining.canonical_multistep_training_run import (
        main as train,
    )

    return train()


if __name__ == "__main__":
    raise SystemExit(main())
