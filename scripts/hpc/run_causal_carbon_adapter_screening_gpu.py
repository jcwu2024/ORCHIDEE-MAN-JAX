"""GPU-only launcher for Experiment C post-training screening."""

from __future__ import annotations

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def main() -> int:
    initialize_gpu_backend()

    from research.daily_coarse_graining.causal_carbon_adapter_screening import (
        main as run_screening,
    )

    return run_screening()


if __name__ == "__main__":
    raise SystemExit(main())
