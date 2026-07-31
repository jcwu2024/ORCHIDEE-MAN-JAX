"""GPU-only launcher for all causal-carbon adapter experiment phases."""

from __future__ import annotations

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def main() -> int:
    initialize_gpu_backend()

    from research.daily_coarse_graining.causal_carbon_adapter_run import (
        main as run_adapter,
    )

    return run_adapter()


if __name__ == "__main__":
    raise SystemExit(main())
