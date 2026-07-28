"""GPU-only launcher for the matched 669-point architecture A/B."""

from __future__ import annotations

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def main() -> int:
    initialize_gpu_backend()

    from research.daily_coarse_graining.canonical_architecture_ab_run import (
        main as run_ab,
    )

    return run_ab()


if __name__ == "__main__":
    raise SystemExit(main())
