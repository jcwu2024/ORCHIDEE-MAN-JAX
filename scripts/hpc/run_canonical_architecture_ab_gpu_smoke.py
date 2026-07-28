"""GPU-only launcher for the real-shard architecture A/B smoke."""

from __future__ import annotations

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def main() -> int:
    initialize_gpu_backend()

    from research.daily_coarse_graining.canonical_architecture_ab_smoke import (
        main as run_smoke,
    )

    return run_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
