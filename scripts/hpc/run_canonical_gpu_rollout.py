"""GPU-only launcher for canonical daily validation rollouts."""

from __future__ import annotations

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def main() -> None:
    initialize_gpu_backend()

    from research.daily_coarse_graining.canonical_rollout import main as rollout

    rollout()


if __name__ == "__main__":
    main()
