"""GPU-only launcher for canonical daily coarse-graining training."""

from __future__ import annotations

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def main() -> int:
    initialize_gpu_backend()

    # Delay this import until CUDA plugin registration is complete. The
    # training module imports jax.numpy and creates compiled functions.
    from research.daily_coarse_graining.canonical_training_run import main as train

    return train()


if __name__ == "__main__":
    raise SystemExit(main())
