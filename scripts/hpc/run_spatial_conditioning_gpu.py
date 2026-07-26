"""Run spatial-conditioning diagnostics on an accepted JAX GPU runtime."""

from __future__ import annotations

from typing import Sequence

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def main(argv: Sequence[str] | None = None) -> int:
    initialize_gpu_backend()
    from research.daily_coarse_graining.spatial_conditioning_diagnostic import (
        main as diagnostic_main,
    )

    return diagnostic_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
