"""Compatibility entry point for the production multiyear runner."""

from jax_orchidee.runners.multiyear import (
    PRODUCTION_COMPILED_SECHIBA_DAY_DEFAULT,
    PRODUCTION_NUMPY_ACCUMULATOR_DEFAULT,
    _latest_year_checkpoint,
    main,
)

__all__ = [
    "PRODUCTION_COMPILED_SECHIBA_DAY_DEFAULT",
    "PRODUCTION_NUMPY_ACCUMULATOR_DEFAULT",
    "_latest_year_checkpoint",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
