"""Compatibility entry point for paper-landpoint acceptance."""

from jax_orchidee.runners.acceptance import (
    KEY_MODEL_OUTPUTS,
    _evaluate,
    _multiyear_command,
    _payload_matches_request,
    main,
)

__all__ = [
    "KEY_MODEL_OUTPUTS",
    "_evaluate",
    "_multiyear_command",
    "_payload_matches_request",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
