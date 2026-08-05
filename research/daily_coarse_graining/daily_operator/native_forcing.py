"""Native-record forcing assembly and a minimal order-sensitive encoder."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .types import ForcingEncoderParameters, NativeForcingInput

SECONDS_PER_DAY = 86400.0


def validate_native_forcing(sequence: NativeForcingInput) -> None:
    """Fail closed on precision, shape, interval, and mask contract drift."""

    values = np.asarray(sequence.values)
    times = np.asarray(sequence.record_time_seconds)
    durations = np.asarray(sequence.duration_seconds)
    predecessor = np.asarray(sequence.predecessor_mask)
    mask = np.asarray(sequence.record_mask)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("native forcing values must have shape [n_record, n_feature]")
    n_record = values.shape[0]
    if times.shape != (n_record,) or durations.shape != (n_record,):
        raise ValueError("forcing time and duration must share the record axis")
    if predecessor.shape != (n_record,) or mask.shape != (n_record,):
        raise ValueError("forcing predecessor and record masks must share the record axis")
    for name, array in {"values": values, "times": times, "durations": durations}.items():
        if array.dtype != np.dtype(np.float64):
            raise TypeError(f"native forcing {name} must use float64")
    if predecessor.dtype != np.dtype(np.bool_) or mask.dtype != np.dtype(np.bool_):
        raise TypeError("native forcing masks must use exact bool dtype")
    if np.any(mask & ~np.isfinite(times)) or np.any(mask & ~np.isfinite(durations)):
        raise ValueError("active forcing interval metadata must be finite")
    if np.any(mask[:, None] & ~np.isfinite(values)):
        raise ValueError("active native forcing values must be finite")
    if np.any(mask & (durations <= 0.0)):
        raise ValueError("active native forcing durations must be positive")
    active_times = times[mask]
    if active_times.size > 1 and np.any(np.diff(active_times) < 0.0):
        raise ValueError("active native forcing records must be time ordered")
    if np.any(predecessor & ~mask):
        raise ValueError("a predecessor record must also be active")


def assemble_native_forcing_features(
    sequence: NativeForcingInput,
    value_scale: Any,
) -> Any:
    """Assemble unit-scaled values and interval metadata without interpolation."""

    values = jnp.asarray(sequence.values, dtype=jnp.float64)
    scale = jnp.asarray(value_scale, dtype=jnp.float64)
    if scale.shape != (values.shape[1],):
        raise ValueError("forcing value_scale must contain one named-field scale")
    safe_scale = jnp.where(scale > 0.0, scale, jnp.ones_like(scale))
    time = jnp.asarray(sequence.record_time_seconds, dtype=jnp.float64) / SECONDS_PER_DAY
    duration = jnp.asarray(sequence.duration_seconds, dtype=jnp.float64) / SECONDS_PER_DAY
    predecessor = jnp.asarray(sequence.predecessor_mask, dtype=jnp.float64)
    mask = jnp.asarray(sequence.record_mask, dtype=jnp.bool_)
    features = jnp.concatenate(
        (values / safe_scale, time[:, None], duration[:, None], predecessor[:, None]),
        axis=1,
    )
    return jnp.where(mask[:, None], features, jnp.zeros_like(features))


def encode_native_forcing(
    sequence: NativeForcingInput,
    parameters: ForcingEncoderParameters,
) -> Any:
    """Encode the short ordered sequence with one masked recurrent scan.

    This is an E1 plumbing encoder, not a selected E2 architecture. It consumes
    exactly the supplied native records and has no 48-step reconstruction.
    """

    features = assemble_native_forcing_features(sequence, parameters.value_scale)
    input_kernel = jnp.asarray(parameters.input_kernel, dtype=jnp.float64)
    recurrent_kernel = jnp.asarray(parameters.recurrent_kernel, dtype=jnp.float64)
    bias = jnp.asarray(parameters.bias, dtype=jnp.float64)
    if input_kernel.shape[0] != features.shape[1]:
        raise ValueError("forcing input kernel does not match assembled features")
    hidden_size = input_kernel.shape[1]
    if recurrent_kernel.shape != (hidden_size, hidden_size) or bias.shape != (hidden_size,):
        raise ValueError("forcing recurrent parameter shapes are inconsistent")
    mask = jnp.asarray(sequence.record_mask, dtype=jnp.bool_)

    def step(carry, record):
        feature, active = record
        candidate = jnp.tanh(feature @ input_kernel + carry @ recurrent_kernel + bias)
        return jnp.where(active, candidate, carry), None

    initial = jnp.zeros((hidden_size,), dtype=jnp.float64)
    context, _ = jax.lax.scan(step, initial, (features, mask))
    return context
