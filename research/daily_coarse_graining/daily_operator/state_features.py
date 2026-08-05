"""Axis-preserving state and PFT feature assembly for Gate E1."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .types import PFTAxisInput, StateDefinedMasks, StateFeatureGroups


def validate_state_feature_groups(state: StateFeatureGroups, masks: StateDefinedMasks) -> None:
    """Validate grouped float64 arrays and exact defined masks."""

    for name in StateFeatureGroups._fields:
        values = np.asarray(getattr(state, name))
        defined = np.asarray(getattr(masks, name))
        if values.dtype != np.dtype(np.float64):
            raise TypeError(f"state group {name} must use float64")
        if defined.dtype != np.dtype(np.bool_) or defined.shape != values.shape:
            raise TypeError(f"state group {name} requires a shape-matched bool mask")
        if np.any(defined & ~np.isfinite(values)):
            raise ValueError(f"defined values in state group {name} must be finite")


def validate_pft_input(pft: PFTAxisInput) -> None:
    """JAX-side equivalent of the frozen five-array Gate-B2 boundary."""

    state = np.asarray(pft.pft_state)
    parameters = np.asarray(pft.pft_parameters)
    traits = np.asarray(pft.pft_traits)
    fraction = np.asarray(pft.pft_fraction)
    active = np.asarray(pft.active_pft_mask)
    if state.ndim < 2:
        raise ValueError("pft_state requires an explicit leading PFT axis")
    n_pft = state.shape[0]
    if parameters.ndim != 2 or parameters.shape[0] != n_pft:
        raise ValueError("pft_parameters must share the leading PFT axis")
    if traits.ndim != 2 or traits.shape[0] != n_pft:
        raise ValueError("pft_traits must share the leading PFT axis")
    if fraction.shape != (n_pft,) or active.shape != (n_pft,):
        raise ValueError("PFT fractions and mask must share the leading PFT axis")
    for name, value in {
        "pft_state": state,
        "pft_parameters": parameters,
        "pft_traits": traits,
        "pft_fraction": fraction,
    }.items():
        if value.dtype != np.dtype(np.float64):
            raise TypeError(f"{name} must use float64")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")
    if active.dtype != np.dtype(np.bool_):
        raise TypeError("active_pft_mask must use exact bool dtype")
    if np.any((fraction < 0.0) | (fraction > 1.0)):
        raise ValueError("pft_fraction must lie in [0, 1]")
    if np.any(~active & (fraction != 0.0)):
        raise ValueError("inactive PFT slots must have zero fraction")


def assemble_local_pft_features(pft: PFTAxisInput) -> Any:
    """Flatten within each PFT only; never collapse or identify the PFT axis."""

    state = jnp.asarray(pft.pft_state, dtype=jnp.float64).reshape((pft.pft_state.shape[0], -1))
    traits = jnp.asarray(pft.pft_traits, dtype=jnp.float64)
    fraction = jnp.asarray(pft.pft_fraction, dtype=jnp.float64)[:, None]
    active = jnp.asarray(pft.active_pft_mask, dtype=jnp.bool_)[:, None]
    features = jnp.concatenate((state, traits, fraction), axis=1)
    return jnp.where(active, features, jnp.zeros_like(features))


def fraction_weighted_pft_context(local_features: Any, pft: PFTAxisInput) -> Any:
    """Permutation-invariant global context with inactive-slot isolation."""

    local = jnp.asarray(local_features, dtype=jnp.float64)
    fraction = jnp.asarray(pft.pft_fraction, dtype=jnp.float64)
    active = jnp.asarray(pft.active_pft_mask, dtype=jnp.bool_)
    weights = jnp.where(active, fraction, 0.0)
    total = jnp.sum(weights)
    denominator = jnp.where(total > 0.0, total, jnp.ones_like(total))
    return jnp.sum(local * weights[:, None], axis=0) / denominator


def summarize_state_groups(state: StateFeatureGroups, masks: StateDefinedMasks) -> Any:
    """Return differentiable per-group mean/RMS summaries while retaining inputs."""

    summaries = []
    for name in StateFeatureGroups._fields:
        values = jnp.asarray(getattr(state, name), dtype=jnp.float64)
        defined = jnp.asarray(getattr(masks, name), dtype=jnp.bool_)
        safe = jnp.where(defined, values, jnp.zeros_like(values))
        count = jnp.maximum(jnp.sum(defined, dtype=jnp.float64), 1.0)
        mean = jnp.sum(safe) / count
        rms = jnp.sqrt(jnp.sum(safe * safe) / count + jnp.asarray(1.0e-24, dtype=jnp.float64))
        summaries.extend((mean, rms))
    return jnp.stack(summaries)
