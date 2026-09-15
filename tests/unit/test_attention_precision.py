from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.fresh_start import build_fresh_head
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
    NeuralPrecisionPolicy,
    StrongBaselineConfig,
    StrongBaselineContext,
    _attention,
    _linear_parameters,
)
from research.daily_coarse_graining.daily_operator.target_oriented_attention import _source_family_bank
from scripts.hpc.run_mixed_learning import lowered_precision_summary

jax.config.update('jax_enable_x64', True)


def _case():
    keys = jax.random.split(jax.random.PRNGKey(915), 6)
    parameters = {name: _linear_parameters(key, 32, 32)
                  for name, key in zip(('query', 'key', 'value', 'output'), keys)}
    q = jax.random.normal(keys[4], (2, 5, 32), dtype=jnp.float32)
    kv = jax.random.normal(keys[5], (2, 7, 32), dtype=jnp.float32)
    mask = jnp.ones((2, 5, 7), dtype=bool).at[:, 0, :].set(False).at[:, :, -1].set(False)
    return parameters, q, kv, mask


def test_explicit_fp32_removes_promoted_matrix_products():
    parameters, q, kv, mask = _case()
    legacy = jax.jit(lambda p, q, kv: _attention(p, q, kv, heads=4, mask=mask))
    candidate = jax.jit(lambda p, q, kv: _attention(p, q, kv, heads=4, mask=mask,
                                                  score_dtype=jnp.float32))
    old = lowered_precision_summary(legacy.lower(parameters, q, kv))
    new = lowered_precision_summary(candidate.lower(parameters, q, kv))
    assert old['matrix_operation_result_types']['f64'] == 1
    assert new['matrix_operation_result_types'].get('f64', 0) == 0
    assert new['matrix_operation_result_types']['f32'] == 6
    np.testing.assert_allclose(candidate(parameters, q, kv), legacy(parameters, q, kv),
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('score_dtype', [None, jnp.float32])
def test_masked_precision_path_has_finite_native_gradients_and_jvp(score_dtype):
    p, q, kv, mask = _case()

    def loss(p, q, kv):
        result = _attention(p, q, kv, heads=4, mask=mask, score_dtype=score_dtype)
        return jnp.mean(jnp.square(result))

    value, gradients = jax.value_and_grad(loss, argnums=(0, 1, 2))(p, q, kv)
    assert np.isfinite(value)
    assert all(np.all(np.isfinite(g)) for g in jax.tree_util.tree_leaves(gradients))
    # Fully masked queries and masked key/value positions carry no input gradient.
    np.testing.assert_array_equal(gradients[1][:, 0, :], 0.)
    np.testing.assert_array_equal(gradients[2][:, -1, :], 0.)
    direction = jnp.ones_like(q) / 10
    _, tangent = jax.jvp(lambda q: loss(p, q, kv), (q,), (direction,))
    np.testing.assert_allclose(tangent, jnp.vdot(gradients[1], direction), rtol=1e-5, atol=1e-6)


def test_neural_precision_policy_is_default_off_and_independently_selectable():
    legacy = NeuralPrecisionPolicy()
    attention_only = NeuralPrecisionPolicy(attention_scores_fp32=True)
    pooling_only = NeuralPrecisionPolicy(source_family_pooling_fp32=True)

    assert StrongBaselineConfig().neural_precision == legacy
    assert legacy.attention_score_dtype is None
    assert legacy.source_family_pooling_dtype == jnp.float64
    assert attention_only.attention_score_dtype == jnp.float32
    assert attention_only.source_family_pooling_dtype == jnp.float64
    assert pooling_only.attention_score_dtype is None
    assert pooling_only.source_family_pooling_dtype == jnp.float32


def _pooling_case(tokens):
    context = StrongBaselineContext(
        source_tokens=tokens,
        source_mask=jnp.asarray((True, True, False, True)),
        source_family_index=jnp.asarray((0, 0, 1, 1), dtype=jnp.int32),
        expert_axis_tokens=None,
        expert_axis_mask=None,
        pft_tokens=None,
        pft_mask=None,
    )
    layout = SimpleNamespace(
        source_family_names=("forcing", "state"),
        source_allowed=np.asarray(((True, True),), dtype=bool),
    )
    return context, layout


def test_fp32_source_family_pooling_removes_fp64_accumulator_and_preserves_gradients():
    tokens = jnp.asarray(
        ((1.0, 3.0), (3.0, 7.0), (100.0, 100.0), (5.0, 11.0)),
        dtype=jnp.float32,
    )

    def pooled(tokens, policy):
        context, layout = _pooling_case(tokens)
        return _source_family_bank(context, layout, 0, neural_precision=policy)[0]

    legacy_policy = NeuralPrecisionPolicy()
    candidate_policy = NeuralPrecisionPolicy(source_family_pooling_fp32=True)
    legacy = jax.jit(lambda value: pooled(value, legacy_policy))
    candidate = jax.jit(lambda value: pooled(value, candidate_policy))
    legacy_jaxpr = str(jax.make_jaxpr(lambda value: pooled(value, legacy_policy))(tokens))
    candidate_jaxpr = str(jax.make_jaxpr(lambda value: pooled(value, candidate_policy))(tokens))

    assert "new_dtype=float64" in legacy_jaxpr
    assert "new_dtype=float64" not in candidate_jaxpr
    assert candidate(tokens).dtype == jnp.float32
    np.testing.assert_allclose(candidate(tokens), legacy(tokens), rtol=1e-6, atol=1e-6)
    candidate_gradient = jax.grad(lambda value: jnp.sum(candidate(value)))(tokens)
    assert np.all(np.isfinite(candidate_gradient))
    np.testing.assert_array_equal(candidate_gradient[2], 0.0)


def test_all_active_ordinary_attention_callers_bind_the_shared_policy():
    root = Path(__file__).resolve().parents[2]
    sources = (
        "strong_baseline_architecture.py",
        "target_oriented_attention.py",
        "target_oriented_distribution_aware_maintenance.py",
        "target_oriented_structured_gpp.py",
        "target_oriented_coupled_gpp_carbon_feedback.py",
    )
    call_count = 0
    for name in sources:
        path = root / "research" / "daily_coarse_graining" / "daily_operator" / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_attention"
        )
        for call in calls:
            call_count += 1
            assert any(keyword.arg == "score_dtype" for keyword in call.keywords), (
                f"{name}:{call.lineno} does not bind the neural precision policy"
            )
    assert call_count == 11


class _Normalization(NamedTuple):
    continuous_mean: np.ndarray
    continuous_tail_threshold: float | None
    continuous_tail_mask: np.ndarray | None


@dataclass(frozen=True)
class _FreshHeadStub:
    config: StrongBaselineConfig
    contract: object
    input_normalization: _Normalization
    warm_snow_free_boundary: bool = False
    apply_gsmean_applicability: bool = False


def test_build_fresh_head_binds_one_shared_precision_policy(monkeypatch):
    import research.daily_coarse_graining.daily_operator.fresh_start as fresh_start

    policy = NeuralPrecisionPolicy(attention_scores_fp32=True, source_family_pooling_fp32=True)
    head = _FreshHeadStub(
        config=StrongBaselineConfig(),
        contract=SimpleNamespace(
            state_leaves=(SimpleNamespace(key="slowproc_stomate_previous_step_state.DOC", start=1, stop=3),)
        ),
        input_normalization=_Normalization(np.zeros((4,)), None, None),
    )
    observed = []

    def preserve(current, *args, **kwargs):
        del args, kwargs
        observed.append(current.config.neural_precision)
        return current

    monkeypatch.setattr(fresh_start, "build_target_oriented_distribution_aware_maintenance_head", preserve)
    monkeypatch.setattr(fresh_start, "build_target_oriented_structured_gpp_head", preserve)
    monkeypatch.setattr(fresh_start, "build_target_oriented_coupled_gpp_carbon_feedback_head", preserve)
    monkeypatch.setattr(fresh_start, "build_target_oriented_coupled_gpp_conditioned_attention_head", preserve)

    result = build_fresh_head(head, object(), neural_precision=policy)
    assert observed == [policy] * 4
    assert result.config.neural_precision is policy
    np.testing.assert_array_equal(result.input_normalization.continuous_tail_mask, (False, True, True, False))
