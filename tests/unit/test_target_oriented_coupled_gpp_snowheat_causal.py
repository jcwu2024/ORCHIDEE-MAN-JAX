from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_snowheat_causal import (
    classify_snowheat_day5_intervention,
    factor_log_attribution,
    resolve_coupled_gpp_snowheat_sources,
)
from scripts.hpc import audit_target_oriented_coupled_gpp_snowheat_causal as runner


class _Inputs(NamedTuple):
    continuous_state: object
    discrete_state: object


def _leaf(key: str, start: int) -> SimpleNamespace:
    return SimpleNamespace(
        key=key,
        discrete=False,
        start=start,
        stop=start + 2,
        shape=(2,),
        axis_names=("snow_layer",),
    )


def test_snowheat_sources_resolve_both_mirrors_and_gpp_routes():
    canopy = _leaf("diffuco_previous_step_state.snowheat", 0)
    water = _leaf("hydrol_previous_step_state.snowheat", 2)
    markov = SimpleNamespace(state_leaves=(canopy, water))
    owner = SimpleNamespace(
        leaves=(
            SimpleNamespace(
                key=canopy.key,
                role="mirror_of_canonical_owner",
                canonical_owner=water.key,
            ),
            SimpleNamespace(key=water.key, role="canonical_owner", canonical_owner=water.key),
        )
    )
    layout = SimpleNamespace(
        expert_names=("canopy_gpp",),
        source_family_names=("state_canopy", "state_water"),
        state_field_family_index=np.asarray((0, 1), dtype=np.int32),
        source_allowed=np.asarray(((True, True),), dtype=np.bool_),
    )

    leaves, records = resolve_coupled_gpp_snowheat_sources(markov, owner, layout)

    assert tuple(leaf.key for leaf in leaves) == (water.key, canopy.key)
    assert records[water.key]["source_family"] == "state_water"
    assert records[canopy.key]["source_family"] == "state_canopy"
    assert all(record["canopy_gpp_admitted"] for record in records.values())


@pytest.mark.parametrize(
    ("matched", "expected"),
    (
        (1.2, "snowheat_path_causal_for_day5_jump"),
        (2.0, "snowheat_path_partial_for_day5_jump"),
        (2.5, "snowheat_path_not_explanatory_for_day5_jump"),
    ),
)
def test_snowheat_day5_classification_includes_frozen_boundaries(matched, expected):
    result = classify_snowheat_day5_intervention(
        control_gpp=1.0,
        treatment_gpp=3.0,
        matched_snowheat_gpp=matched,
    )
    assert result["classification"] == expected


def test_factor_log_attribution_reconstructs_routed_gpp_ratio():
    control = {
        "normalization_scale": np.asarray((7.0, 1.0)),
        "potential": np.asarray((7.0, 1.0)),
        "limitation": np.asarray((7.0, 1.0)),
        "correction": np.asarray((7.0, 1.0)),
        "vcmax25_factor": np.asarray((7.0, 1.0)),
        "routed_gpp": np.asarray(((7.0, 4.0),)),
    }
    treatment = {
        "normalization_scale": np.asarray((7.0, 2.0)),
        "potential": np.asarray((7.0, 0.5)),
        "limitation": np.asarray((7.0, 1.25)),
        "correction": np.asarray((7.0, 1.0)),
        "vcmax25_factor": np.asarray((7.0, 2.0)),
        "routed_gpp": np.asarray(((7.0, 10.0),)),
    }

    result = factor_log_attribution(control, treatment, pft_index=1)

    assert result["routed_gpp_log_ratio"] == pytest.approx(np.log(2.5))
    assert result["factor_log_ratio_sum"] == pytest.approx(np.log(2.5))
    assert result["factor_log_ratio_residual"] == pytest.approx(0.0, abs=1.0e-15)
    assert result["dominant_absolute_log_ratio_factor"] == "normalization_scale"


def test_scenario_day_starts_replaces_only_selected_treatment_coordinates():
    microbatch = SimpleNamespace(
        inputs=_Inputs(
            continuous_state=jnp.asarray(
                (
                    (
                        (1.0, 2.0, 3.0, 4.0),
                        (101.0, 102.0, 103.0, 104.0),
                        (201.0, 202.0, 203.0, 204.0),
                    ),
                )
            ),
            discrete_state={"flag": jnp.asarray((((1,), (9,), (9,)),))},
        )
    )
    trace = SimpleNamespace(
        result=SimpleNamespace(
            next_state=SimpleNamespace(
                other=jnp.asarray(
                    (
                        (10.0, 20.0, 30.0, 40.0),
                        (11.0, 21.0, 31.0, 41.0),
                        (12.0, 22.0, 32.0, 42.0),
                    )
                )
            ),
            next_discrete_state={"flag": jnp.asarray(((2,), (3,), (4,)))},
        )
    )
    reference = jnp.asarray(
        (
            (1000.0, 2000.0, 3000.0, 4000.0),
            (1001.0, 2001.0, 3001.0, 4001.0),
            (1002.0, 2002.0, 3002.0, 4002.0),
        )
    )

    _, starts, discrete = runner._scenario_day_starts(
        SimpleNamespace(jax=jax, jnp=jnp),
        microbatch,
        trace,
        state_mask=jnp.asarray((False, True, False, True)),
        reference=reference,
    )

    np.testing.assert_array_equal(
        np.asarray(starts),
        np.asarray(
            (
                (1.0, 2000.0, 3.0, 4000.0),
                (10.0, 2001.0, 30.0, 4001.0),
                (11.0, 2002.0, 31.0, 4002.0),
            )
        ),
    )
    np.testing.assert_array_equal(np.asarray(discrete["flag"]), np.asarray(((1,), (2,), (3,))))


def test_factorization_comparison_uses_relative_tolerance_for_nonzero_gpp():
    observed = np.asarray(((5.0,),), dtype=np.float64)
    trace = SimpleNamespace(
        result=SimpleNamespace(
            prediction=SimpleNamespace(named_outputs={runner.GPP_FIELD: observed}),
        )
    )
    comparison = runner._factorization_comparison(
        trace,
        {"routed_gpp": observed + 4.0e-7},
    )
    assert comparison["maximum_absolute_error"] > runner.FACTORIZATION_ATOL
    assert comparison["within_tolerance"] is True


def test_snowheat_runner_delegates_reconstruction_and_routes_required_options():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text(encoding="utf-8")
    wrapper = (root / "scripts/hpc/run_target_oriented_coupled_gpp_snowheat_causal.sh").read_text(encoding="utf-8")
    required = {
        option
        for action in runner._parser()._actions
        if action.required
        for option in action.option_strings
        if option.startswith("--")
    }
    assert sorted(option for option in required if option not in shared and option not in wrapper) == []
    run_source = inspect.getsource(runner.run)
    assert run_source.count("reconstruct_localized_jump_parameters(args, bootstrap=bootstrap)") == 1
    assert 'bootstrap["head"].layout' in run_source
    assert 'bootstrap["layout"]' not in run_source
    assert "_measure_direct_artifact_gradients" not in run_source
    assert "sbatch" not in wrapper
    assert "free gln01" in wrapper
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-60m}" in wrapper
