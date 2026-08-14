from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.long_horizon_objective import (
    MaskedTrajectory,
)
from research.daily_coarse_graining.daily_operator.named_biomass_tendency import (
    NAMED_BIOMASS_TENDENCY_FIELDS,
    biomass_stock_identity_residual,
    fit_named_biomass_tendency_scales,
    named_biomass_tendency_objective,
)
from scripts.hpc.prepare_successor_named_biomass_tendency_scales import (
    _select_records,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COEFFICIENTS = {"tendency_mse": 1.0, "tendency_bias_squared": 1.0}
SCALES = {
    horizon: {
        "aboveground_biomass_carbon": 2.0,
        "belowground_biomass_carbon": 4.0,
    }
    for horizon in (1, 7, 30)
}


def _trajectory(value, defined=None):
    array = jnp.asarray(value, dtype=jnp.float64)
    mask = jnp.ones(array.shape, dtype=jnp.bool_) if defined is None else defined
    return MaskedTrajectory(array, jnp.asarray(mask, dtype=jnp.bool_))


def _stocks(above, below, *, above_defined=None, below_defined=None):
    return {
        "aboveground_biomass_carbon": _trajectory(above, above_defined),
        "belowground_biomass_carbon": _trajectory(below, below_defined),
    }


def test_teacher_teacher_named_biomass_tendency_loss_is_exact_zero():
    teacher = _stocks(jnp.ones((30, 3)), jnp.ones((30, 3)) * 2.0)
    result = named_biomass_tendency_objective(
        teacher,
        teacher,
        scales=SCALES,
        coefficients=COEFFICIENTS,
    )

    assert float(result.total) == 0.0
    assert float(result.tendency_mse) == 0.0
    assert float(result.tendency_bias_squared) == 0.0


def test_named_biomass_tendency_uses_equal_fields_and_horizons():
    expected = _stocks(jnp.zeros((30, 2)), jnp.zeros((30, 2)))
    predicted = _stocks(jnp.ones((30, 2)), jnp.ones((30, 2)) * 2.0)
    result = named_biomass_tendency_objective(
        predicted,
        expected,
        scales=SCALES,
        coefficients=COEFFICIENTS,
    )

    for horizon in (1, 7, 30):
        for name in NAMED_BIOMASS_TENDENCY_FIELDS:
            error = result.by_horizon[horizon].errors[name]
            assert float(error.mse) == pytest.approx(0.25)
            assert float(error.mean_bias) == pytest.approx(0.5)
        assert float(result.by_horizon[horizon].macro_mse) == pytest.approx(0.25)
        assert float(result.by_horizon[horizon].macro_bias_squared) == pytest.approx(
            0.25
        )
    assert float(result.total) == pytest.approx(0.5)


def test_named_biomass_tendency_excludes_undefined_field_from_macro():
    expected = _stocks(jnp.zeros((1, 2)), jnp.zeros((1, 2)))
    predicted = _stocks(
        jnp.ones((1, 2)) * 2.0,
        jnp.ones((1, 2)) * 100.0,
        below_defined=jnp.zeros((1, 2), dtype=jnp.bool_),
    )
    result = named_biomass_tendency_objective(
        predicted,
        expected,
        scales={1: SCALES[1]},
        coefficients=COEFFICIENTS,
        horizons=(1,),
    )

    assert int(
        result.by_horizon[1].errors["belowground_biomass_carbon"].count
    ) == 0
    assert float(result.tendency_mse) == pytest.approx(1.0)
    assert float(result.tendency_bias_squared) == pytest.approx(1.0)


def test_terminal_residual_is_exactly_the_shared_start_tendency_residual():
    start = jnp.asarray([10.0, 20.0])
    predicted = jnp.asarray([13.0, 19.0])
    teacher = jnp.asarray([11.0, 18.0])

    terminal_residual = predicted - teacher
    tendency_residual = (predicted - start) - (teacher - start)

    np.testing.assert_array_equal(terminal_residual, tendency_residual)


def test_fit_named_biomass_tendency_scales_uses_train_change_and_floor():
    initial = _stocks([10.0, 20.0, 30.0], [5.0, 10.0, 15.0])
    above = np.broadcast_to(np.asarray([10.0, 20.0, 30.0]), (30, 3)).copy()
    below = np.broadcast_to(np.asarray([5.0, 10.0, 15.0]), (30, 3)).copy()
    above[0] += np.asarray([1.0, 2.0, 3.0])
    above[6] += 0.01
    above[29] += np.asarray([2.0, 4.0, 6.0])
    below[0] += 0.01
    below[6] += np.asarray([2.0, 4.0, 6.0])
    below[29] += np.asarray([3.0, 6.0, 9.0])
    teacher = _stocks(above, below)

    result = fit_named_biomass_tendency_scales(
        initial,
        teacher,
        stock_scales={
            "aboveground_biomass_carbon": 100.0,
            "belowground_biomass_carbon": 200.0,
        },
        floor_ratio=1.0e-3,
    )

    assert result.scales[1]["aboveground_biomass_carbon"] == pytest.approx(1.0)
    assert result.scales[7]["aboveground_biomass_carbon"] == pytest.approx(0.1)
    assert result.scales[1]["belowground_biomass_carbon"] == pytest.approx(0.2)
    assert result.scales[7]["belowground_biomass_carbon"] == pytest.approx(2.0)
    assert result.audit["records"][7]["aboveground_biomass_carbon"][
        "floor_applied"
    ]
    assert result.audit["records"][1]["aboveground_biomass_carbon"]["count"] == 3


def test_named_biomass_tendency_is_jittable_with_finite_gradient():
    expected = _stocks(jnp.zeros((30, 2)), jnp.zeros((30, 2)))

    def objective(parameter):
        predicted = _stocks(
            jnp.ones((30, 2)) * parameter,
            jnp.ones((30, 2)) * (2.0 * parameter),
        )
        return named_biomass_tendency_objective(
            predicted,
            expected,
            scales=SCALES,
            coefficients=COEFFICIENTS,
        ).total

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.0))

    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(gradient))
    assert float(gradient) > 0.0


def test_biomass_stock_identity_residual_is_exact_and_masked():
    stocks = _stocks([[1.0, 2.0]], [[3.0, 4.0]])
    stocks["total_live_biomass_carbon"] = _trajectory([[4.0, 6.0]])
    result = biomass_stock_identity_residual(stocks)

    np.testing.assert_array_equal(result.value, np.zeros((1, 2)))
    np.testing.assert_array_equal(result.defined, np.ones((1, 2), dtype=bool))


def test_named_biomass_tendency_rejects_scale_and_horizon_drift():
    stocks = _stocks(jnp.zeros((7, 1)), jnp.zeros((7, 1)))
    with pytest.raises(ValueError, match="horizon registry drift"):
        named_biomass_tendency_objective(
            stocks,
            stocks,
            scales={1: SCALES[1]},
            coefficients=COEFFICIENTS,
            horizons=(1, 7),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        named_biomass_tendency_objective(
            stocks,
            stocks,
            scales={1: dict(SCALES[1]) | {"aboveground_biomass_carbon": 0.0}},
            coefficients=COEFFICIENTS,
            horizons=(1,),
        )
    with pytest.raises(ValueError, match="exceeds"):
        named_biomass_tendency_objective(
            stocks,
            stocks,
            scales=SCALES,
            coefficients=COEFFICIENTS,
        )


def test_named_biomass_tendency_manifest_is_hash_bound_and_bounded():
    path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_named_biomass_tendency_v1.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["optimized_stock_fields"] == list(NAMED_BIOMASS_TENDENCY_FIELDS)
    assert manifest["scale_policy"]["stock_relative_floor_ratio"] == 1.0e-3
    assert not manifest["negative_design"]["reopens_process_increment_v2"]
    assert manifest["preparation_gates"]["optimizer_updates"] == 0
    assert manifest["training_gate"]["status"].startswith("thresholds_require")


def test_named_biomass_tendency_scale_selection_is_deterministic_and_unique():
    entries = [
        {
            "update": index,
            "landpoint_id": f"point-{index % 4}",
            "year": 1961 + (index // 4),
            "sha256": f"{index:064x}",
        }
        for index in range(1, 21)
    ]
    first = _select_records(entries, point_year_count=8, seed=17)
    second = _select_records(tuple(reversed(entries)), point_year_count=8, seed=17)

    assert [record["update"] for record in first] == [
        record["update"] for record in second
    ]
    assert len({(record["landpoint_id"], record["year"]) for record in first}) == 8


def test_named_biomass_tendency_scale_launcher_is_free_cpu_and_zero_update():
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_named_biomass_tendency_scale_fit.sh"
    ).read_text()
    source = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "prepare_successor_named_biomass_tendency_scales.py"
    ).read_text()

    assert "biomass tendency scales require a free CPU test node" in launcher
    assert "JAX_PLATFORMS=cpu" in launcher
    assert "#SBATCH" not in launcher
    assert '"training_updates": 0' in source
    assert '"checkpoint_written": False' in source
    assert '"validation_values_evaluated": False' in source
    assert "successor_adamw_update" not in source
