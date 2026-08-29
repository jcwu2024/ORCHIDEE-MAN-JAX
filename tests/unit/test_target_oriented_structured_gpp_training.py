from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    FROZEN_CONTRACT_SHA256,
    calibrate_structured_gpp_gradient_norms,
    combine_specialist_gradients,
    linearized_structured_gpp_direct_loss,
    load_structured_gpp_training_preflight_contract,
    replace_specialist,
    specialist_tree,
    structured_gpp_direct_loss,
    structured_gpp_direct_statistics,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_error_sufficient_statistics,
)
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    _make_direct_steps,
)


class _PFTInputs(NamedTuple):
    active_pft_mask: object
    pft_fraction: object
    pft_traits: object


class _Inputs(NamedTuple):
    pft: _PFTInputs
    signal: object


class _Target(NamedTuple):
    value: object
    defined: object


class _OneBatch(NamedTuple):
    inputs: _Inputs
    process_targets: object


class _WindowBatch(NamedTuple):
    inputs: _Inputs
    expected_fast_gpp: object
    expected_fast_gpp_defined: object


class _IdentityAssembler:
    @staticmethod
    def assemble(inputs):
        return inputs


class _FakeHead:
    @staticmethod
    def assemble_context(parameters, inputs):
        del parameters, inputs
        return ()

    @staticmethod
    def learned_outputs(parameters, context, inputs):
        del context
        scale = parameters["target_oriented"]["structured_gpp"]["scale"]
        return {"daily.fast_gpp": scale * inputs.signal}


def test_structured_gpp_training_preflight_contract_is_exact_and_disjoint():
    contract = load_structured_gpp_training_preflight_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.calibration_updates == (352, 353, 354, 355, 356, 357, 359, 360)
    assert contract.ephemeral_update["artifact_update"] == 252
    assert contract.ephemeral_update["optimizer_updates_persisted"] == 0
    assert contract.payload["data"]["direct_gpp_teacher_days_per_artifact"] == 208
    assert contract.payload["gradient_calibration"]["gradient_sources"] == ["direct_gpp_208_teacher_days"]
    assert contract.payload["short_horizon_forward_diagnostic"] == {
        "artifact_update": 352,
        "horizons_days": [7, 30],
        "optimizer_updates": 0,
        "gradients_evaluated": False,
        "require_objectives_finite_positive": True,
        "require_physical_hard_failure_count_zero": True,
        "purpose": (
            "prove recursive structured-GPP rollout compatibility before warm start; "
            "joint gradients are calibrated only at the warm-start endpoint"
        ),
    }
    assert contract.payload["planned_bounded_training"]["authorized_by_this_contract"] is False
    assert contract.payload["execution"]["paid_training_authorized"] is False


def test_structured_gpp_direct_loss_masks_inactive_values():
    statistics = structured_gpp_direct_statistics(
        jnp.asarray([1.0, 3.0, 9.0, 7.0]),
        jnp.asarray([1.0, 1.0, 1.0, 1.0]),
        jnp.asarray([True, True, True, False]),
        jnp.ones((4,)),
        jnp.asarray([True, True, False, True]),
    )
    assert float(statistics.sum_squared) == pytest.approx(4.0)
    assert float(statistics.sum_signed) == pytest.approx(2.0)
    assert float(statistics.count) == pytest.approx(2.0)
    assert float(structured_gpp_direct_loss(statistics)) == pytest.approx(1.5)


def test_direct_runner_steps_accept_one_day_and_window_batch_axes():
    runtime = {
        "jax": jax,
        "jnp": jnp,
        "adapter": SimpleNamespace(input_assembler=_IdentityAssembler()),
        "process_statistics": SimpleNamespace(outputs={"daily.fast_gpp": SimpleNamespace(scale=np.ones((1, 3)))}),
    }
    specialist = {"scale": jnp.asarray(2.0)}
    parameters = {"target_oriented": {"structured_gpp": specialist}}
    steps = _make_direct_steps(runtime, _FakeHead(), parameters)

    pft = _PFTInputs(
        active_pft_mask=np.asarray([[True, True, True], [True, True, True]]),
        pft_fraction=np.asarray([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
        pft_traits=np.asarray(
            [
                [[0.0], [1.0], [0.0]],
                [[0.0], [0.0], [0.0]],
            ]
        ),
    )
    signal = np.asarray([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    one = _OneBatch(
        inputs=_Inputs(pft=pft, signal=signal),
        process_targets={
            "daily.fast_gpp": _Target(
                value=np.zeros_like(signal),
                defined=np.ones_like(signal, dtype=np.bool_),
            )
        },
    )
    statistics = steps["one_forward"](specialist, one)
    assert int(statistics.count) == 4
    _value, gradient = steps["one_gradient"](specialist, one, statistics)
    assert np.isfinite(float(gradient["scale"]))
    assert float(gradient["scale"]) > 0.0

    window = _WindowBatch(
        inputs=jax.tree_util.tree_map(lambda value: np.stack((value, value)), one.inputs),
        expected_fast_gpp=np.zeros((2, 2, 1, 3)),
        expected_fast_gpp_defined=np.ones((2, 2, 1, 3), dtype=np.bool_),
    )
    window_statistics = steps["window_forward"](specialist, window)
    assert int(window_statistics.count) == 8


def test_linearized_direct_microbatches_have_exact_full_batch_gradient():
    def statistics(value, offset):
        return structured_gpp_direct_statistics(
            jnp.asarray([value + offset]),
            jnp.asarray([0.0]),
            jnp.asarray([True]),
            jnp.asarray([2.0]),
            jnp.asarray([True]),
        )

    def full(value):
        total = add_error_sufficient_statistics(statistics(value, 0.0), statistics(value, 1.0))
        return structured_gpp_direct_loss(total)

    def accumulated(value):
        first = statistics(value, 0.0)
        second = statistics(value, 1.0)
        total = add_error_sufficient_statistics(first, second)
        return linearized_structured_gpp_direct_loss(first, total) + linearized_structured_gpp_direct_loss(
            second, total
        )

    assert float(jax.grad(full)(1.5)) == pytest.approx(float(jax.grad(accumulated)(1.5)))


def test_structured_gpp_gradient_calibration_uses_direct_p90():
    result = calibrate_structured_gpp_gradient_norms(
        range(1, 9),
    )
    assert result.direct_p90_clip_threshold == pytest.approx(7.3)
    assert result.median_direct_norm == 4.5

    with pytest.raises(ValueError, match="eight direct norms"):
        calibrate_structured_gpp_gradient_norms((1.0,) * 7)
    with pytest.raises(ValueError, match="finite and positive"):
        calibrate_structured_gpp_gradient_norms((0.0,) * 8)


def test_structured_gpp_specialist_tree_replace_and_gradient_combination():
    parameters = {"target_oriented": {"structured_gpp": {"a": jnp.asarray([1.0])}, "parent": object()}}
    specialist = specialist_tree(parameters)
    proposed = {"a": jnp.asarray([2.0])}
    replaced = replace_specialist(parameters, proposed)
    assert replaced["target_oriented"]["structured_gpp"] is proposed
    assert replaced["target_oriented"]["parent"] is parameters["target_oriented"]["parent"]
    assert specialist is parameters["target_oriented"]["structured_gpp"]

    combined = combine_specialist_gradients(
        jax.tree_util,
        ({"a": jnp.asarray([1.0, 2.0])}, {"a": jnp.asarray([3.0, 4.0])}),
        (1.0, 0.5),
    )
    np.testing.assert_allclose(combined["a"], [2.5, 4.0])


def test_structured_gpp_training_preflight_runner_is_ephemeral_and_free_gpu_only():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/run_target_oriented_structured_gpp_training_preflight.py").read_text(encoding="utf-8")
    launcher = (root / "scripts/hpc/run_target_oriented_structured_gpp_training_preflight.sh").read_text(
        encoding="utf-8"
    )
    assert "gradient_parameters" not in runner
    assert "_horizon_gradient" not in runner
    assert "make_trainable_gradient_steps" not in runner
    assert "specialist_tree(parameters)" in runner
    assert "replace_specialist" in runner
    assert '"training_updates_executed": 1' in runner
    assert '"training_updates_persisted": 0' in runner
    assert '"checkpoint_written": False' in runner
    assert '"recovery_written": False' in runner
    assert '"gradients_evaluated": False' in runner
    assert "write_target_oriented_stage_checkpoint" not in runner
    assert "write_distribution_aware_checkpoint" not in runner
    assert "free gln01" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-240m}" in launcher
