from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import research.daily_coarse_graining.daily_operator.shared_recurrent_training as module
from research.daily_coarse_graining.daily_operator.long_horizon_objective import MaskedTrajectory
from research.daily_coarse_graining.daily_operator.shared_training import compose_shared_groups
from research.daily_coarse_graining.daily_operator.shared_trajectory_loss import (
    STOCK_KEYS,
    SharedComponentTrajectories,
    SharedTrajectoryHardCounts,
)


class PFT(NamedTuple):
    pft_fraction: object
    active_pft_mask: object
    pft_traits: object


class Inputs(NamedTuple):
    continuous_state: object
    discrete_state: object
    day_index: object
    pft: PFT


class TinyTerms(NamedTuple):
    flux: object
    stock: object
    living_stock: object
    nonliving_stock: object
    hard_counts: SharedTrajectoryHardCounts


def _initial_masks(value=True):
    return {key: jnp.asarray([value]) for key in STOCK_KEYS}


def test_prepare_window_selects_one_window_broadcasts_retained_and_uses_start_mask(monkeypatch):
    windows, horizon, width = 2, 7, 3
    continuous = np.arange(windows * horizon * width).reshape(windows, horizon, width)
    inputs = Inputs(
        continuous_state=continuous,
        discrete_state={"flag": np.arange(windows * horizon).reshape(windows, horizon)},
        day_index=np.stack((np.arange(1, 8), np.arange(20, 27))),
        pft=PFT(
            pft_fraction=np.full((windows, horizon, 2), 0.5),
            active_pft_mask=np.ones((windows, horizon, 2), dtype=bool),
            pft_traits=np.zeros((windows, horizon, 2, 1)),
        ),
    )
    batch = SimpleNamespace(
        inputs=inputs,
        expected_next_state=SimpleNamespace(other=continuous + 1000),
        expected_next_state_defined=SimpleNamespace(other=np.ones_like(continuous, dtype=bool)),
        expected_fast_gpp=np.zeros((windows, horizon, 1, 2)),
        expected_fast_gpp_defined=np.ones((windows, horizon, 1, 2), dtype=bool),
        expected_next_discrete_state={"flag": np.arange(windows * horizon).reshape(windows, horizon) + 1},
        day_inputs={"owner": np.asarray([[11.0], [29.0]])},
    )
    captured = {}

    class Assembler:
        def assemble(self, first):
            return SimpleNamespace(
                state=SimpleNamespace(other=first.continuous_state),
                state_defined=SimpleNamespace(other=jnp.isfinite(first.continuous_state)),
            )

    def initial_mask(state, defined, gpp, first, contract):
        del defined, gpp, contract
        captured["state"] = np.asarray(state.other)
        captured["day"] = int(first.day_index)
        return _initial_masks()

    monkeypatch.setattr(module, "initial_component_stock_defined", initial_mask)
    head = SimpleNamespace(input_assembler=Assembler(), contract=object())
    case = module.prepare_shared_window(head, batch, window_index=1, horizon=horizon)

    np.testing.assert_array_equal(case.initial_continuous, continuous[1, 0])
    assert int(case.initial_discrete["flag"]) == int(inputs.discrete_state["flag"][1, 0])
    np.testing.assert_array_equal(captured["state"], continuous[1, 0])
    assert captured["day"] == 20
    np.testing.assert_array_equal(case.days[0].continuous_state, continuous[1])
    np.testing.assert_array_equal(case.days[1]["owner"], np.full((horizon, 1), 29.0))
    np.testing.assert_array_equal(case.targets["state"], continuous[1] + 1000)
    assert case.loss_context.initial_stock_defined is case.targets["initial_stock_defined"]


def test_public_canonical_step_uses_recurrent_state_not_later_teacher_template(monkeypatch):
    import research.daily_coarse_graining.daily_operator.canonical_rollout as canonical

    class Template(NamedTuple):
        continuous_state: object
        discrete_state: object
        pft: PFT

    def replace_state(template, state, discrete, contract):
        del contract
        return template._replace(continuous_state=state, discrete_state=discrete)

    def bind(**_kwargs):
        def transition(parameters, current, retained):
            del retained
            next_state = current.continuous_state + parameters
            return SimpleNamespace(
                next_state=SimpleNamespace(other=next_state),
                next_state_defined=SimpleNamespace(other=jnp.ones_like(next_state, dtype=bool)),
                next_discrete_state={"flag": current.discrete_state["flag"] + 1},
                prediction=SimpleNamespace(named_outputs={"daily.fast_gpp": jnp.zeros((1, 2))}),
                diagnostics={"water_budget_residual": 0.0, "carbon_budget_residual": 0.0},
            )
        return transition

    monkeypatch.setattr(canonical, "rollout_input_from_template", replace_state)
    monkeypatch.setattr(canonical, "bind_daily_operator_dynamic_retained_inputs", bind)
    head = SimpleNamespace(
        input_assembler=object(),
        learned_schema=SimpleNamespace(fields=()),
        contract=object(),
    )
    step = canonical.make_canonical_step(head, object())
    pft = PFT(jnp.asarray([0.5, 0.5]), jnp.asarray([True, True]), jnp.zeros((2, 1)))
    first = Template(jnp.asarray([100.0]), {"flag": jnp.asarray(90)}, pft)
    second = Template(jnp.asarray([200.0]), {"flag": jnp.asarray(91)}, pft)
    state, discrete = jnp.asarray([0.0]), {"flag": jnp.asarray(0)}
    state, discrete, _ = step(1.0, state, discrete, (first, None))
    state, discrete, _ = step(1.0, state, discrete, (second, None))

    np.testing.assert_allclose(state, [2.0])
    assert int(discrete["flag"]) == 2


def _fake_recurrent_factory(
    monkeypatch,
    *,
    negative_stock=False,
    science_hard=(2, 3, 4, 5),
    physical_failure=7,
):
    def step_factory(_head, _template):
        def step(parameters, state, discrete, day):
            del day
            next_state = state + parameters
            observation = {
                "defined": jnp.asarray([True, False]),
                "gpp": jnp.zeros((1, 2)),
                "gpp_defined": jnp.ones((1, 2), dtype=bool),
                "fraction": jnp.asarray([0.5, 0.5]),
                "active": jnp.asarray([True, True]),
                "discrete": {"flag": discrete["flag"]},
                "budgets": jnp.zeros((2,)),
                "learned_output_nonfinite": jnp.asarray(0, dtype=jnp.int32),
            }
            return next_state, discrete, observation
        return step

    class Science:
        def terms(self, parameters, initial, observations, targets, context):
            del initial, context
            assert "initial_stock_defined" in observations
            assert "initial_stock_defined" in targets
            residual = jnp.sum((observations["state"] - targets["state"]) ** 2)
            hard_values = tuple(jnp.asarray(value) for value in science_hard)
            hard = SharedTrajectoryHardCounts(*hard_values, sum(hard_values))
            return TinyTerms(residual + parameters, 2.0 * residual, 0.25, 0.5, hard)

    monkeypatch.setattr(module, "make_canonical_step", step_factory)
    monkeypatch.setattr(module, "make_shared_trajectory_loss", lambda **_kwargs: Science())
    monkeypatch.setattr(
        module,
        "physical_counts",
        lambda *_args: {"budget_failure": jnp.asarray(physical_failure)},
    )

    stock_value = -1.0 if negative_stock else 1.0

    def trajectories(_observation, _contract):
        stocks = {
            key: MaskedTrajectory(jnp.asarray([[stock_value]]), jnp.asarray([[True]]))
            for key in STOCK_KEYS
        }
        return SharedComponentTrajectories(fluxes={}, stocks=stocks)

    monkeypatch.setattr(module, "shared_component_trajectories", trajectories)


def test_checkpointed_recurrent_loss_matches_plain_native_gradient_and_exposes_hard_counts(monkeypatch):
    _fake_recurrent_factory(monkeypatch, negative_stock=True)
    head = SimpleNamespace(contract=object())
    scales = SimpleNamespace(
        state_declared_dynamic=SimpleNamespace(other=jnp.asarray([False, True]))
    )
    science_scales = {
        "flux_scales": {"unused": 1.0},
        "stock_change_scales": {"unused": 1.0},
    }
    rollout, checkpointed = module.make_shared_recurrent_objective(
        head, object(), scales, science_scales
    )
    context = module.SharedRecurrentLossContext(
        expected_discrete={"flag": jnp.zeros((3,), dtype=jnp.int32)},
        templates=jnp.zeros((3,)),
        # A supervision-ineligible initial stock must not hide a physical
        # negative endpoint from the unmasked recurrent hard gate.
        initial_stock_defined=_initial_masks(False),
    )
    initial = jnp.asarray([0.0, 0.0])
    discrete = {"flag": jnp.asarray(0, dtype=jnp.int32)}
    days = jnp.arange(3)
    targets = {
        "state": jnp.zeros((3, 2)),
        "defined": jnp.asarray([[True, True], [True, True], [True, True]]),
        "initial_stock_defined": context.initial_stock_defined,
    }

    def plain(parameter):
        return rollout.loss(parameter, initial, discrete, days, targets, context)

    def checked(parameter):
        return checkpointed(parameter, initial, discrete, days, targets, context)[0]

    parameter = jnp.asarray(0.2)
    np.testing.assert_allclose(plain(parameter), checked(parameter), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(jax.grad(plain)(parameter), jax.grad(checked)(parameter), rtol=1e-12)
    plain_gradient = jax.grad(plain)(parameter)
    tangent = jax.jvp(plain, (parameter,), (jnp.asarray(1.0),))[1]
    assert np.isfinite(float(tangent)) and float(tangent) != 0.0
    np.testing.assert_allclose(tangent, plain_gradient, rtol=1e-12, atol=1e-12)

    compiled = jax.jit(jax.value_and_grad(checkpointed, has_aux=True))
    (value, auxiliary), gradient = compiled(
        parameter, initial, discrete, days, targets, context
    )
    assert np.isfinite(float(value)) and np.isfinite(float(gradient))
    assert int(auxiliary["hard"]["budget_failure"]) == 7
    assert int(auxiliary["hard"]["science_nonfinite_prediction"]) == 2
    assert int(auxiliary["hard"]["science_nonfinite_target"]) == 3
    assert int(auxiliary["hard"]["science_mask_mismatch"]) == 4
    assert int(auxiliary["hard"]["science_missing_valid_term"]) == 5
    assert int(auxiliary["hard"]["shared_stock_negative"]) == len(STOCK_KEYS)
    assert int(auxiliary["declared_mask_change"]) == 3


def test_finite_matching_recurrent_case_reports_no_hard_failure(monkeypatch):
    _fake_recurrent_factory(
        monkeypatch,
        negative_stock=False,
        science_hard=(0, 0, 0, 0),
        physical_failure=0,
    )
    head = SimpleNamespace(contract=object())
    scales = SimpleNamespace(
        state_declared_dynamic=SimpleNamespace(other=jnp.asarray([False, True]))
    )
    rollout, checkpointed = module.make_shared_recurrent_objective(
        head,
        object(),
        scales,
        {"flux_scales": {"unused": 1.0}, "stock_change_scales": {"unused": 1.0}},
    )
    del rollout
    context = module.SharedRecurrentLossContext(
        expected_discrete={"flag": jnp.zeros((2,), dtype=jnp.int32)},
        templates=jnp.zeros((2,)),
        initial_stock_defined=_initial_masks(),
    )
    targets = {
        "state": jnp.zeros((2, 2)),
        "defined": jnp.asarray([[True, False], [True, False]]),
        "initial_stock_defined": context.initial_stock_defined,
    }
    (value, auxiliary), gradient = jax.jit(
        jax.value_and_grad(checkpointed, has_aux=True)
    )(
        jnp.asarray(0.1),
        jnp.zeros((2,)),
        {"flag": jnp.asarray(0, dtype=jnp.int32)},
        jnp.arange(2),
        targets,
        context,
    )
    assert np.isfinite(float(value)) and np.isfinite(float(gradient))
    assert all(int(count) == 0 for count in auxiliary["hard"].values())
    assert int(auxiliary["declared_mask_change"]) == 0


def test_joint_daily_contribution_has_exact_fixed_horizon_value_and_gradient(monkeypatch):
    def daily_factory(*_args, **_kwargs):
        def objective(parameter, sample):
            del sample
            auxiliary = {
                "process_day": parameter,
                "flux": 2.0 * parameter,
                "stock": 3.0 * parameter,
                "hard": {},
            }
            return 2.0 * parameter, auxiliary
        return objective

    monkeypatch.setattr(module, "make_shared_daily_objective", daily_factory)
    objective = module.make_shared_joint_daily_objective(None, None, None, None, None)
    parameter = jnp.asarray(4.0)
    daily_value = objective(parameter, None)[0]
    t7_flux, t7_stock = 5.0 * parameter, 7.0 * parameter
    t30_flux, t30_stock = 11.0 * parameter, 13.0 * parameter
    assembled = daily_value + (t7_flux + t7_stock + t30_flux + t30_stock) / 9.0
    expected = compose_shared_groups(
        parameter,
        2.0 * parameter,
        3.0 * parameter,
        recurrent_flux=(t7_flux, t30_flux),
        recurrent_stock=(t7_stock, t30_stock),
    )
    np.testing.assert_allclose(assembled, expected, rtol=0.0, atol=1e-12)
    assembled_grad = jax.grad(
        lambda value: objective(value, None)[0]
        + (5.0 * value + 7.0 * value + 11.0 * value + 13.0 * value) / 9.0
    )(parameter)
    expected_grad = jax.grad(
        lambda value: compose_shared_groups(
            value,
            2.0 * value,
            3.0 * value,
            recurrent_flux=(5.0 * value, 11.0 * value),
            recurrent_stock=(7.0 * value, 13.0 * value),
        )
    )(parameter)
    np.testing.assert_allclose(assembled_grad, expected_grad, rtol=0.0, atol=1e-12)
