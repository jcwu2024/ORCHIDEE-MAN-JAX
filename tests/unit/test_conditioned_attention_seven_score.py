import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.successor_training import BiomassProductivityTrajectories
from scripts.hpc import audit_conditioned_attention_seven_score as audit
from tests.unit.test_conditioned_attention_streamed_integration import (
    Group,
    Prediction,
    Result,
    extract,
    make_batch,
    setup_bootstrap,
    transition,
)


@pytest.mark.parametrize("trace_mode", ["verbose", "compact"])
def test_full_two_window_diagnostic_with_real_statistics_and_streamed_derivatives(monkeypatch, trace_mode):
    bootstrap = setup_bootstrap(monkeypatch)
    original_trace = audit.window._make_trace_evaluator(bootstrap)

    def trajectories(values, micro, *, rematerialize=False):
        trace, _ = original_trace(values, micro)
        pred = extract(trace.result.next_state.other, trace.result.next_state_defined.other,
                       daily_fast_gpp=trace.result.prediction.named_outputs["daily.fast_gpp"], pft_fraction=trace.pft_fraction)
        expected = extract(micro.expected_next_state.other[0], micro.expected_next_state_defined.other[0],
                           daily_fast_gpp=micro.expected_fast_gpp[0], pft_fraction=micro.inputs.pft.pft_fraction[0])
        hard = bootstrap["steps"]["seven_forward"](values, micro)[1]
        return BiomassProductivityTrajectories(pred.fluxes, pred.stocks), BiomassProductivityTrajectories(expected.fluxes, expected.stocks), hard

    def traced(values, micro):
        return original_trace(values, micro)[0], trajectories(values, micro)[0]

    monkeypatch.setattr(audit.window, "_make_trace_evaluator", lambda b: traced)
    bootstrap["steps"]["window_trajectories"] = trajectories
    if trace_mode == "compact":
        import research.daily_coarse_graining.daily_operator.successor_training as training

        bootstrap["context"]["template"] = None
        bootstrap["context"]["runtime"]["contract"] = SimpleNamespace(continuous_state_width=2)
        bootstrap["context"]["scales"].state = Group(jnp.ones(2))
        monkeypatch.setattr(audit.window, "_state_location", lambda c, i: {"global_state_index": i})
        monkeypatch.setattr(training, "rollout_input_from_template", lambda t, s, d, c: t._replace(
            continuous_state=s, discrete_state=d, pft=t.pft._replace(pft_fraction=jax.nn.sigmoid(s))))

        def dynamic(values, current, retained):
            obs = transition(values, current.continuous_state, bootstrap["fixed_parameters"], (current, {}, retained))
            return Result(Group(obs["state"]), Group(jnp.ones_like(obs["state"], dtype=bool)),
                          Prediction({"daily.fast_gpp": obs["gpp"], "unused_debug": jnp.ones(1000)}), current.discrete_state)

        def from_trace(trace, contract):
            predicted = extract(trace.result.next_state.other, trace.result.next_state_defined.other,
                                daily_fast_gpp=trace.result.prediction.named_outputs["daily.fast_gpp"], pft_fraction=trace.pft_fraction)
            return BiomassProductivityTrajectories(predicted.fluxes, predicted.stocks)

        bootstrap["steps"]["transition_builder"] = lambda template: dynamic
        monkeypatch.setattr(training, "trajectories_from_rollout", from_trace)
    values = {"weight": jnp.array([0.2, -0.1], dtype=jnp.float32), "bias": jnp.array(0.03, dtype=jnp.float32)}
    artifact = SimpleNamespace(batch=SimpleNamespace(seven_day=make_batch(7, 0.04)))
    result = audit.experiment(bootstrap, values, artifact, emit=lambda event, data: json.dumps(data), trace_mode=trace_mode)
    assert [r["window"] for r in result["windows"]] == [5, 1]
    assert [r["original_callback_ordinal"] for r in result["windows"]] == [37, 33]
    assert result["reverse_compile_count"] == 1
    assert result["global_statistics_window_count"] == 8
    for row in result["windows"]:
        assert row["hard_failure_count"] == 0
        if trace_mode == "compact":
            assert len(row["recorded_vs_verbose_states"]) == 7
            assert all(x["mask_mismatches"] == 0 for x in row["recorded_vs_verbose_states"])
        assert all(v["agrees_with_production"] for v in row["scores"].values())
        assert all(v["max_abs_difference"] < 1e-12 for v in row["verbose_vs_compact_trajectories"].values())
        for finite_difference in row["finite_differences"]:
            for name in ("production", "observation"):
                assert finite_difference[name]["relative_derivative_error"] < 0.01
                assert finite_difference[name]["central_secant"] < 0
    np.testing.assert_array_equal(values["weight"], np.array([0.2, -0.1], dtype=np.float32))


def test_score_comparison_does_not_promote_known_discrepancy():
    summary = audit.summarize_scores({"production": 0.028094445001546463, "observation_ad": 0.028076108617189763})
    assert summary["production"]["agrees_with_production"]
    assert not summary["observation_ad"]["agrees_with_production"]


def test_qualified_launcher_branch_and_bounded_wrapper():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text()
    start = shared.index('elif [[ "$PYTHON_MODULE" == "scripts.hpc.run_target_oriented_coupled_gpp_conditioned_attention_persistent_training"')
    branch = shared[start:shared.index("elif [[", start + 8)]
    assert '"scripts.hpc.audit_conditioned_attention_seven_score" ||' in branch
    command = shared[shared.index('"$PYTHON" -m "$PYTHON_MODULE"'):]
    assert all(o in branch or o in command for a in audit.window._parser()._actions if a.required for o in a.option_strings)
    wrapper = (root / "scripts/hpc/audit_conditioned_attention_seven_score.sh").read_text()
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-45m}" in wrapper
    assert '--trace-mode "${TRACE_MODE:-verbose}"' in branch
    assert audit._parser().get_default("trace_mode") == "verbose"
    assert "sbatch" not in shared + wrapper
