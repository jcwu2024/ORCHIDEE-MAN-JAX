import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.successor_training import BiomassProductivityTrajectories
from scripts.hpc import audit_conditioned_attention_seven_score as audit
from tests.unit.test_conditioned_attention_streamed_integration import extract, make_batch, setup_bootstrap


def test_full_two_window_diagnostic_with_real_statistics_and_streamed_derivatives(monkeypatch):
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
    values = {"weight": jnp.array([0.2, -0.1], dtype=jnp.float32), "bias": jnp.array(0.03, dtype=jnp.float32)}
    artifact = SimpleNamespace(batch=SimpleNamespace(seven_day=make_batch(7, 0.04)))
    result = audit.experiment(bootstrap, values, artifact, emit=lambda event, data: json.dumps(data))
    assert [r["window"] for r in result["windows"]] == [5, 1]
    assert [r["original_callback_ordinal"] for r in result["windows"]] == [37, 33]
    assert result["reverse_compile_count"] == 1
    assert result["global_statistics_window_count"] == 8
    for row in result["windows"]:
        assert row["hard_failure_count"] == 0
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
    assert "sbatch" not in shared + wrapper
