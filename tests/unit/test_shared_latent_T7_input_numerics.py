"""Meaningful bounded input contracts; run with -B and pytest cache disabled."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.hpc import audit_shared_latent_T7_input_numerics as audit
from scripts.hpc.audit_shared_latent_T7_input_numerics import layer_norm_oracle, summary_oracle


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    # This nonexistent directory exercises the ordinary clean-checkout path.
    # Synthetic contracts must never need ignored historical result assets.
    return audit.build_report(diagnostics=tmp_path_factory.mktemp("input-audit") / "absent-assets")


def test_optional_external_artifact299_normalization_and_input_identity():
    """Known scalar primals/scale derivatives, not incidental GPU route truth."""
    required = ("T7-state-inputs-20261009.json", "T7-state-input-ties-20261009.json",
                "T7-state-repeats-20261009.npz")
    missing = [str(audit.DIAGNOSTICS / name) for name in required if not (audit.DIAGNOSTICS / name).is_file()]
    if missing:
        pytest.skip("Optional external artifact299 metadata/identity assets unavailable: " + ", ".join(missing))
    history = audit.load_saved_history(audit.DIAGNOSTICS)
    assert all(asset["executed"] for asset in history["assets"].values())
    rows = history["saved_artifact299_normalization"]
    assert [r["coordinate"] for r in rows] == [434, 2382, 2621, 200]
    assert all(r["historical_primal"]["passed"] for r in rows)
    assert all(r["cast_scale_analytic_jvp"]["passed"] for r in rows)
    assert all(r["raw_coordinate_jvp"] > 0. for r in rows)
    # Do not silently replace the original input with a saved JVP state.
    candidate = history["rejected_complete_input_candidate"]
    assert not candidate["is_required_original_input"]
    assert candidate["actual_sha256"] != candidate["required_original_state_sha256"]
    # Being in a pool with an extremum tie does not mean this direction touches
    # that tied extremum: all four saved directions are non-extremal there.
    assert all(r["coordinate_on_extrema_tie_count"] == 0 for r in history["saved_pool_tie_metadata"])


def test_clean_report_and_explicit_synthetic_only_do_not_claim_history(report, tmp_path, monkeypatch):
    assert report["saved_artifact299_normalization"] == []
    assert report["saved_pool_tie_metadata"] == []
    assert report["rejected_complete_input_candidate"] is None
    assert report["metadata_inputs"] == []
    assert all(asset["availability"] == "unavailable" and not asset["executed"]
               for asset in report["external_history"]["assets"].values())
    assert report["bounded_check_summary"]["saved_scalar_normalization_checks_executed"] == 0
    assert report["bounded_check_summary"]["all_saved_scalar_normalization_checks_passed"] is None
    def forbidden_history(*args, **kwargs):
        raise AssertionError("synthetic-only mode must not probe or load historical assets")
    monkeypatch.setattr(audit, "load_saved_history", forbidden_history)
    monkeypatch.setattr(audit, "synthetic_checks", lambda: report["checks"])
    synthetic_only = audit.build_report(diagnostics=tmp_path, include_history=False)
    assert synthetic_only["external_history"]["mode"] == "disabled_synthetic_only"
    assert synthetic_only["metadata_inputs"] == []
    assert all(asset["availability"] == "not_checked" and not asset["executed"]
               for asset in synthetic_only["external_history"]["assets"].values())


def test_finite_difference_summary_uses_actual_results():
    rows = [{"name": "summary-only-fixture", "FP64_oracle_finite_difference": [
        {"passed": True}, {"passed": False}, {"passed": True}]}]
    summary = audit.summarize_checks(rows, [])
    assert summary["finite_difference_cases_executed"] == 1
    assert summary["finite_difference_cases_converged_at_two_smallest_steps"] == []
    assert summary["all_finite_difference_cases_converged_at_two_smallest_steps"] is False
    rows[0]["FP64_oracle_finite_difference"][1]["passed"] = True
    assert audit.summarize_checks(rows, [])["all_finite_difference_cases_converged_at_two_smallest_steps"] is True
    assert audit.summarize_checks([], [])["all_finite_difference_cases_converged_at_two_smallest_steps"] is None


def test_reconstruction_and_sample_isolation(report):
    checks = report["checks"]
    for row in checks:
        if "serial_vs_vmap_primal" not in row:
            assert row["passed"]
            continue
        assert row["serial_vs_vmap_primal"]["passed"], row["name"]
        assert row["serial_vs_vmap_jvp"]["passed"], row["name"]
        assert row["jvp_vjp_pairing"]["passed"], row["name"]
        assert row["other_sample_zero_tangent"], row["name"]
    reconstruction = checks[0]
    assert reconstruction["input_shape"] == [3854]
    assert reconstruction["FP64_oracle_primal"]["max_absolute_error"] == 0.
    assert reconstruction["FP64_oracle_jvp"]["max_absolute_error"] == 0.


def test_independent_smooth_oracles_and_converged_finite_differences(report):
    for row in report["checks"]:
        if "FP64_oracle_jvp" not in row:
            continue
        if row["name"] == "layer_norm_near_constant_64":
            # A stress probe may expose rounding sensitivity; it is evidence,
            # not a request to relax the production tolerance or suppress it.
            assert "rounding_sensitivity" in row
        else:
            assert row["FP64_oracle_primal"]["passed"], row["name"]
            assert row["FP64_oracle_jvp"]["passed"], row["name"]
        if "FP64_oracle_finite_difference" in row:
            # Require convergence at both smaller displacements. Large-step
            # truncation in a sensitive normalization is not an AD failure.
            assert all(r["passed"] for r in row["FP64_oracle_finite_difference"][-2:]), row["name"]


def test_oracle_invariants_and_nonsmooth_extrema():
    """Independent mathematical constraints, including nonsmooth boundaries."""
    x = np.asarray([-1., -1., 2., 2.])
    dx = np.asarray([2., 0., 3., 0.])
    pool, defined = np.ones((1, 4), bool), np.ones(4, bool)
    y, dy = summary_oracle(x, dx, pool, defined)
    permutation = np.asarray([3, 0, 2, 1])
    yp, dyp = summary_oracle(x[permutation], dx[permutation], pool, defined)
    np.testing.assert_allclose(y, yp, rtol=0, atol=1e-14)
    np.testing.assert_allclose(dy, dyp, rtol=0, atol=1e-14)
    assert dy[0, 2] == 1. and dy[0, 3] == 1.5
    step = 1e-5
    right = (summary_oracle(x + step * dx, dx, pool, defined)[0] - y) / step
    assert abs(right[0, 2]) < 1e-12
    assert abs(right[0, 3] - 3.) < 1e-10
    assert not np.allclose(right[0, 2:], dy[0, 2:])
    # LayerNorm's real formula removes any constant shift and its tangent.
    base = np.linspace(-1., 1., 64)
    direction = np.linspace(-.1, .1, 64)
    y, dy = layer_norm_oracle(base, direction)
    shifted, shifted_tangent = layer_norm_oracle(base + 4., direction + 7.)
    np.testing.assert_allclose(y, shifted, rtol=0, atol=1e-14)
    np.testing.assert_allclose(dy, shifted_tangent, rtol=0, atol=1e-14)
