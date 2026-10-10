"""Small CPU checks for daily mask preservation and strict-repeat separation."""
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from scripts.hpc import probe_shared_latent_corrected_daily as probe


def result():
    values = np.ones((4, 3), dtype=np.float64)
    values[:, 2] = np.nan
    mask = np.ones((4, 3), dtype=bool)
    mask[:, 2] = False
    diagnostic = {"predicted": {"x": values.copy()}, "expected": {"x": values.copy()},
                  "predicted_defined": {"x": mask.copy()}, "expected_defined": {"x": mask.copy()}}
    return ((np.float64(0.1), {"hard": {"total": np.int64(0)}, "state_diagnostics": diagnostic}),
            ({"p": np.ones(2, dtype=np.float32)}, np.ones((4, 3854), dtype=np.float64)))


def test_masked_undefined_values_do_not_relax_defined_parity():
    baseline, actual = result(), result()
    actual[0][1]["state_diagnostics"]["predicted"]["x"][:, 2] = np.inf
    comparison = probe.compare_repeats(jax, actual, baseline, SimpleNamespace(state_leaves=[]))
    assert comparison["forward_and_gates"]["passed"]
    actual[0][1]["state_diagnostics"]["predicted"]["x"][0, 0] = 2
    assert not probe.compare_repeats(jax, actual, baseline, SimpleNamespace(state_leaves=[]))["forward_and_gates"]["passed"]


def test_mask_and_parameter_failures_remain_distinct_from_state_drift():
    baseline, actual = result(), result()
    actual[1][1][0, 33] = 2
    comparison = probe.compare_repeats(jax, actual, baseline, SimpleNamespace(state_leaves=[]))
    assert comparison["forward_and_gates"]["passed"]
    assert comparison["all_parameter_gradients"]["passed"]
    assert comparison["all_initial_state_gradients"]["failed"] == 1
    assert comparison["all_initial_state_gradients"]["failures"][0]["artifact_row"] == 0
    actual[1][0]["p"][0] = 2
    assert not probe.compare_repeats(jax, actual, baseline, SimpleNamespace(state_leaves=[]))["all_parameter_gradients"]["passed"]
    actual[0][1]["state_diagnostics"]["predicted_defined"]["x"][0, 0] = False
    assert not probe.compare_repeats(jax, actual, baseline, SimpleNamespace(state_leaves=[]))["forward_and_gates"]["passed"]


def test_replay_shape_change_is_hard_error():
    baseline, actual = result(), result()
    actual = actual[0], (actual[1][0], actual[1][1][:3])
    with pytest.raises(ValueError, match="ABI"):
        probe.compare_repeats(jax, actual, baseline, SimpleNamespace(state_leaves=[]))


def test_absent_original_asset_is_rejected_before_backend_setup(tmp_path, monkeypatch):
    asset = tmp_path / "asset"
    asset.write_text("fixture")
    output = tmp_path / "candidate.json"
    manifest = {"admission_mode": "corrected_performance_diagnostic", "candidate_parameters": str(asset),
                "input_bundle": str(asset), "original_checkpoint": str(asset),
                "architectures": {"candidate": {"daily": {"report": str(output), "executable": str(output.with_suffix('.executable.pkl'))}},
                                  "original": {"daily": {"report": str(asset), "executable": str(tmp_path / 'absent.pkl'), "verification_report": str(asset)}}}}
    def adjacent(name):
        assert name == "probe_shared_latent_dual", "backend must not be requested for missing assets"
        return SimpleNamespace(read_manifest=lambda path: manifest)
    monkeypatch.setattr(probe, "adjacent", adjacent)
    argv = []
    for name in ("manifest", "source-profile", "prepared-context", "parameters", "field-adapter", "backbone-adapter", "transformer-module", "latent-module", "turnover-overlay", "runtime-preflight"):
        argv.extend(("--" + name, str(asset)))
    argv.extend(("--output", str(output)))
    with pytest.raises(FileNotFoundError, match="absent.pkl"):
        probe.run(probe.parser().parse_args(argv))
