from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc.probe_litter_mixed_admission import (
    COMPONENTS,
    _artifact_paths,
    _checkpoint_method,
    _combine_component_gradients,
    _gradient_bundle_identity,
    _load_gradient_bundle,
    _matched_start_record,
    _save_gradient_bundle,
    _select_batches,
    _select_weight,
    _tree_gram_report,
    _validate_protocol,
    parser,
)

jax.config.update("jax_enable_x64", True)


def _protocol():
    return {
        "schema_version": "litter_mixed_admission_v1",
        "source_update": 4000,
        "calibration_batch_indices": [4000, 4001],
        "update_batch_indices": list(range(4000, 4020)),
        "admission_updates": 2,
        "updates_per_arm": 20,
        "daily_batch": 64,
        "microbatch": 8,
        "outer_weights": {"daily": 1.0, "T7": 0.5, "T30": 0.5},
        "weight_selection": {
            "target_median_ratio": 0.5,
            "maximum_ratio": 1.0,
            "maximum_weight": 0.001,
        },
        "litter_carbon_stock_change": {
            "weight": 0.1,
            "penalty": "pseudo_huber",
            "scale_source_pft_index": 13,
            "minimum_scale_gC_m2_day": 0.01,
        },
        "daily_coefficients_old": [1.0, 0.1, 0.1, 0.0],
        "daily_coefficients_litter_only": [0.0, 0.0, 0.0, 1.0],
        "development_monitor": "unchanged_8_T7_and_8_T30_initial_and_after_each_arm",
        "scientific_acceptance": False,
    }


def _tree(value):
    return {
        "a": jnp.asarray([value, 2.0 * value], dtype=jnp.float64),
        "b": jnp.asarray(-value, dtype=jnp.float64),
    }


def _combine(left, right, left_weight, right_weight):
    return jax.tree_util.tree_map(
        lambda a, b: left_weight * a + right_weight * b, left, right
    )


def test_protocol_freezes_two_calibration_batches_two_update_gate_and_twenty_updates():
    admitted = _validate_protocol(_protocol())
    assert admitted["calibration_indices"] == (4000, 4001)
    assert admitted["admission_updates"] == 2
    assert admitted["update_indices"] == tuple(range(4000, 4020))
    assert admitted["updates_per_arm"] == 20

    changed = _protocol()
    changed["update_batch_indices"][4] = 3999
    with pytest.raises(ValueError, match="successive"):
        _validate_protocol(changed)

    changed = _protocol()
    changed["admission_updates"] = 3
    with pytest.raises(ValueError, match="update 2"):
        _validate_protocol(changed)


def test_selected_batches_require_exact_successive_indices_and_microbatch_geometry():
    batches = []
    for index in range(4020):
        batches.append(SimpleNamespace(
            batch_index=index,
            draw_count=64,
            microbatches=tuple(SimpleNamespace(draw_count=8) for _ in range(8)),
        ))
    selected = _select_batches({"batches": batches}, range(4000, 4020))
    assert [batch.batch_index for batch in selected] == list(range(4000, 4020))

    batches[4019] = SimpleNamespace(
        batch_index=4019,
        draw_count=64,
        microbatches=tuple(SimpleNamespace(draw_count=8) for _ in range(7)),
    )
    with pytest.raises(ValueError, match="microbatch-8"):
        _select_batches({"batches": batches}, range(4000, 4020))


def test_component_merge_is_exact_daily_plus_litter_plus_half_t7_and_t30():
    gradients = {
        "daily_old": _tree(1.0),
        "litter": _tree(10.0),
        "T7": _tree(2.0),
        "T30": _tree(-4.0),
    }
    observed = _combine_component_gradients(_combine, gradients, litter_weight=0.001)
    expected = jax.tree_util.tree_map(
        lambda daily, litter, t7, t30: daily + 0.001 * litter + 0.5 * t7 + 0.5 * t30,
        gradients["daily_old"], gradients["litter"], gradients["T7"], gradients["T30"],
    )
    for actual, wanted in zip(
        jax.tree_util.tree_leaves(observed), jax.tree_util.tree_leaves(expected), strict=True
    ):
        np.testing.assert_allclose(actual, wanted, rtol=0.0, atol=0.0)

    old = _combine_component_gradients(_combine, gradients, litter_weight=0.0)
    assert not np.allclose(np.asarray(old["a"]), np.asarray(observed["a"]))

    # JAX canonicalizes ordinary dict keys, so cache reuse cannot depend on
    # Python insertion order after device placement.
    device_sorted = jax.device_put({name: gradients[name] for name in reversed(COMPONENTS)})
    sorted_observed = _combine_component_gradients(
        _combine, device_sorted, litter_weight=0.001
    )
    for actual, wanted in zip(
        jax.tree_util.tree_leaves(sorted_observed),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_allclose(actual, wanted, rtol=0.0, atol=0.0)


def test_full_parameter_gram_and_shared_weight_selection_bind_component_order():
    gradients = {
        "daily_old": _tree(1.0),
        "litter": _tree(2.0),
        "T7": _tree(-0.5),
        "T30": _tree(0.25),
    }

    def gram(trees):
        return jnp.stack([
            jnp.stack([
                sum(jnp.vdot(a, b) for a, b in zip(
                    jax.tree_util.tree_leaves(left),
                    jax.tree_util.tree_leaves(right),
                    strict=True,
                ))
                for right in trees
            ])
            for left in trees
        ])

    report = _tree_gram_report(jax, gradients, jax.jit(gram))
    assert report["components"] == list(COMPONENTS)
    assert np.asarray(report["matrix"]).shape == (4, 4)
    assert report["raw_norms"]["litter"] == pytest.approx(
        2.0 * report["raw_norms"]["daily_old"]
    )

    selected = _select_weight([report["matrix"], report["matrix"]], _validate_protocol(_protocol()))
    assert selected["components"] == list(COMPONENTS)
    assert 0.0 < selected["selected_weight"] <= 0.001
    assert max(
        panel["weighted_litter_to_old_mixed_ratio"] for panel in selected["panels"]
    ) <= 1.0


def test_gradient_bundle_partial_recovery_and_exact_identity(tmp_path):
    batch = SimpleNamespace(
        batch_index=4000,
        as_metadata=lambda: {"batch_index": 4000, "rows": [[1, 2]]},
    )
    identity = _gradient_bundle_identity(
        source_checkpoint=tmp_path / "source.pkl",
        source_update=4000,
        source_checkpoint_identity={"method": {"name": "source"}},
        source_git_head="source-head",
        runner_git_head="runner-head",
        plan_identities={"daily": "identity"},
        batch=batch,
        recurrent_rows={"T7": {"identifier": "a"}, "T30": {"identifier": "b"}},
        protocol=_protocol(),
    )
    path = tmp_path / "gradient.pkl"
    partial_gradients = {"daily_old": {"x": np.asarray([1.0, 2.0])}}
    _save_gradient_bundle(
        path,
        identity=identity,
        gradients=partial_gradients,
        gram=None,
        losses={"daily_old": 1.25},
        hard_counts={"daily_old": {"invalid": 0}},
    )
    partial = _load_gradient_bundle(path, expected_identity=identity, require_complete=False)
    assert partial["completed_components"] == ["daily_old"]
    with pytest.raises(ValueError, match="incomplete"):
        _load_gradient_bundle(path, expected_identity=identity)

    complete_gradients = {
        name: {"x": np.asarray([index + 1.0, index + 2.0])}
        for index, name in enumerate(COMPONENTS)
    }
    matrix = np.eye(4, dtype=np.float64)
    _save_gradient_bundle(
        path,
        identity=identity,
        gradients=complete_gradients,
        gram={"matrix": matrix.tolist()},
        losses={name: index + 0.5 for index, name in enumerate(COMPONENTS)},
        hard_counts={name: {"invalid": 0} for name in COMPONENTS},
    )
    complete = _load_gradient_bundle(path, expected_identity=identity)
    assert complete["completed_components"] == list(COMPONENTS)

    changed = dict(identity, runner_git_head="different")
    with pytest.raises(ValueError, match="different source metadata"):
        _load_gradient_bundle(path, expected_identity=changed)


def test_matched_start_checkpoint_method_and_artifact_names_are_unambiguous(tmp_path):
    source = tmp_path / "source.pkl"
    matched = _matched_start_record(source, 4000, {"old": 4000, "selected": 4000})
    assert matched["exact_same_parameter_payload"]
    with pytest.raises(ValueError, match="same source AdamW"):
        _matched_start_record(source, 4000, {"old": 4000, "selected": 4001})

    method = _checkpoint_method(
        source_checkpoint=source,
        source_update=4000,
        selected_weight=3.0e-4,
        arm="selected",
        protocol=_protocol(),
        source_git_head="abc123",
    )
    assert method["arm"] == "selected"
    assert method["selected_weight"] == pytest.approx(3.0e-4)
    assert method["claim"] == "bounded_diagnostic_not_formal_mixed_training_checkpoint"
    assert _artifact_paths(tmp_path / "result.json", update=4000).name.endswith(
        ".calibration-4000.pkl"
    )
    assert _artifact_paths(tmp_path / "result.json", arm="old", update=2).name.endswith(
        ".old-update-000002.checkpoint.pkl"
    )


def test_cli_inherits_source_checkpoint_inputs_and_adds_only_mixed_admission_protocol(tmp_path):
    values = [
        "--dataset-manifest", str(tmp_path / "dataset.json"),
        "--acceptance-report", str(tmp_path / "acceptance.json"),
        "--base-statistics", str(tmp_path / "base.npz"),
        "--process-statistics", str(tmp_path / "process.npz"),
        "--fast-statistics", str(tmp_path / "fast.npz"),
        "--prepared-manifest", str(tmp_path / "prepared.json"),
        "--output", str(tmp_path / "result.json"),
        "--mixed-protocol", str(tmp_path / "mixed.json"),
        "--catalogue-report", str(tmp_path / "catalogue.json"),
        "--recurrent-schedule", str(tmp_path / "schedule.json"),
        "--selection", str(tmp_path / "selection.json"),
        "--execution-policy", str(tmp_path / "execution.json"),
        "--training-report", str(tmp_path / "training.json"),
        "--checkpoint", str(tmp_path / "source.pkl"),
        "--mixed-admission-protocol", str(tmp_path / "admission.json"),
    ]
    args = parser().parse_args(values)
    assert args.mixed_admission_protocol == tmp_path / "admission.json"
    assert args.checkpoint == tmp_path / "source.pkl"
    assert args.expected_update == 4000
    assert not hasattr(args, "admission_protocol")
