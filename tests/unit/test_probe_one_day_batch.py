import gzip
import json

import numpy as np

from scripts.hpc.probe_one_day_batch import (
    BATCH_SIZE,
    EXPECTED_ROWS,
    _add_gradients,
    _hard_rows,
    _hard_summary,
    _scale_gradient,
    catalogue_selection,
    gradient_evaluators,
)


def test_catalogue_selection_resolves_source_and_keeps_exact_first_four(tmp_path):
    source = tmp_path / "prepared_manifest.json"
    source.write_text("{}")
    catalogue = tmp_path / "catalogue.json.gz"
    identity = {"training_contract_sha256": "contract"}
    with gzip.open(catalogue, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "source_manifest": source.name,
                "source_identity": identity,
            },
            handle,
        )
    rows = [
        {
            "artifact_id": artifact_id,
            "batch_row": batch_row,
            "landpoint_id": landpoint_id,
            "year": year,
            "start_row": start_row,
        }
        for artifact_id, batch_row, landpoint_id, year, start_row in EXPECTED_ROWS
    ]
    rows.extend(
        [
            {
                "artifact_id": 2500,
                "batch_row": 3,
                "landpoint_id": "extra",
                "year": 2001,
                "start_row": 10,
            },
            {
                "artifact_id": 2501,
                "batch_row": 4,
                "landpoint_id": "extra-2",
                "year": 2002,
                "start_row": 11,
            },
        ]
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "passed",
                "schema_version": "prepared_one_day_catalogue_v1",
                "catalogue": catalogue.name,
                "first_batch": {"rows": rows},
            }
        )
    )

    loaded_report, loaded_catalogue, source_manifest, selected = catalogue_selection(report)

    assert loaded_report["status"] == "passed"
    assert loaded_catalogue["source_identity"] == identity
    assert source_manifest == source.resolve()
    assert selected == tuple(rows[:BATCH_SIZE])


def test_vmapped_mean_gradient_matches_dynamic_single_sample_gradients():
    import jax
    import jax.numpy as jnp

    def objective(parameters, sample):
        prediction = jnp.vdot(parameters["weight"], sample["features"]) + parameters["bias"]
        loss = (prediction - sample["target"]) ** 2
        hard = {
            "nonfinite": jnp.sum(~jnp.isfinite(prediction)),
            "negative": jnp.asarray(0, dtype=jnp.int32),
        }
        return loss, {"hard": hard, "unused": prediction}

    parameters = {
        "weight": jnp.asarray([0.5, -0.25]),
        "bias": jnp.asarray(0.1),
    }
    samples = {
        "features": jnp.asarray([[1.0, 2.0], [2.0, 0.0], [-1.0, 3.0], [4.0, -2.0]]),
        "target": jnp.asarray([0.0, 1.5, -0.5, 2.0]),
    }
    single, vmapped = gradient_evaluators(jax, jnp, objective)

    (batch_loss, auxiliary), batch_gradient = vmapped(parameters, samples)
    single_results = [
        single(parameters, jax.tree_util.tree_map(lambda value: value[index], samples))
        for index in range(BATCH_SIZE)
    ]
    single_losses = np.asarray([float(result[0][0]) for result in single_results])
    gradient_sum = single_results[0][1]
    for result in single_results[1:]:
        gradient_sum = _add_gradients(jax, gradient_sum, result[1])
    mean_gradient = _scale_gradient(jax, gradient_sum, 1.0 / BATCH_SIZE)

    np.testing.assert_allclose(float(batch_loss), np.mean(single_losses), rtol=1e-6)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(batch_gradient),
        jax.tree_util.tree_leaves(mean_gradient),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    hard = _hard_rows(auxiliary["hard"])
    assert _hard_summary(hard)["all_zero"]

    shifted = {**samples, "target": samples["target"] + 1.0}
    (shifted_loss, _), shifted_gradient = vmapped(parameters, shifted)
    assert not np.isclose(float(batch_loss), float(shifted_loss))
    assert any(
        not np.allclose(left, right)
        for left, right in zip(
            jax.tree_util.tree_leaves(batch_gradient),
            jax.tree_util.tree_leaves(shifted_gradient),
            strict=True,
        )
    )
