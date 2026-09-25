"""Parity checks for native-AD field-grouped reference execution."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from research.daily_coarse_graining.daily_operator.shared_decoder_macro_packed import (
    reference_decode_fields,
)
from research.daily_coarse_graining.daily_operator.shared_transformer import SharedTransformerConfig
from scripts.hpc import probe_shared_decoder_macro_packed as probe


@pytest.mark.parametrize("microbatch", (1, 4))
def test_partitioned_reference_sums_loss_shared_and_input_gradients(microbatch):
    shared, payload, masks = probe.fixture(8)
    # Expert 5 appears on opposite sides of the partition boundary.
    selected_fields = (5, 0, 7)
    specs = tuple(probe.FIELD_SPECS[index] for index in selected_fields)
    payload = dict(payload, fields=tuple(payload["fields"][index] for index in selected_fields))
    masks = tuple(masks[index] for index in selected_fields)
    if microbatch == 4:
        samples = [payload]
        for seed in range(20260926, 20260929):
            sample = probe.fixture(8, seed=seed)[1]
            samples.append(dict(
                sample, fields=tuple(sample["fields"][index] for index in selected_fields),
            ))
        payload = jax.tree_util.tree_map(lambda *items: jnp.stack(items), *samples)
    config = SharedTransformerConfig(
        width=8, ffn_width=32, coordinate_blocks=2, readout_width=8,
    )
    total_rows = sum(item[1] for item in specs)

    def one(parameters, inputs):
        fields, _ = probe.prepare_fields(parameters, inputs, masks, specs=specs)
        outputs = jnp.concatenate(reference_decode_fields(parameters, fields, config))
        target = jnp.sin(jnp.arange(outputs.size, dtype=jnp.float32) * 0.13)
        return jnp.mean((outputs - target) ** 2), outputs

    def monolithic(parameters, inputs):
        if microbatch == 1:
            return one(parameters, inputs)
        losses, outputs = jax.vmap(one, in_axes=(None, 0))(parameters, inputs)
        return jnp.mean(losses), outputs

    expected = jax.value_and_grad(monolithic, argnums=(0, 1), has_aux=True)(shared, payload)
    pieces = probe._reference_groups(specs, 2)
    accumulated = None
    offset = 0
    for indices in pieces:
        piece_size = sum(specs[index][1] for index in indices)
        decoder = probe.load_candidate(None)

        def objective(parameters, inputs):
            return probe._reference_piece(
                parameters, inputs, masks=masks, specs=specs, indices=indices,
                config=config, decoder=decoder, output_offset=offset,
                total_rows=total_rows, microbatch=microbatch,
            )

        result = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)(shared, payload)
        if accumulated is None:
            (loss, output), (parameter_gradient, input_gradient) = result
            accumulated = (loss, [output], parameter_gradient, input_gradient)
        else:
            (loss, output), (parameter_gradient, input_gradient) = result
            old_loss, outputs, old_parameter_gradient, old_input_gradient = accumulated
            accumulated = (
                old_loss + loss, outputs + [output],
                probe._tree_add(old_parameter_gradient, parameter_gradient),
                probe._tree_add(old_input_gradient, input_gradient),
            )
        offset += piece_size

    loss, outputs, parameter_gradient, input_gradient = accumulated
    actual = ((loss, jnp.concatenate(outputs, axis=-1)), (parameter_gradient, input_gradient))
    assert probe.compare(actual, expected, rtol=1e-5, atol=1e-6)["passed"]


def test_reference_groups_cover_fields_in_order_without_overflow():
    groups = probe._reference_groups(probe.FIELD_SPECS, 3)
    assert groups == ((0, 1, 2), (3, 4, 5), (6, 7))
    assert tuple(index for group in groups for index in group) == tuple(range(len(probe.FIELD_SPECS)))


def test_reference_checkpoint_replace_preserves_last_complete_file(tmp_path, monkeypatch):
    artifact = tmp_path / "reference.pkl"
    artifact.write_bytes(b"last-complete")

    def fail_dump(_state, _stream, **_kwargs):
        raise OSError("simulated interrupted checkpoint")

    monkeypatch.setattr(probe.pickle, "dump", fail_dump)
    try:
        probe._write_reference_artifact(artifact, {"partial": True})
    except OSError:
        pass
    else:
        raise AssertionError("simulated checkpoint failure did not propagate")
    assert artifact.read_bytes() == b"last-complete"
