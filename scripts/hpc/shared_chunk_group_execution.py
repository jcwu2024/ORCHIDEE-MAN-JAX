"""Group two canonical expert5 chunks using native lax.map batch_size.

Source: shared_decoder_macro_packed._decode_mapped_rows, lines25–76.
Only the map invocation changes; full128 calls, masks, projection use,
native AD, readout, padding and output restoration follow that source.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import inspect

import jax
import jax.numpy as jnp
from research.daily_coarse_graining.daily_operator import shared_transformer as st
from research.daily_coarse_graining.daily_operator import strong_baseline_architecture as sba

PATCHED_ROWS = []


def validate_map_api():
    if "batch_size" not in inspect.signature(jax.lax.map).parameters:
        raise RuntimeError("native lax.map batch_size is unavailable")
    return {"jax_version": jax.__version__, "signature": str(inspect.signature(jax.lax.map))}


@contextmanager
def patches(config):
    validate_map_api()
    if (config.resolved_decoder_width, config.resolved_decoder_attention_heads,
            config.coordinate_blocks) != (384, 8, 2):
        raise ValueError("requires original width384/eight-head/two-block decoder")
    macro = importlib.import_module("research.daily_coarse_graining.daily_operator.shared_decoder_macro_packed")
    original = macro._decode_mapped_rows

    def decode(shared, *, query, dynamic, dynamic_mask, family_mask, process_mask,
               active, axis_indices, local_indices, projections, config, chunk_size):
        rows = int(query.shape[0])
        if rows != 7542:
            return original(shared, query=query, dynamic=dynamic, dynamic_mask=dynamic_mask,
                family_mask=family_mask, process_mask=process_mask, active=active,
                axis_indices=axis_indices, local_indices=local_indices,
                projections=projections, config=config, chunk_size=chunk_size)
        if chunk_size != 128:
            raise ValueError("G2 requires original128 chunk size")
        padded_rows, chunk_count, pad = 7552, 59, 10
        PATCHED_ROWS.append({"route_shape": rows, "padded_rows": padded_rows,
            "padding_rows": pad, "chunk": chunk_size, "chunk_count": chunk_count,
            "chunk_axis_primitive": "jax.lax.map", "batch_size": 2,
            "full_groups": 29, "remainder_chunks": 1})

        def chunks(value):
            padded = jnp.pad(value, ((0, pad),) + ((0, 0),) * (value.ndim - 1))
            return padded.reshape((chunk_count, chunk_size) + value.shape[1:])

        values = tuple(chunks(value) for value in (
            query, dynamic, dynamic_mask, family_mask, process_mask,
            active, axis_indices, local_indices))

        def decode_chunk(items):
            q, dyn, dyn_mask, fam_mask, proc_mask, on, axis_idx, local_idx = items
            encoded = st._run_coordinate_blocks_reusing_projections(
                q, dyn, dynamic_mask=dyn_mask, family_mask=fam_mask,
                process_mask=proc_mask, carbon_mask=None,
                blocks=shared["coordinate_blocks"], projections=projections,
                config=config, dynamic_projection_indices=(axis_idx, local_idx, on))[:, 0, :]
            raw = sba._linear(shared["readout_second"], jax.nn.gelu(
                sba._linear(shared["readout_first"], encoded)))[:, 0]
            return jnp.where(on, raw, 0.0)

        return jax.lax.map(decode_chunk, values, batch_size=2).reshape(-1)[:rows]

    try:
        macro._decode_mapped_rows = decode
        yield
    finally:
        macro._decode_mapped_rows = original
