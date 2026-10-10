"""Exact expert5 whole chunk-axis vmap for the original daily-loss screen.

Packing/readout follows shared_decoder_macro_packed._decode_mapped_rows.
Coordinate math and masks remain canonical; only its chunk-axis lax.map becomes
vmap. The sample-axis vmap and native JAX AD remain in the public objective.
"""
from __future__ import annotations

from contextlib import contextmanager
import sys
import importlib

import jax
import jax.numpy as jnp
from research.daily_coarse_graining.daily_operator import shared_transformer as st
from research.daily_coarse_graining.daily_operator import strong_baseline_architecture as sba

PATCHED_ROWS = []


@contextmanager
def patches(config):
    if (config.resolved_decoder_width, config.resolved_decoder_attention_heads,
            config.coordinate_blocks) != (384, 8, 2):
        raise ValueError("requires original expert5 width384/eight-head/two-block decoder")
    macro_name = "research.daily_coarse_graining.daily_operator.shared_decoder_macro_packed"
    macro = sys.modules.get(macro_name)
    if macro is None:
        macro = importlib.import_module(macro_name)
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
            raise ValueError("expert5 whole-vmap requires the original 128-row chunks")
        padded_rows = 7552
        chunk_count = 59
        pad = padded_rows - rows
        PATCHED_ROWS.append({"route_shape": rows, "padded_rows": padded_rows,
                             "padding_rows": pad, "chunk": chunk_size,
                             "chunk_count": chunk_count,
                             "chunk_axis_primitive": "jax.vmap"})

        def chunks(value):
            padded = jnp.pad(value, ((0, pad),) + ((0, 0),) * (value.ndim - 1))
            return padded.reshape((chunk_count, chunk_size) + value.shape[1:])

        values = tuple(chunks(value) for value in (
            query, dynamic, dynamic_mask, family_mask, process_mask,
            active, axis_indices, local_indices,
        ))

        def decode_chunk(items):
            q, dyn, dyn_mask, fam_mask, proc_mask, on, axis_idx, local_idx = items
            encoded = st._run_coordinate_blocks_reusing_projections(
                q, dyn, dynamic_mask=dyn_mask, family_mask=fam_mask,
                process_mask=proc_mask, carbon_mask=None,
                blocks=shared["coordinate_blocks"], projections=projections,
                config=config, dynamic_projection_indices=(axis_idx, local_idx, on),
            )[:, 0, :]
            raw = sba._linear(shared["readout_second"], jax.nn.gelu(
                sba._linear(shared["readout_first"], encoded)))[:, 0]
            return jnp.where(on, raw, 0.0)

        mapped = jax.vmap(decode_chunk)(values)
        return mapped.reshape(-1)[:rows]

    try:
        macro._decode_mapped_rows = decode
        yield
    finally:
        macro._decode_mapped_rows = original
