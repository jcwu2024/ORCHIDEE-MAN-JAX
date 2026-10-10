"""Opt-in expert5 chunk execution; reuse canonical mathematics and native AD.

Packing/readout: shared_decoder_macro_packed._decode_mapped_rows:25-76.
Blocks: shared_transformer._run_coordinate_blocks_reusing_projections:537-610.
Linear/norm/FFN: strong_baseline_architecture:886-899. No physical processes.
"""
from __future__ import annotations

from contextlib import contextmanager
import sys

import jax
import jax.numpy as jnp
from research.daily_coarse_graining.daily_operator import shared_transformer as st
from research.daily_coarse_graining.daily_operator import strong_baseline_architecture as sba

PATCHED_ROWS = []


@contextmanager
def learned_shape_patches(rows, chunk):
    original_linear, original_norm = st._linear, st._layer_norm
    sba_linear, sba_norm = sba._linear, sba._layer_norm
    def linear(parameters, tokens):
        if tokens.shape[0] != rows:
            return original_linear(parameters, tokens)
        if tokens.shape not in ((rows,1,384),(rows,1,1536),(rows,384),(rows,256)):
            raise ValueError("unexpected same-length tensor outside declared expert5 query/FFN/readout scope")
        shaped = tokens.reshape((-1, chunk) + tokens.shape[1:])
        result = jax.lax.map(lambda v: original_linear(parameters, v), shaped)
        return result.reshape((-1,) + result.shape[2:])
    def norm(tokens, scale, bias):
        if tokens.shape[0] != rows:
            return original_norm(tokens, scale, bias)
        if tokens.shape != (rows,1,384):
            raise ValueError("unexpected same-length tensor outside declared expert5 layer normalization scope")
        shaped = tokens.reshape((-1, chunk) + tokens.shape[1:])
        result = jax.lax.map(lambda v: original_norm(v, scale, bias), shaped)
        return result.reshape((-1,) + result.shape[2:])
    try:
        st._linear = sba._linear = linear
        st._layer_norm = sba._layer_norm = norm
        yield
    finally:
        st._linear, st._layer_norm = original_linear, original_norm
        sba._linear, sba._layer_norm = sba_linear, sba_norm


@contextmanager
def patches(config):
    if (config.resolved_decoder_width, config.coordinate_blocks) != (384,2):
        raise ValueError("requires original two-block width384")
    macro = sys.modules["research.daily_coarse_graining.daily_operator.shared_decoder_macro_packed"]
    original = macro._decode_mapped_rows
    def decode(shared, *, query, dynamic, dynamic_mask, family_mask, process_mask,
               active, axis_indices, local_indices, projections, config, chunk_size):
        if int(query.shape[0]) != 7542:
            return original(shared, query=query, dynamic=dynamic, dynamic_mask=dynamic_mask,
                family_mask=family_mask, process_mask=process_mask, active=active,
                axis_indices=axis_indices, local_indices=local_indices, projections=projections,
                config=config, chunk_size=chunk_size)
        if chunk_size != 128:
            raise ValueError("preserve original expert128 padding")
        if any(value is not None and value.shape[0] == 7552 for projected in projections for value in projected):
            raise ValueError("a projected bank collides with the declared query-row scope")
        PATCHED_ROWS.append({"rows":7542,"padded_rows":7552,"chunk":128,"chunk_count":59})
        def padded(value):
            return jnp.pad(value, ((0,10),)+((0,0),)*(value.ndim-1))
        q,d,m,f,p,on,a,l = (padded(v) for v in
            (query,dynamic,dynamic_mask,family_mask,process_mask,active,axis_indices,local_indices))
        with learned_shape_patches(7552,128):
            encoded = st._run_coordinate_blocks_reusing_projections(q,d,dynamic_mask=m,
                family_mask=f,process_mask=p,carbon_mask=None,blocks=shared["coordinate_blocks"],
                projections=projections,config=config,dynamic_projection_indices=(a,l,on))[:,0,:]
            raw = st._linear(shared["readout_second"],jax.nn.gelu(st._linear(shared["readout_first"],encoded)))[:,0]
        return jnp.where(on,raw,0.0)[:7542]
    try:
        macro._decode_mapped_rows = decode
        yield
    finally:
        macro._decode_mapped_rows = original
