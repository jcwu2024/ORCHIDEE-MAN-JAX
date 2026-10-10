"""Hoist only expert5 raw projected-bank gathers before its original128 map.

Neural provenance: shared_decoder_macro_packed._decode_mapped_rows:25-76;
shared_transformer._run_coordinate_blocks_reusing_projections:537-610 and
_attention_from_projected_segments:451-534. No physical/source process changes.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from research.daily_coarse_graining.daily_operator import shared_transformer as st
from research.daily_coarse_graining.daily_operator import strong_baseline_architecture as sba

PATCHED_ROWS = []


def validate_original_helpers():
    if st._linear is not sba._linear or st._ffn is not sba._ffn or st._layer_norm is not sba._layer_norm:
        raise ValueError('gather-only execution requires original FP32 helper aliases')
    expected=sba.__name__
    if any(fn.__module__!=expected for fn in (sba._linear,sba._ffn,sba._layer_norm)):
        raise ValueError('strong baseline helper was replaced')
    return {'linear_ffn_layer_norm_original_identity':True,'helper_module':expected,
        'helper_file':str(Path(sba.__file__).resolve()),
        'helper_source_sha256':hashlib.sha256(Path(sba.__file__).read_bytes()).hexdigest(),
        'shared_transformer_file':str(Path(st.__file__).resolve()),
        'shared_transformer_sha256':hashlib.sha256(Path(st.__file__).read_bytes()).hexdigest()}


def gathered_decode(shared, *, query, dynamic, dynamic_mask, family_mask,
                    process_mask, active, axis_indices, local_indices,
                    projections, config, chunk_size, macro, include_readout=True):
    """Keep all learned/attention operations in the original per-chunk body."""
    rows = int(query.shape[0])
    chunk_size = min(chunk_size,rows)
    padded_rows = ((rows+chunk_size-1)//chunk_size)*chunk_size
    chunk_count,pad = padded_rows//chunk_size,padded_rows-rows

    def padded(value):
        return jnp.pad(value, ((0,pad),)+((0,0),)*(value.ndim-1))

    def reshape_padded(value):
        return value.reshape((chunk_count,chunk_size)+value.shape[1:])

    values = tuple(reshape_padded(padded(value)) for value in (
        query,dynamic,dynamic_mask,family_mask,process_mask,active))
    # Padding before selection preserves index0 on the ten original padding
    # rows. Bias replacement remains inside the map, retaining its AD reduction.
    padded_axis, padded_local = padded(axis_indices), padded(local_indices)
    selected = tuple(tuple(reshape_padded(value) for value in (
        projected.axis_key[padded_axis], projected.axis_value[padded_axis],
        projected.local_key[padded_local,None,:], projected.local_value[padded_local,None,:]))
        for projected in projections)

    def decode_chunk(items):
        (q,dyn,dyn_mask,fam_mask,proc_mask,on), selected_blocks = items
        for block,projected,(axis_key,axis_value,local_key,local_value) in zip(
                shared['coordinate_blocks'],projections,selected_blocks,strict=True):
            dynamic_key = jnp.concatenate((axis_key,local_key),axis=1)
            dynamic_value = jnp.concatenate((axis_value,local_value),axis=1)
            dynamic_key = jnp.where(on[:,None,None],dynamic_key,block['attention']['key']['bias'])
            dynamic_value = jnp.where(on[:,None,None],dynamic_value,block['attention']['value']['bias'])
            message = st._attention_from_projected_segments(
                block['attention'],q,dyn,dynamic_key=dynamic_key,dynamic_value=dynamic_value,
                family_key=projected.family_key,family_value=projected.family_value,family_mask=fam_mask,
                dynamic_mask=dyn_mask,process_key=projected.process_key,process_value=projected.process_value,
                process_mask=proc_mask,carbon_key=projected.carbon_key,carbon_value=projected.carbon_value,
                carbon_mask=None,heads=config.resolved_decoder_attention_heads,
                normalization_epsilon=config.normalization_epsilon)
            q = st._layer_norm(q+message,block['attention_norm_scale'],block['attention_norm_bias'])
            q = st._layer_norm(q+st._ffn(block['ffn'],q),block['ffn_norm_scale'],block['ffn_norm_bias'])
        if not include_readout:
            return q
        encoded=q[:,0,:]
        raw=macro._linear(shared['readout_second'],jax.nn.gelu(macro._linear(shared['readout_first'],encoded)))[:,0]
        return jnp.where(on,raw,0.0)

    mapped=jax.lax.map(decode_chunk,(values,selected))
    return mapped.reshape((-1,) if include_readout else (padded_rows,)+query.shape[1:])[:rows]


def component(blocks, values, masks, indices, config, chunk):
    """Original captured coordinate-only scope, excluding the public readout."""
    validate_original_helpers()
    if (config.resolved_decoder_width,config.resolved_decoder_attention_heads,config.coordinate_blocks)!=(384,8,2):
        raise ValueError('component requires original width384/eight-head/two-block route')
    query,dynamic,family,process,carbon,axis,local=values
    if query.shape!=(7542,1,384) or chunk!=128:
        raise ValueError('component requires original expert5 rows7542/chunk128')
    if any(value.dtype!=jnp.float32 for value in jax.tree_util.tree_leaves((blocks,values))):
        raise ValueError('captured neural parameter/input dtype is not original float32')
    if carbon is not None or masks[3] is not None:
        raise ValueError('captured expert5 component is noncarbon')
    projections=[]
    for block in blocks:
        pairs=[st._project_key_value(block['attention'],bank) if bank is not None else (None,None)
               for bank in (family,process,carbon,axis,local)]
        projections.append(st._DecoderBlockProjection(*(value for pair in pairs for value in pair)))
    return gathered_decode({'coordinate_blocks':blocks},query=query,dynamic=dynamic,
        dynamic_mask=masks[1],family_mask=masks[0],process_mask=masks[2],active=indices[2],
        axis_indices=indices[0],local_indices=indices[1],projections=tuple(projections),
        config=config,chunk_size=chunk,macro=None,include_readout=False)


@contextmanager
def patches(config):
    if (config.resolved_decoder_width,config.resolved_decoder_attention_heads,
            config.coordinate_blocks)!=(384,8,2):
        raise ValueError('requires original width384/eight-head/two-block route')
    name='research.daily_coarse_graining.daily_operator.shared_decoder_macro_packed'
    macro=sys.modules.get(name) or importlib.import_module(name)
    validate_original_helpers()
    if macro._linear is not sba._linear:
        raise ValueError('gather-only execution requires original FP32 aliases')
    original=macro._decode_mapped_rows

    def decode(shared, **arguments):
        if int(arguments['query'].shape[0])!=7542:
            return original(shared,**arguments)
        if arguments['chunk_size']!=128:
            raise ValueError('preserve exact original128 map')
        projections=arguments['projections']
        if len(projections)!=2 or any(p.carbon_key is not None or p.carbon_value is not None for p in projections):
            raise ValueError('expert5 original two-block/noncarbon projection scope drift')
        PATCHED_ROWS.append({'rows':7542,'padded_rows':7552,'padding_rows':10,'chunk':128,'chunk_count':59,
            'raw_gathers_hoisted_per_block':4,'blocks':2,'chunk_axis_primitive':'jax.lax.map',
            'all_learned_attention_normalization_readout_inside_original_chunk':True,
            'concat_active_where_bias_inside_original_chunk':True,
            'actual_macro_file':str(Path(macro.__file__).resolve()),
            'actual_macro_sha256':hashlib.sha256(Path(macro.__file__).read_bytes()).hexdigest()})
        return gathered_decode(shared,**arguments,macro=macro)

    try:
        macro._decode_mapped_rows=decode
        yield
    finally:
        macro._decode_mapped_rows=original
