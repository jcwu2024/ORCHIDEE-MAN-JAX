"""Prepared normalized-state fan-out intervention; no JAX on metadata paths.

This is a diagnostic precision choice, not a production policy or model change.
The native source functions remain authoritative; artificial summary cotangents
do not represent the public recurrent loss.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

COMMIT = 'c599236bcf57b745e9bafed678298010f4c3b58f'
SOURCE = 'research/daily_coarse_graining/daily_operator/strong_baseline_architecture.py'
CONDITIONING = 'research/daily_coarse_graining/daily_operator/input_conditioning.py'
PREPARED = 'outputs/diagnostics/shared-bank-adjoint-proved-binary-T7-20261003/next-component-preparation'
COORDINATES = (33, 199, 428)

def _function(tree, name):
    matches = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(matches) != 1: raise ValueError('source function is not unique: ' + name)
    return matches[0]

def _source_pin(root, path, names):
    local = (root / path).read_text(encoding='utf-8')
    frozen = subprocess.run(['git', 'show', COMMIT + ':' + path], cwd=root, check=True, text=True, capture_output=True).stdout
    actual_tree, frozen_tree = ast.parse(local), ast.parse(frozen)
    records = []
    for name in names:
        actual, saved = _function(actual_tree, name), _function(frozen_tree, name)
        if ast.dump(actual, include_attributes=False) != ast.dump(saved, include_attributes=False):
            raise ValueError('component source differs from immutable checkout: ' + name)
        records.append(dict(file=path, function=name, start=actual.lineno, end=actual.end_lineno, immutable_AST_equal=True))
    return actual_tree, records

def prepare_plan(root, output_dir=None):
    """Read source/JSON only; never evaluate normalization, model, AD or JAX."""
    root = Path(root)
    tree, provenance = _source_pin(root, SOURCE, ('_masked_summary', '_mask_inactive_pft_elements', 'assemble_strong_baseline_context'))
    _, condition = _source_pin(root, CONDITIONING, ('standardize_continuous',))
    function = _function(tree, 'assemble_strong_baseline_context')
    assignments = [node for node in function.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'state' for target in node.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Call) or not isinstance(assignments[0].value.func, ast.Name) or assignments[0].value.func.id != 'standardize_continuous':
        raise ValueError('single shared normalized state boundary changed')
    widening = ast.parse("state = jnp.asarray(state, dtype=jnp.float64)").body[0]
    function.body.insert(function.body.index(assignments[0]) + 1, widening)
    ast.fix_missing_locations(tree)
    compile(tree, '<prospective-state-fanout-overlay>', 'exec')  # syntax only
    prepared = Path(output_dir) if output_dir else root / ('runtime/' + PREPARED if (root / 'runtime').exists() and not (root / PREPARED).exists() else PREPARED)
    metadata_path = prepared / 'component-inputs.metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    bridge_path = prepared / 'component-input-bridge.json'
    bridge = json.loads(bridge_path.read_text()) if bridge_path.exists() else None
    return dict(status='source_AST_prepared_no_execution', immutable_checkout=COMMIT,
                boundary='one standardize_continuous result shared by field and seven state-axis _masked_summary consumers',
                candidate='widen exact normalized FP32 state once to F64; every existing summary casts F32',
                provenance=provenance + condition, prospective_overlay_AST_compiles=True,
                actual_inputs=metadata, actual_input_bridge=bridge, summary_shapes=[[257, 4]] + [[140, 4]] * 7,
                mask_shapes=[[257]] + [[140]] * 7, raw_state_shape=[3854], coordinates=list(COORDINATES),
                cotangent_origin='artificial sin(arange+1+consumer_index*0.125), computed float64 and stored float32',
                JAX_imports=0, component_evaluations=0, reference_adoption=False,
                evidence_limit='source shared boundary; saved IR is representative, not coordinate cause or actual new accumulation proof')

def load_component_inputs(root, prepared_dir):
    """Load the consolidated NumPy bundle; caller controls approved execution."""
    import numpy as np
    path = Path(prepared_dir) / 'component-inputs.pkl'
    raw_bytes = path.read_bytes()
    bridge = json.loads((Path(prepared_dir) / 'component-input-bridge.json').read_text())
    pin = bridge['component_bundle']
    if len(raw_bytes) != pin['bytes'] or hashlib.sha256(raw_bytes).hexdigest() != pin['sha256']:
        raise ValueError('prepared small component bundle differs from the reviewed bridge')
    bundle = pickle.loads(raw_bytes)
    for name in ('raw', 'mean', 'scale', 'tail_mask', 'defined'):
        if np.shape(bundle[name]) != (3854,): raise ValueError(name + ' shape drift')
    if bundle['raw'].dtype != np.float64 or bundle['defined'].dtype != np.bool_:
        raise ValueError('raw/defined dtype drift')
    if bundle['field_pool'].shape != (257, 3854) or bundle['axis_pool'].shape != (7, 140, 3854):
        raise ValueError('source pools differ')
    if len(bundle['cotangents']) != 8 or any(np.asarray(x).dtype != np.float32 or np.shape(x) != shape for x, shape in zip(bundle['cotangents'], [(257,4)] + [(140,4)] * 7, strict=True)):
        raise ValueError('artificial cotangent ABI differs')
    return bundle

def make_component(field_pool, axis_pool, threshold, *, candidate=False):
    """Construct native AD only during a separately approved runtime window."""
    import jax
    import jax.numpy as jnp
    from research.daily_coarse_graining.daily_operator.input_conditioning import standardize_continuous
    from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import _masked_summary
    pools = (field_pool,) + tuple(axis_pool[index] for index in range(7))

    def component(raw, mean, scale, tail_mask, defined, cotangents):
        def scalar(state_raw):
            state = standardize_continuous(state_raw, mean, scale, threshold, coordinate_mask=tail_mask)
            if candidate:
                with jax.named_scope('normalized_state_fanout_fp64'):
                    state = jnp.asarray(state, dtype=jnp.float64)
            summaries, masks = zip(*(_masked_summary(pool, state, defined) for pool in pools), strict=True)
            # Both arms preserve this declared F32 scalar and summary reduction.
            loss = sum((jnp.sum(summary * cotangent, dtype=jnp.float32) for summary, cotangent in zip(summaries, cotangents, strict=True)), jnp.asarray(0, dtype=jnp.float32))
            return loss, (tuple(summaries), tuple(masks))
        (loss, (summaries, masks)), state_vjp = jax.value_and_grad(scalar, has_aux=True)(raw)
        directions = jax.nn.one_hot(jnp.asarray(COORDINATES), raw.shape[0], dtype=raw.dtype)
        state_jvps = jnp.stack(tuple(jax.jvp(lambda x: scalar(x)[0], (raw,), (direction,))[1] for direction in directions))
        return loss, summaries, masks, state_vjp, state_jvps
    return component
