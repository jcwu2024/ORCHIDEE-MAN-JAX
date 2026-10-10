"""First-day raw neural prefix; composition only, no new physical process.

Source boundary: successor_training.py::rollout_input_from_template (161-184),
model.py::daily_operator_transition (79-85), shared_process_head.py::
SharedTransformerProcessHead.assemble_context/learned_outputs (120-129).
The prefix calls these source owners and stops before transform_shared_output.
"""
from __future__ import annotations

import hashlib
import inspect
from typing import Any, NamedTuple
import numpy as np

try:
    from scripts.hpc import shared_latent_T7_day_jvp as day
except ImportError:
    import shared_latent_T7_day_jvp as day

first_day_inputs = day.first_day_inputs
BATCH_SIZE, SAMPLE_INDEX, RTOL, ATOL = day.BATCH_SIZE, day.SAMPLE_INDEX, day.RTOL, day.ATOL
BOUNDARY = 'canonical_input_rebuild__input_assembler__assemble_context__shared_transformer_raw_outputs'
RESULT_SCHEMA = 'RawNeuralJVPResult(primal=all_raw_fields,tangent=all_raw_fields,auxiliary=exact_active_masks)'
INFERENCE_LIMIT = ('A strict pass means the discrepancy is not reproduced at this boundary; '
                   'smaller raw differences may be amplified downstream. A failure localizes occurrence '
                   'inside this prefix but establishes no cause, reference or scientific adoption.')


class RawNeuralJVPResult(NamedTuple):
    primal: Any
    tangent: Any
    auxiliary: Any


def make_raw_prefix(head, *, rebuild=None, raw_outputs=None):
    """Use the same patched raw callable and kwargs as head.learned_outputs.

    The retained part of the selected daily argument remains an input ABI
    member; this prefix has no consumer for it. No next state is fabricated.
    Optional callables exist only for tiny analytic composition fixtures.
    """
    if rebuild is None:
        from research.daily_coarse_graining.daily_operator.successor_training import rollout_input_from_template
        rebuild = rollout_input_from_template
    if raw_outputs is None:
        from research.daily_coarse_graining.daily_operator import shared_process_head
        raw_outputs = shared_process_head.shared_transformer_raw_outputs
    shapes = {field.name: tuple(field.shape) for field in head.learned_schema.fields}
    expected = set(shapes)

    def prefix(parameters, state, discrete, first_day):
        template, _retained_inputs = first_day
        current = rebuild(template, state, discrete, head.contract)
        assembled = head.input_assembler.assemble(current)
        context = head.assemble_context(parameters, assembled)
        raw, active = raw_outputs(parameters, context, assembled,
            schema=head.learned_schema, layout=head.layout,
            input_normalization=head.input_normalization, config=head.shared_config,
            carbon_state_layout=head.carbon_state_layout)
        if set(raw) != expected or set(active) != expected:
            raise ValueError('all declared raw learned fields and masks are required')
        for name in expected:
            if np.dtype(raw[name].dtype).kind != 'f' or np.dtype(active[name].dtype).kind != 'b':
                raise ValueError('raw fields must be floating and active masks boolean')
            if raw[name].shape != active[name].shape:
                raise ValueError('raw field and active mask shapes differ: '+name)
            if tuple(raw[name].shape) != shapes[name]:
                raise ValueError('full declared raw field shape required: '+name)
        return raw, active
    return prefix


def source_contract(head, raw_function, backbone_provider):
    """Record actual Python owner bindings without tracing/evaluating a model."""
    from research.daily_coarse_graining.daily_operator.successor_training import rollout_input_from_template
    bindings = {}
    for name, function in {'input_rebuild': rollout_input_from_template,
                            'input_assembler': type(head.input_assembler).assemble,
                            'context_builder': head.assemble_context,
                            'backbone_context_provider': backbone_provider,
                            'raw_output_provider': raw_function,
                            'learned_outputs_reference_only': head.learned_outputs}.items():
        source, start = inspect.getsourcelines(function)
        text = ''.join(source)
        bindings[name] = {'qualified_name': function.__qualname__,
                          'source_file': inspect.getsourcefile(function),
                          'start_line': start, 'end_line': start+len(source)-1,
                          'source_sha256': hashlib.sha256(text.encode()).hexdigest()}
    return {'boundary': BOUNDARY, 'result_schema': RESULT_SCHEMA,
            'source_bindings': bindings,
            'learned_fields': [{'name': field.name, 'declared_shape': list(field.shape)}
                               for field in head.learned_schema.fields],
            'active_masks': 'exact provider masks, one mask per raw learned field',
            'outputs_are_pre_physical_transform': True, 'next_state_output': False,
            'inference_limit': INFERENCE_LIMIT}


def route_functions(jax, prefix):
    def b1(parameters, state, discrete, first_day, direction):
        day._require_state_direction(state, direction, batched=False)
        primal, tangent, active = jax.jvp(
            lambda value: prefix(parameters, value, discrete, first_day),
            (state,), (direction,), has_aux=True)
        return RawNeuralJVPResult(primal, tangent, active)

    def b4(parameters, states, discretes, first_days, directions):
        day._require_state_direction(states, directions, batched=True)
        primal, tangent, active = jax.jvp(
            lambda values: jax.vmap(prefix, in_axes=(None,0,0,0))(parameters, values, discretes, first_days),
            (states,), (directions,), has_aux=True)
        return RawNeuralJVPResult(primal, tangent, active)
    return b1, b4


def _float_comparison(jax, actual, reference, label, *, rtol, atol):
    paths_a, tree_a = jax.tree_util.tree_flatten_with_path(actual)
    paths_b, tree_b = jax.tree_util.tree_flatten_with_path(reference)
    if tree_a != tree_b or [p for p,_ in paths_a] != [p for p,_ in paths_b]:
        raise ValueError('raw prefix pytree structure differs')
    rows = []
    for (path,a),(_,b) in zip(paths_a,paths_b,strict=True):
        a,b = np.asarray(a),np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype or a.dtype.kind != 'f':
            raise ValueError('raw prefix floating shape/dtype differs')
        finite = np.isfinite(a)&np.isfinite(b)
        difference = np.zeros(a.shape,dtype=np.float64)
        np.subtract(a.astype(np.float64),b.astype(np.float64),out=difference,where=finite)
        difference = np.abs(difference)
        allowance = atol+rtol*np.abs(b.astype(np.float64))
        failed = ~finite|(difference>allowance)
        ratio = np.zeros(a.shape,dtype=np.float64)
        np.divide(difference,allowance,out=ratio,where=finite&(allowance>0))
        ratio[~finite] = np.inf
        worst = int(np.argmax(ratio)) if ratio.size else 0
        position = [int(x) for x in np.unravel_index(worst,a.shape)] if a.shape and a.size else []
        index = tuple(position)
        def value(array):
            number = float(array[index]) if array.size else None
            return number if number is not None and np.isfinite(number) else None
        rows.append({'path':label+jax.tree_util.keystr(path),'shape':list(a.shape),'dtype':str(a.dtype),
            'passed':bool(not np.any(failed)), 'exact_match':bool(np.array_equal(a,b,equal_nan=True)),
            'failed_coordinates':int(np.count_nonzero(failed)),
            'nonfinite_coordinates':int(np.count_nonzero(~finite)),
            'max_absolute_error':float(np.max(difference,initial=0)),
            'max_error_to_allowance':float(np.max(ratio,initial=0)) if np.all(finite) else None,
            'worst_position':position,'worst_actual_value':value(a),'worst_reference_value':value(b),
            'worst_absolute_error':value(difference),'worst_allowance':value(allowance)})
    return rows


def compare_results(jax, actual, reference, *, sample_index=None, batch_size=BATCH_SIZE, rtol=RTOL, atol=ATOL):
    if not isinstance(actual,RawNeuralJVPResult) or not isinstance(reference,RawNeuralJVPResult):
        raise TypeError('truthful RawNeuralJVPResult required')
    if sample_index is not None:
        reference = day._select_sample(jax,reference,sample_index,batch_size)
    rows = (_float_comparison(jax,actual.primal,reference.primal,'primal',rtol=rtol,atol=atol)
            +_float_comparison(jax,actual.tangent,reference.tangent,'tangent',rtol=rtol,atol=atol))
    masks = day._auxiliary_comparison(jax,actual.auxiliary,reference.auxiliary)
    if any(row['dtype'] != 'bool' for row in masks):
        raise ValueError('raw applicability auxiliaries must be exact boolean masks')
    return {'passed':all(row['passed'] for row in rows) and all(row['exact_match'] for row in masks),
            'exact_match':all(row['exact_match'] for row in rows+masks), 'rtol':rtol,'atol':atol,
            'float_leaf_count':len(rows),'float_leaves':rows,'auxiliary_leaves':masks,
            'failed_float_coordinates':sum(row['failed_coordinates'] for row in rows),
            'nonfinite_float_coordinates':sum(row['nonfinite_coordinates'] for row in rows),
            'auxiliary_mismatches':sum(not row['exact_match'] for row in masks),
            'scope':'first_day_raw_neural_prefix_only','result_schema':RESULT_SCHEMA,
            'inference_limit':INFERENCE_LIMIT}


def other_sample_tangent_isolation(jax, result, *, sample_index=SAMPLE_INDEX, batch_size=BATCH_SIZE):
    """Measure full raw-field tangents for the three zero-direction cases.

    This operates on already preserved host arrays and adds no model calls.
    Exact zero is expected for an independent vmap with zero input directions.
    """
    if not isinstance(result, RawNeuralJVPResult):
        raise TypeError('truthful RawNeuralJVPResult required')
    if type(sample_index) is not int or not 0 <= sample_index < batch_size:
        raise ValueError('valid isolated sample index required')
    rows=[]
    for path,value in jax.tree_util.tree_flatten_with_path(result.tangent)[0]:
        value=np.asarray(value)
        if not value.shape or value.shape[0] != batch_size or value.dtype.kind != 'f':
            raise ValueError('full B4 floating tangent batch axis required')
        others=np.delete(value,sample_index,axis=0)
        finite=np.isfinite(others)
        nonzero=finite&(others!=0)
        rows.append({'path':'tangent'+jax.tree_util.keystr(path),'shape':list(value.shape),
            'dtype':str(value.dtype),'other_samples':[i for i in range(batch_size) if i != sample_index],
            'checked_coordinates':int(others.size),'zero_coordinates':int(np.count_nonzero(finite&(others==0))),
            'nonzero_coordinates':int(np.count_nonzero(nonzero)),
            'nonfinite_coordinates':int(np.count_nonzero(~finite)),
            'max_absolute_finite_value':float(np.max(np.abs(others[finite]),initial=0)),
            'passed':bool(np.all(finite)&np.all(others==0))})
    return {'passed':all(row['passed'] for row in rows),'sample_index':sample_index,
            'expected_other_sample_tangent':0.0,'gate':'exact_zero_all_raw_fields_no_tolerance',
            'nonzero_coordinates':sum(row['nonzero_coordinates'] for row in rows),
            'nonfinite_coordinates':sum(row['nonfinite_coordinates'] for row in rows),'leaves':rows}
