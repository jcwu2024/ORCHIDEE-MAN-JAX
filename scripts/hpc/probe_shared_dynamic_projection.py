"""Bounded FP32 union-bank projection diagnostic, not a model/science gradient.

Source: shared_transformer::_decoder_dynamic_plan/_decoder_dynamic_tokens and
_field_dynamic_projection_cache; shared_decoder_expert_execution union helpers.
Actual-token capture runs the original source-reuse context assembler only.
The component returns every field's dynamic K/V banks, with a seeded synthetic
cotangent. No attention, physics, optimizer, low precision or recurrent graph.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('ORCHIDEE_REPO_ROOT', Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@contextmanager
def deadline(seconds, label, emit):
    def expired():
        emit('timeout', status='timeout', timeout_scope=label, limit_seconds=seconds)
        os._exit(124)
    timer = threading.Timer(seconds, expired)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def layout_counts(plan, *, width, blocks):
    axis = int(plan.axis_expert_index.size)
    local = int(plan.local_state_indices.size)
    field_axis = sum(value.size for value in plan.field_axis_bank_indices)
    field_local = sum(value.size for value in plan.field_local_bank_indices)
    return {
        'fields': len(plan.field_axis_bank_indices), 'axis_union_rows': axis,
        'local_union_rows': local, 'local_sentinel_rows': int(np.sum(plan.local_state_indices == -1)),
        'field_axis_rows_with_repetition': field_axis,
        'field_local_rows_with_repetition': field_local,
        'union_dynamic_tokens_bytes': (axis + local) * width * 4,
        'union_dynamic_kv_bytes': (axis + local) * width * 4 * 2 * blocks,
        'all_returned_field_dynamic_kv_bytes': (field_axis + field_local) * width * 4 * 2 * blocks,
        'source_projection_dots_field': 4 * blocks * len(plan.field_axis_bank_indices),
        'source_projection_dots_union': 4 * blocks,
        'note': 'Nominal bank/interface bytes, not measured live GPU memory or FLOPs.',
    }


def projection_component(parameters, tokens, axis_indices, local_indices, *, union):
    """Same original K/V projection and field bank order in both executions."""
    import jax.numpy as jnp
    from research.daily_coarse_graining.daily_operator.shared_transformer import _project_key_value
    def project(values):
        axis, local = values
        return tuple((*_project_key_value(block, axis), *_project_key_value(block, local))
                     for block in parameters)
    if not union:
        return tuple(project((tokens[0][jnp.asarray(a)], tokens[1][jnp.asarray(l)]))
                     for a, l in zip(axis_indices, local_indices, strict=True))
    banks = project(tokens)
    return tuple(tuple((ak[jnp.asarray(a)], av[jnp.asarray(a)], lk[jnp.asarray(l)], lv[jnp.asarray(l)])
                       for ak, av, lk, lv in banks)
                 for a, l in zip(axis_indices, local_indices, strict=True))


def forward_vjp(parameters, tokens, cotangent, axis_indices, local_indices, *, union):
    import jax
    forward = lambda p, x: projection_component(p, x, axis_indices, local_indices, union=union)
    outputs, pullback = jax.vjp(forward, parameters, tokens)
    return outputs, pullback(cotangent)


def scalar_forward_grad(parameters, tokens, packed_cotangent, axis_indices, local_indices,
                        shapes, structure, *, union):
    """Consume the same cotangent on device, avoiding a many-leaf host boundary."""
    import jax
    import jax.numpy as jnp
    def objective(p, x):
        outputs = projection_component(p, x, axis_indices, local_indices, union=union)
        values, _ = jax.tree_util.tree_flatten(outputs)
        offset, terms = 0, []
        for value, shape in zip(values, shapes, strict=True):
            size = int(np.prod(shape))
            terms.append(jnp.vdot(value, packed_cotangent[offset:offset + size].reshape(shape)))
            offset += size
        return jnp.sum(jnp.stack(terms))
    return jax.value_and_grad(objective, argnums=(0, 1))(parameters, tokens)


def load_helper(path):
    name = 'scripts.hpc.probe_shared_expert_real_daily'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(args):
    # Set cache before helper imports, physical-package initialization or JAX compilation.
    os.environ['JAX_COMPILATION_CACHE_DIR'] = str(args.stage_cache)
    os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'
    os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '0'
    import jax
    jax.config.update('jax_compilation_cache_dir', str(args.stage_cache))
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
    helper = load_helper(args.real_daily_probe_file)
    if args.output.exists() or (args.mode == 'capture' and args.payload.exists()):
        raise FileExistsError('refusing to overwrite diagnostic output')
    report = {'status': 'running', 'mode': args.mode, 'training_updates': 0,
              'scientific_acceptance': False, 'production_adoption': False,
              'scope': 'FP32_fixed_row_dynamic_KV_component_synthetic_cotangent_not_science_loss',
              'exact_rtol': 1e-5, 'exact_atol': 1e-6,
              'timing_order': 'field_then_union_fixed_order_not_randomized',
              'memory_timing_scope': 'all_field_KV_banks_returned_not_integrated_decoder',
              'timing_boundary': args.timing_boundary,
              'jax_version': jax.__version__, 'devices': [str(value) for value in jax.devices()]}
    started = time.perf_counter()
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        helper.atomic_json(args.output, report)
        print(stage, flush=True)
    with deadline(args.process_seconds, 'whole_process', emit):
        # Initialize the original package before native AD/tracing.
        from research.daily_coarse_graining.daily_operator import shared_transformer as st
        if args.mode in {'inspect', 'capture'}:
            _, reference = helper.trusted_profile(args)
            if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != reference['source_git_head']:
                raise ValueError('declared immutable source checkout differs')
            identity = helper.context_identity(reference, helper.initial_checkpoint(reference), 742)
            with args.prepared_context.open('rb') as stream:
                context = pickle.load(stream)
            if context['identity'] != identity:
                raise ValueError('prepared context identity differs')
            head = context['head']
            plan = st._decoder_dynamic_plan(schema=head.learned_schema, layout=head.layout)
            counts = layout_counts(plan, width=head.shared_config.resolved_decoder_width,
                                   blocks=head.shared_config.coordinate_blocks)
            emit('layout_inspected', identity=identity, layout_counts=counts)
            if counts['union_dynamic_kv_bytes'] > 128 * 2**20:
                raise ValueError('nominal union K/V banks exceed bounded 128MiB admission')
            if args.mode == 'inspect':
                emit('completed', status='inspected')
                return report
            if jax.default_backend() != 'gpu':
                raise ValueError('actual-token capture requires GPU')
            from scripts.hpc.run_shared_learning import load_learning_checkpoint
            checkpoint = load_learning_checkpoint(Path(identity['initial_checkpoint']), expected_method=reference['method'])
            parameters = jax.device_put(checkpoint['parameters'])
            sample = jax.tree_util.tree_map(lambda x: x[0], context['daily4'])
            adapter_assembler = helper.load_backbone_assembler(args.backbone_module_file)
            emit('capture_start')
            began = time.perf_counter()
            # This eager producer stops before the decoder, physics and loss.
            with deadline(180, 'actual_token_producer', emit), jax.disable_jit():
                inputs = head.input_assembler.assemble(sample.inputs)
                neural_context = adapter_assembler(parameters, inputs, normalization=head.input_normalization,
                    layout=head.layout, config=head.shared_config)
                family, family_mask = st._source_family_bank(neural_context, head.layout)
                decoder_context, _ = st._project_decoder_context(parameters, neural_context, family, family_mask)
                tokens = st._decoder_dynamic_tokens(parameters, decoder_context, inputs,
                    input_normalization=head.input_normalization, layout=head.layout, plan=plan)
                tokens = jax.device_get(jax.block_until_ready(tokens))
            if not all(np.all(np.isfinite(value)) for value in tokens):
                raise ValueError('actual dynamic tokens nonfinite')
            saved = {'schema_version': 'shared_dynamic_projection_payload_v1', 'identity': identity,
                     'row': 0, 'tokens': tokens, 'layout_counts': counts,
                     'parameters': jax.device_get(tuple({name: block['attention'][name] for name in ('key', 'value')}
                         for block in parameters['shared_transformer']['coordinate_blocks'])),
                     'axis_indices': plan.field_axis_bank_indices, 'local_indices': plan.field_local_bank_indices,
                     'raw_state_defined': np.asarray(inputs.state_defined.other),
                     'context_axis_mask': np.asarray(decoder_context.expert_axis_mask),
                     'context_pft_mask': np.asarray(decoder_context.pft_mask)}
            helper.atomic_pickle(args.payload, saved)
            emit('completed', status='captured', payload=str(args.payload), capture_seconds=time.perf_counter() - began)
            return report
        if jax.default_backend() != 'gpu':
            raise ValueError('component timing requires GPU')
        with args.payload.open('rb') as stream:
            saved = pickle.load(stream)
        if saved['schema_version'] != 'shared_dynamic_projection_payload_v1' or saved['row'] != 0:
            raise ValueError('unknown or wrong-row projection payload')
        parameters, tokens = jax.device_put((saved['parameters'], saved['tokens']))
        axis, local = saved['axis_indices'], saved['local_indices']
        output_shape = jax.eval_shape(lambda p, x: projection_component(p, x, axis, local, union=True), parameters, tokens)
        leaves, structure = jax.tree_util.tree_flatten(output_shape)
        cotangent = jax.tree_util.tree_unflatten(structure, [
            jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(20260927), i), value.shape, dtype=value.dtype)
            / max(1, value.size)**0.5 for i, value in enumerate(leaves)])
        if args.timing_boundary == 'scalar':
            import jax.numpy as jnp
            shapes = tuple(value.shape for value in leaves)
            cotangent = jnp.concatenate(tuple(value.reshape(-1) for value in jax.tree_util.tree_leaves(cotangent)))
            cotangent = jax.block_until_ready(cotangent)
            helper.atomic_pickle(args.output.with_suffix('.cotangent.pkl'), jax.device_get(cotangent))
            report['memory_timing_scope'] = 'scalar_synthetic_objective_and_parameter_token_gradients_not_integrated_decoder'
            old_result_path = args.output.with_name('shared-dynamic-projection-20260927.compare.field.result.pkl')
            if old_result_path.exists():
                with old_result_path.open('rb') as stream:
                    ready_result = jax.device_put(pickle.load(stream))
                jax.block_until_ready(ready_result)
                ready_cost = []
                for _ in range(20):
                    began = time.perf_counter()
                    jax.block_until_ready(ready_result)
                    ready_cost.append(time.perf_counter() - began)
                report['old_banks_boundary_already_ready_block'] = {
                    'leaves': len(jax.tree_util.tree_leaves(ready_result)),
                    'seconds': ready_cost, 'median_seconds': float(np.median(ready_cost)),
                    'note': 'Host blocking traversal on already-ready saved result; not full call marshaling/dispatch cost.'}
                del ready_result
        report['identity'] = saved['identity']
        report['layout_counts'] = saved['layout_counts']
        report['cotangent'] = 'seed20260927_fold_in_leaf_normal_div_sqrt_leaf_size_same_for_both'
        timings, results = {}, {}
        for name, union in (('field', False), ('union', True)):
            emit('component_compile_start', active_route=name)
            if args.timing_boundary == 'scalar':
                function = jax.jit(lambda p, x, c: scalar_forward_grad(
                    p, x, c, axis, local, shapes, structure, union=union))
            else:
                function = jax.jit(lambda p, x, c: forward_vjp(p, x, c, axis, local, union=union))
            began = time.perf_counter()
            with deadline(180, name + '_lower_compile', emit):
                lowered = function.lower(parameters, tokens, cotangent)
                lower_seconds = time.perf_counter() - began
                executable = lowered.compile()
            compile_seconds = time.perf_counter() - began - lower_seconds
            stable = lowered.compiler_ir('stablehlo')
            from collections import Counter
            def count(operation):
                counts = Counter([operation.name])
                for region in operation.regions:
                    for block in region.blocks:
                        for child in block.operations:
                            counts.update(count(child.operation))
                return counts
            operations = count(stable.operation)
            helper.atomic_pickle(args.output.with_suffix('.' + name + '.hlo.pkl'), stable.operation.get_asm())
            stats = executable.memory_analysis()
            memory = {key: int(getattr(stats, key)) for key in
                      ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
            memory['estimated_live_bytes'] = memory['argument_size_in_bytes'] + memory['output_size_in_bytes'] + memory['temp_size_in_bytes'] - memory['alias_size_in_bytes']
            if memory['estimated_live_bytes'] > 2 * 2**30:
                raise ValueError('component estimated memory exceeds 2GiB diagnostic bound')
            began = time.perf_counter()
            result = jax.block_until_ready(executable(parameters, tokens, cotangent))
            cold_execution_seconds = time.perf_counter() - began
            warm = []
            for _ in range(args.repeats):
                began = time.perf_counter()
                result = jax.block_until_ready(executable(parameters, tokens, cotangent))
                warm.append(time.perf_counter() - began)
            results[name] = jax.device_get(result)
            helper.atomic_pickle(args.output.with_suffix('.' + name + '.result.pkl'), results[name])
            timings[name] = {'lower_seconds': lower_seconds, 'compile_seconds': compile_seconds,
                             'cold_execution_seconds': cold_execution_seconds, 'warm_seconds': warm,
                             'warm_median_seconds': float(np.median(warm)), 'memory': memory,
                             'operations': sum(operations.values()), 'dot_general': operations['stablehlo.dot_general'],
                             'input_leaves': len(jax.tree_util.tree_leaves((parameters, tokens, cotangent))),
                             'result_leaves': len(jax.tree_util.tree_leaves(result))}
            emit('component_complete', active_route=name, timings=timings)
        comparisons = {'outputs': helper.compare_trees(jax, results['union'][0], results['field'][0]),
                       'projection_parameter_vjp': helper.compare_trees(jax, results['union'][1][0], results['field'][1][0]),
                       'token_vjp': helper.compare_trees(jax, results['union'][1][1], results['field'][1][1])}
        passed = all(value['passed'] for value in comparisons.values())
        emit('completed', status='passed' if passed else 'exact_comparison_failed', comparisons=comparisons,
             timing_ratio_field_over_union=timings['field']['warm_median_seconds'] / timings['union']['warm_median_seconds'])
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('inspect', 'capture', 'compare'), required=True)
    parser.add_argument('--prepared-context', type=Path, required=True)
    parser.add_argument('--source-profile', type=Path, required=True)
    parser.add_argument('--real-daily-probe-file', type=Path, required=True)
    parser.add_argument('--backbone-module-file', type=Path, required=True)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage-cache', type=Path, required=True)
    parser.add_argument('--process-seconds', type=int, default=600)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--timing-boundary', choices=('banks', 'scalar'), default='banks')
    run(parser.parse_args())
