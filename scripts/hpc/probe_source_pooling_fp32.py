"""Synthetic FP64 versus FP32 source-family pooling probe; no model training.

The fixed fixture exercises ``_source_family_bank`` directly.  It is not a
sample from the Teacher/training distribution and cannot establish a full-model
speedup or scientific equivalence.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FAMILY_COUNT = 20
TOKEN_COUNT = 393
WIDTH = 384
EXPERT_COUNT = 7
SEED = 20260915


def _fixture(jax, jnp):
    """Return a declared source-shaped fixture with meaningful empty masks."""
    key = jax.random.PRNGKey(SEED)
    key, token_key, weight_key = jax.random.split(key, 3)
    # Family 19 is intentionally absent; several source keys are invalid.
    family = np.arange(TOKEN_COUNT, dtype=np.int32) % (FAMILY_COUNT - 1)
    source_mask = np.ones(TOKEN_COUNT, dtype=np.bool_)
    source_mask[::17] = False
    source_mask[5::41] = False
    allowed = np.ones((EXPERT_COUNT, FAMILY_COUNT), dtype=np.bool_)
    disabled = ((0, 0), (1, 4), (2, 8), (3, 12), (4, 16), (5, 18), (6, 3))
    for expert, source_family in disabled:
        allowed[expert, source_family] = False
    layout = SimpleNamespace(
        source_family_names=tuple(f"synthetic_family_{index:02d}" for index in range(FAMILY_COUNT)),
        source_allowed=allowed,
    )
    context = SimpleNamespace(
        source_family_index=family,
        source_mask=source_mask,
        source_tokens=None,
    )
    tokens = jax.random.normal(token_key, (TOKEN_COUNT, WIDTH), dtype=jnp.float32)
    weights = jax.random.normal(weight_key, (EXPERT_COUNT, FAMILY_COUNT, WIDTH), dtype=jnp.float32)
    return context, layout, tokens, weights, disabled


def _summary(samples):
    values = [float(value) for value in samples]
    return dict(samples_seconds=values, median_seconds=float(np.median(values)),
                minimum_seconds=min(values), maximum_seconds=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('explore_gpu', 'gpu', 'cpu'), default='explore_gpu')
    parser.add_argument('--repeats', type=int, default=51)
    parser.add_argument('--warmup-seconds', type=float, default=.5)
    # Accepted solely so the shared GPU launcher can supply its standard paths.
    # This primitive reads no Teacher payload or runtime artifact.
    for name in ('dataset-manifest', 'acceptance-report', 'base-statistics',
                 'process-statistics', 'fast-statistics', 'prepared-manifest'):
        parser.add_argument('--' + name)
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.warmup_seconds < 0:
        raise ValueError('repeats must be positive and warmup-seconds nonnegative')

    from scripts.hpc.check_daily_training_runtime import _atomic_json

    started = time.perf_counter()
    report = dict(
        status='running', scope='synthetic_source_family_pooling_primitive_not_actual_distribution_or_full_model',
        fixture=dict(family_count=FAMILY_COUNT, token_count=TOKEN_COUNT, width=WIDTH,
                     expert_count=EXPERT_COUNT, empty_family=19,
                     invalid_source_key_stride=(17, 41),
                     disabled_expert_families=[list(value) for value in ((0, 0), (1, 4), (2, 8),
                                                                          (3, 12), (4, 16), (5, 18), (6, 3))]),
        seed=SEED, repeats=arguments.repeats, parameter_updates=0, teacher_data_read=False,
        attention_precision_unchanged=True,
    )

    def emit(stage):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started)
        _atomic_json(arguments.output, report)
        print(json.dumps(dict(stage=stage, seconds=report['elapsed_seconds'])), flush=True)

    try:
        emit('initialize')
        if arguments.backend == 'explore_gpu':
            from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
            jax, devices = initialize_gpu_backend()
        else:
            if arguments.backend == 'cpu':
                os.environ.setdefault('JAX_PLATFORMS', 'cpu')
            import jax
            devices = jax.devices(arguments.backend)
        jax.config.update('jax_enable_x64', True)
        jax.config.update('jax_default_matmul_precision', 'highest')
        import jax.numpy as jnp

        from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import NeuralPrecisionPolicy
        from research.daily_coarse_graining.daily_operator.target_oriented_attention import _source_family_bank
        from scripts.hpc.probe_public_rollout import gradient_comparison

        context, layout, tokens, weights, disabled = _fixture(jax, jnp)
        source_mask = jnp.asarray(context.source_mask)
        phase_arguments = {'forward': (tokens, source_mask),
                           'value_and_grad': (tokens, source_mask, weights)}
        report.update(jax_version=jax.__version__, backend=devices[0].platform, device=str(devices[0]),
                      device_kind=devices[0].device_kind)

        executables, results = {}, {}
        for mode, policy in (
            ('legacy_fp64_pooling', NeuralPrecisionPolicy()),
            ('fp32_pooling', NeuralPrecisionPolicy(source_family_pooling_fp32=True)),
        ):
            def forward(source_tokens, active_source_mask, policy=policy):
                active_context = SimpleNamespace(
                    source_family_index=context.source_family_index,
                    source_mask=active_source_mask,
                    source_tokens=source_tokens,
                )
                banks, valid = zip(*(
                    _source_family_bank(active_context, layout, expert, neural_precision=policy)
                    for expert in range(EXPERT_COUNT)
                ), strict=True)
                return jnp.stack(banks), jnp.stack(valid)

            def scalar(source_tokens, active_source_mask, cotangent_weights):
                bank, _ = forward(source_tokens, active_source_mask)
                return jnp.mean(bank * cotangent_weights)

            executables[mode] = {}
            results[mode] = {}
            for phase, function in (
                ('forward', jax.jit(forward)),
                ('value_and_grad', jax.jit(jax.value_and_grad(scalar))),
            ):
                emit('compile_' + mode + '_' + phase)
                began = time.perf_counter()
                compiled = function.lower(*phase_arguments[phase]).compile()
                results[mode][phase] = dict(compile_seconds=time.perf_counter() - began,
                                            samples_seconds=[])
                executables[mode][phase] = compiled
                results[mode][phase]['first_result'] = jax.device_get(
                    jax.block_until_ready(compiled(*phase_arguments[phase]))
                )

        old_banks, old_valid = results['legacy_fp64_pooling']['forward']['first_result']
        new_banks, new_valid = results['fp32_pooling']['forward']['first_result']
        old_loss, old_gradient = results['legacy_fp64_pooling']['value_and_grad']['first_result']
        new_loss, new_gradient = results['fp32_pooling']['value_and_grad']['first_result']
        output_error = np.asarray(new_banks) - np.asarray(old_banks)
        comparisons = dict(
            output=dict(maximum_absolute_error=float(np.max(np.abs(output_error))),
                        relative_l2=float(np.linalg.norm(output_error) / max(np.linalg.norm(old_banks), 1e-12)),
                        passed=bool(np.allclose(new_banks, old_banks, rtol=1e-5, atol=1e-6))),
            masks_equal=bool(np.array_equal(new_valid, old_valid)),
            loss=dict(old=float(old_loss), new=float(new_loss),
                      passed=bool(np.isclose(new_loss, old_loss, rtol=1e-5, atol=1e-6))),
            input_gradient=gradient_comparison(jax, new_gradient, old_gradient),
            finite=dict(output=bool(np.all(np.isfinite(new_banks))),
                        input_gradient=bool(np.all(np.isfinite(new_gradient)))),
            mask_checks=dict(empty_family_invalid=bool(not np.any(new_valid[:, 19])),
                             masked_input_gradient_zero=bool(np.all(new_gradient[~context.source_mask] == 0)),
                             invalid_output_zero=bool(np.all(new_banks[~new_valid] == 0)),
                             disabled_expert_families_invalid=bool(all(
                                 not new_valid[expert, source_family]
                                 for expert, source_family in disabled
                             ))),
        )
        report['comparisons'] = comparisons
        emit('comparison_completed')

        warmup_until = time.perf_counter() + arguments.warmup_seconds
        while time.perf_counter() < warmup_until:
            for mode in executables.values():
                for phase, executable in mode.items():
                    jax.block_until_ready(executable(*phase_arguments[phase]))
        for repeat in range(-2, arguments.repeats):
            modes = tuple(executables) if repeat % 2 else tuple(reversed(executables))
            for mode in modes:
                for phase, executable in executables[mode].items():
                    began = time.perf_counter()
                    result = jax.block_until_ready(executable(*phase_arguments[phase]))
                    if repeat >= 0:
                        results[mode][phase]['samples_seconds'].append(time.perf_counter() - began)
                    del result
        for mode in results:
            for phase in results[mode]:
                if phase == 'forward' or phase == 'value_and_grad':
                    results[mode][phase].pop('first_result')
                    results[mode][phase].update(_summary(results[mode][phase]['samples_seconds']))
        report['modes'] = results
        report['time_reduction'] = {
            phase: 1. - results['fp32_pooling'][phase]['median_seconds'] /
            results['legacy_fp64_pooling'][phase]['median_seconds']
            for phase in ('forward', 'value_and_grad')
        }
        report['primitive_numerical_passed'] = bool(
            comparisons['output']['passed'] and comparisons['masks_equal'] and comparisons['loss']['passed']
            and comparisons['input_gradient']['passed'] and all(comparisons['finite'].values())
            and all(comparisons['mask_checks'].values())
        )
        report.update(status='completed', gpu_timing=arguments.backend != 'cpu', full_model_admission=False)
        emit('completed')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        emit('failed')
        raise


if __name__ == '__main__':
    main()
