"""Attribute the fixed trained forward panel without parameter updates."""
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.hpc import run_fresh_start_one_day as training
from scripts.hpc.trained_screen_common import WindowPool, json_safe, load_model


def run(args):
    import json

    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    started = time.perf_counter()
    report = {'status': 'running', 'updates': 0, 'windows': [],
              'source_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'scope': 'fixed_train_domain_panel_error_attribution_not_acceptance'}

    def emit(stage, **values):
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter()-started)
        _atomic_json(args.output, json_safe(report))
        print(f'forward_attribution stage={stage} windows={len(report["windows"])} seconds={report["elapsed_seconds"]:.1f}', flush=True)

    try:
        emit('bootstrap')
        runtime, head, template, scales, parameters, method = load_model(args)
        jax = runtime['jax']
        import jax.numpy as jnp

        from research.daily_coarse_graining.daily_operator.canonical_rollout import (
            make_canonical_step,
            make_science_probe_loss,
        )
        from research.daily_coarse_graining.daily_operator.forward_attribution import (
            inventory_pools,
            split_pool_changes,
            state_pools,
        )
        from research.daily_coarse_graining.daily_operator.model import bind_daily_operator_dynamic_retained_inputs
        from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
        from research.daily_coarse_graining.daily_operator.successor_training import (
            extract_biomass_productivity_trajectories,
            rollout_input_from_template,
        )
        from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory
        from scripts.hpc.probe_public_rollout import physical_checks

        pool = WindowPool(args.selection, method['source_identity'])
        previous = json.loads(args.reference_report.read_text())
        reference = {(r['selection']['artifact_id'], r['selection']['length']): r for r in previous['windows']}
        if previous['checkpoint'] != str(args.checkpoint) or len(reference) != 16:
            raise ValueError('reference must be the complete final-checkpoint panel')
        component = make_rollout_loss(jax, jnp, make_canonical_step(head, template), make_science_probe_loss(head.contract, scales))
        forward = jax.jit(component.rollout)
        transition = bind_daily_operator_dynamic_retained_inputs(input_assembler=head.input_assembler,
            process_head=head, retained_tail_template=template)

        def diagnostic_day(params, state, discrete, day):
            current = rollout_input_from_template(day[0], state, discrete, head.contract)
            result = transition(params, current, day[1])
            assembled = head.input_assembler.assemble(current)
            gpp = result.prediction.named_outputs['daily.fast_gpp']
            return {'state': result.next_state.other, 'defined': result.next_state_defined.other,
                'gpp': gpp, 'gpp_defined': jnp.broadcast_to(current.pft.active_pft_mask[None, :], gpp.shape),
                'fraction': current.pft.pft_fraction, 'active': current.pft.active_pft_mask,
                'carbon_start': assembled.state.carbon, 'carbon_fast': result.fast_update.carbon.inventory,
                'carbon_input': result.fast_update.carbon.external_input,
                'carbon_output': result.fast_update.carbon.external_output}

        diagnostic = jax.jit(diagnostic_day)

        def extract(obs):
            return extract_biomass_productivity_trajectories(obs['state'], obs['defined'],
                daily_fast_gpp=obs['gpp'], daily_fast_gpp_defined=obs['gpp_defined'],
                pft_fraction=obs['fraction'], active_pft_mask=obs['active'], contract=head.contract)

        extract = jax.jit(extract)
        executables, daily = {}, None
        report['checkpoint'] = str(args.checkpoint)
        for row in pool.selection['windows']:
            h = row['length']
            initial, discrete, days, targets, expected_discrete = pool.case(jax, row)
            day0 = jax.tree_util.tree_map(lambda x: x[0], days)
            if args.shape_only:
                if h not in executables:
                    jax.eval_shape(diagnostic, parameters, initial, discrete, day0)
                    teacher = jax.device_get(extract(targets))
                    pools = state_pools(targets['state'], targets['fraction'], targets['active'], head.contract)
                    err = sum(pools.values()) - (np.asarray(teacher.stocks['ecosystem_carbon'].value)-np.asarray(teacher.stocks['total_live_biomass_carbon'].value)).reshape(-1)
                    if not np.allclose(err, 0., atol=1e-8):
                        raise ValueError('independent pool extraction differs from common extractor')
                    executables[h] = True
                continue
            if h not in executables:
                emit('compile_forward', horizon=h)
                executable = forward.lower(parameters, initial, discrete, days).compile()
                memory = _compiled_memory(executable)
                if memory['estimated_live_bytes'] > 18*1024**3:
                    emit('stopped', status='resource_limited', compiled_memory=memory)
                    return report
                executables[h] = executable
            _, obs = executables[h](parameters, initial, discrete, days)
            jax.block_until_ready(obs)
            hard = physical_checks(obs, targets, expected_discrete, days[0], head.contract, scales)
            if daily is None:
                emit('compile_daily_diagnostic')
                daily = diagnostic.lower(parameters, initial, discrete, day0).compile()
                memory = _compiled_memory(daily)
                if memory['estimated_live_bytes'] > 18*1024**3:
                    emit('stopped', status='resource_limited', compiled_memory=memory)
                    return report
                report['daily_compiled_memory'] = memory
            free_rows, reset_rows = [], []
            for i in range(h):
                day = jax.tree_util.tree_map(lambda x: x[i], days)
                state = initial if i == 0 else obs['state'][i-1]
                disc = discrete if i == 0 else jax.tree_util.tree_map(lambda x: x[i-1], obs['discrete'])
                free_rows.append(jax.device_get(daily(parameters, state, disc, day)))
                reset_rows.append(jax.device_get(daily(parameters, day[0].continuous_state, day[0].discrete_state, day)))
            free = jax.tree_util.tree_map(lambda *x: np.stack(x), *free_rows)
            reset = jax.tree_util.tree_map(lambda *x: np.stack(x), *reset_rows)
            obs_host = jax.device_get(obs)
            science = {key: jax.device_get(extract(value)) for key, value in [('free', obs), ('reset', reset), ('teacher', targets)]}
            pools = {key: state_pools(value['state'], value['fraction'], value['active'], head.contract)
                     for key, value in [('free', obs_host), ('reset', reset), ('teacher', targets)]}
            decomposition = {}
            for label, value in [('free', free), ('reset', reset)]:
                parts = [inventory_pools(value['carbon_'+key], value['fraction'], value['active'], head.contract)
                         for key in ('start', 'fast', 'input', 'output')]
                final = state_pools(value['state'], value['fraction'], value['active'], head.contract)
                decomposition[label] = split_pool_changes(parts[0], parts[1], final, parts[2], parts[3])
            shared_pool_errors = {}
            for label in pools:
                s = science[label].stocks
                shared = (np.asarray(s['ecosystem_carbon'].value)-np.asarray(s['total_live_biomass_carbon'].value)).reshape(-1)
                shared_pool_errors[label] = float(np.max(abs(sum(pools[label].values())-shared)))
            prior = reference[(row['artifact_id'], h)]
            if prior['selection'] != row:
                raise ValueError('reference window differs from fixed selection')
            current_gpp = np.asarray(science['free'].fluxes['gpp'].value)
            old_gpp = np.asarray(prior['physical']['fluxes']['gpp']['predicted_daily'])
            valid = np.asarray(obs_host['defined'])
            delta = np.where(valid, free['state']-obs_host['state'], 0.)
            replay = {'state_max_abs_error': float(np.max(abs(delta))),
                      'state_allclose': bool(np.allclose(free['state'][valid], obs_host['state'][valid], atol=1e-8, rtol=1e-6)),
                      'gpp_max_abs_error': float(np.max(abs(free['gpp']-obs_host['gpp']))),
                      'saved_gpp_max_abs_error': float(np.max(abs(current_gpp-old_gpp))),
                      'shared_pool_max_abs_errors': shared_pool_errors}
            fluxes = {key: {label: np.asarray(value.fluxes[key].value).reshape(-1) for label, value in science.items()}
                      for key in ('gpp', 'npp', 'maintenance_respiration', 'growth_respiration', 'heterotrophic_respiration')}
            archive = args.output.with_name(args.output.stem+f'-{row["artifact_id"]}-T{h}.npz')
            np.savez_compressed(archive, free_state=obs_host['state'], free_defined=obs_host['defined'],
                teacher_state=np.asarray(targets['state']), teacher_defined=np.asarray(targets['defined']),
                teacher_start=np.asarray(days[0].continuous_state), reset_state=reset['state'],
                free_fraction=obs_host['fraction'], teacher_fraction=np.asarray(targets['fraction']))
            report['windows'].append({'selection': row, 'hard': hard, 'replay': replay,
                'pools_gC_m2': pools, 'decomposition_gC_m2': decomposition,
                'fluxes_gC_m2_day': fluxes, 'raw_state_archive': str(archive)})
            emit('window_completed', memory=_device_memory(runtime['devices'][0]))
        emit('completed', status='passed' if args.shape_only else 'completed',
             classification='diagnostic_shapes_and_pool_layout_only' if args.shape_only else 'attribution_evidence_not_model_acceptance')
    except Exception as error:
        emit('failed', status='failed', error=f'{type(error).__name__}: {error}')
        raise
    return report


if __name__ == '__main__':
    parser = training.parser()
    for name in ('selection', 'training-report', 'checkpoint', 'reference-report'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
