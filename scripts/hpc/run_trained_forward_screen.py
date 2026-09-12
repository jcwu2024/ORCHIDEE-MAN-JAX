"""Day-disjoint trained 7/30-day rollout screen, without any parameter update."""
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.hpc import run_fresh_start_one_day as training
from scripts.hpc.trained_screen_common import WindowPool, json_safe, load_model


def run(args):
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    if args.admission_only or args.cpu_forward_only:
        raise ValueError('use --shape-only for this forward screen preflight')
    started = time.perf_counter()
    report = {'status':'running','training_updates':0,'windows':[],'horizons':{},
              'source_git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'checkpoint':str(args.checkpoint),'selection':str(args.selection),
              'scope':'train_domain_disjoint_days_not_formal_spatial_temporal_acceptance'}

    def emit(stage,**values):
        report.update(values,stage=stage,elapsed_seconds=time.perf_counter()-started)
        _atomic_json(args.output,json_safe(report))
        print(f'forward_screen stage={stage} windows={len(report["windows"])} seconds={report["elapsed_seconds"]:.1f}',flush=True)

    try:
        emit('bootstrap')
        runtime,head,template,scales,parameters,method = load_model(args)
        jax = runtime['jax']
        import jax.numpy as jnp

        from research.daily_coarse_graining.daily_operator.canonical_rollout import (
            make_canonical_step,
            make_science_probe_loss,
        )
        from research.daily_coarse_graining.daily_operator.forward_screen_metrics import (
            complete_metric_gate,
            physical_window,
            summarize_windows,
        )
        from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
        from research.daily_coarse_graining.daily_operator.successor_training import (
            extract_biomass_productivity_trajectories,
        )
        from research.daily_coarse_graining.daily_operator.target_oriented_training import (
            _acceptance_trajectories,
            add_target_oriented_acceptance_statistics,
            derive_target_oriented_acceptance_scales,
            finalize_target_oriented_acceptance_statistics,
            target_oriented_acceptance_sufficient_statistics,
        )
        from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory
        from scripts.hpc.probe_public_rollout import physical_checks

        pool = WindowPool(args.selection,method['source_identity'])
        component = make_rollout_loss(jax,jnp,make_canonical_step(head,template),make_science_probe_loss(head.contract,scales))
        forward = jax.jit(component.rollout)
        acceptance_scales = derive_target_oriented_acceptance_scales(
            runtime['base_statistics'].arrays['next_state'].scale,
            runtime['process_statistics'].outputs['daily.fast_gpp'].scale,head.contract)

        def extract(state,defined,gpp,gpp_defined,fraction,active):
            return extract_biomass_productivity_trajectories(state,defined,
                daily_fast_gpp=gpp,daily_fast_gpp_defined=gpp_defined,
                pft_fraction=fraction,active_pft_mask=active,contract=head.contract)

        extract = jax.jit(extract)
        executables,statistics,physical = {},{},{}
        report['training_method'] = method
        for row in pool.selection['windows']:
            h = row['length']
            initial,discrete,days,targets,expected_discrete = pool.case(jax,row)
            arguments = (parameters,initial,discrete,days)
            if args.shape_only:
                if h not in executables:
                    jax.eval_shape(forward,*arguments)
                    expected = extract(targets['state'],targets['defined'],targets['gpp'],targets['gpp_defined'],targets['fraction'],targets['active'])
                    ef,es = _acceptance_trajectories(expected.fluxes,expected.stocks)
                    summarize_windows([physical_window(ef,ef,es,es,es)])
                    target_oriented_acceptance_sufficient_statistics(expected.fluxes,expected.fluxes,
                        expected.stocks,expected.stocks,scales=acceptance_scales,horizon=h)
                    executables[h] = True
                continue
            if h not in executables:
                emit('compile_forward',horizon=h)
                began = time.perf_counter()
                compiled = forward.lower(*arguments).compile()
                memory = _compiled_memory(compiled)
                if memory['estimated_live_bytes']>18*1024**3:
                    emit('stopped',status='resource_limited',horizon=h,compiled_memory=memory)
                    return report
                executables[h] = compiled
                report['horizons'][str(h)] = {'compile_seconds':time.perf_counter()-began,'compiled_memory':memory}
            began = time.perf_counter()
            _,obs = executables[h](*arguments)
            jax.block_until_ready(obs)
            elapsed = time.perf_counter()-began
            hard = physical_checks(obs,targets,expected_discrete,days[0],head.contract,scales)
            predicted = extract(obs['state'],obs['defined'],obs['gpp'],obs['gpp_defined'],obs['fraction'],obs['active'])
            expected = extract(targets['state'],targets['defined'],targets['gpp'],targets['gpp_defined'],targets['fraction'],targets['active'])
            first = jax.tree_util.tree_map(lambda v:v[0],days[0])
            assembled = head.input_assembler.assemble(first)
            initial_science = extract(initial[None,:],assembled.state_defined.other[None,:],
                targets['gpp'][:1],targets['gpp_defined'][:1],targets['fraction'][:1],targets['active'][:1])
            pf,ps = _acceptance_trajectories(predicted.fluxes,predicted.stocks)
            tf,ts = _acceptance_trajectories(expected.fluxes,expected.stocks)
            _,ss = _acceptance_trajectories(initial_science.fluxes,initial_science.stocks)
            metrics = physical_window(pf,tf,ps,ts,ss)
            stat = target_oriented_acceptance_sufficient_statistics(predicted.fluxes,expected.fluxes,
                predicted.stocks,expected.stocks,scales=acceptance_scales,horizon=h)
            statistics[h] = stat if h not in statistics else add_target_oriented_acceptance_statistics(statistics[h],stat)
            physical.setdefault(h,[]).append(metrics)
            a,b = np.asarray(predicted.fluxes['npp'].value),np.asarray(pf['npp'].value)
            npp_valid = np.asarray(predicted.fluxes['npp'].defined)&np.asarray(pf['npp'].defined)
            npp_finite = np.isfinite(a)&np.isfinite(b)
            npp_error = abs(np.where(npp_valid&npp_finite,a,0.)-np.where(npp_valid&npp_finite,b,0.))
            report['windows'].append({'selection':row,'prepared_day_indices':np.asarray(days[0].day_index),
                'forward_seconds':elapsed,'hard':hard,'physical':metrics,
                'source_npp_identity_error':float(np.max(npp_error,initial=0.)),
                'source_npp_nonfinite_defined':int(np.sum(npp_valid&~npp_finite))})
            report['horizons'][str(h)].update(
                science=finalize_target_oriented_acceptance_statistics(statistics[h],horizon=h,target_contract=head.target_contract),
                physical=summarize_windows(physical[h]))
            emit('window_completed',memory=_device_memory(runtime['devices'][0]))
        if args.shape_only:
            emit('completed',status='passed',classification='both_forward_horizon_shapes_and_metric_payloads_only')
            return report
        hard_pass = all(not any(row['hard'].values()) for row in report['windows'])
        complete = set(report['horizons'])=={'7','30'} and all(complete_metric_gate(
            row['physical'],row['science'],int(h),len(physical[int(h)])) for h,row in report['horizons'].items())
        science_pass = complete and all(row['science']['all_key_gates_passed'] for row in report['horizons'].values())
        emit('completed',status='completed',physical_gate_passed=hard_pass,
             metric_population_complete=complete,panel_science_gates_passed=science_pass,preliminary_screen_passed=hard_pass and science_pass,
             formal_spatial_temporal_acceptance=False,memory=_device_memory(runtime['devices'][0]))
    except Exception as error:
        emit('failed',status='failed',error=f'{type(error).__name__}: {error}')
        raise
    return report


if __name__=='__main__':
    parser = training.parser()
    for name in ('selection','training-report','checkpoint'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
