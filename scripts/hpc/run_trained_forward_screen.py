"""Day-disjoint trained 7/30-day rollout screen, without any parameter update."""
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.hpc import run_fresh_start_one_day as training
from scripts.hpc.trained_screen_common import WindowPool, expected_training_boundary, json_safe, load_model


def feedback_diagnostics(jax, jnp, runtime, head, model_protocol, initial, assembled, observations, targets, templates):
    """Report existing optional-loss geometry on free canonical trajectories only."""

    from research.daily_coarse_graining.daily_operator.canopy_change_loss import canopy_carbon_stock_change_loss
    from research.daily_coarse_graining.daily_operator.feedback_state_loss import feedback_state_mse
    from research.daily_coarse_graining.daily_operator.fresh_start_training import (
        _canopy_change_plan,
        _compact_canopy_carbon,
        _litterpart_feedback_plan,
        _pft_eligible,
        _state_leaf,
    )
    from research.daily_coarse_graining.daily_operator.parameter_routing import PFT_TRAIT_BARE_SOIL

    canopy_plan = _canopy_change_plan(runtime, head.contract, model_protocol)
    litterpart_plan = _litterpart_feedback_plan(head, model_protocol['loss'])
    result = {'enabled': bool(canopy_plan is not None or litterpart_plan is not None)}
    if canopy_plan is not None:
        compact = jax.vmap(lambda state: _compact_canopy_carbon(state, canopy_plan))
        predicted_next = compact(observations['state'])
        predicted_start = jnp.concatenate((compact(initial[None, :]), predicted_next[:-1]), axis=0)
        teacher_next = compact(targets['state'])
        teacher_start = jnp.concatenate((compact(initial[None, :]), teacher_next[:-1]), axis=0)
        predicted_next_defined = compact(observations['defined'])
        predicted_start_defined = jnp.concatenate((compact(assembled.state_defined.other[None, :]), predicted_next_defined[:-1]), axis=0)
        teacher_next_defined = compact(targets['defined'])
        teacher_start_defined = jnp.concatenate((compact(assembled.state_defined.other[None, :]), teacher_next_defined[:-1]), axis=0)
        bare = templates.pft.pft_traits[:, :, PFT_TRAIT_BARE_SOIL] > 0.5
        canopy = canopy_carbon_stock_change_loss(
            start_pred=predicted_start,
            next_pred=predicted_next,
            start_teacher=teacher_start,
            next_teacher=teacher_next,
            start_pred_defined=predicted_start_defined,
            next_pred_defined=predicted_next_defined,
            start_teacher_defined=teacher_start_defined,
            next_teacher_defined=teacher_next_defined,
            active_pft_mask=observations['active'],
            bare_pft_mask=bare,
            scale=jnp.asarray(canopy_plan.scale, dtype=jnp.float64),
        )
        result['canopy_stock_change'] = {
            **canopy._asdict(),
            'weight': jnp.asarray(canopy_plan.weight, dtype=jnp.float64),
            'scale': jnp.asarray(canopy_plan.scale, dtype=jnp.float64),
            'scale_source_pft_index': jnp.asarray(canopy_plan.scale_source_pft_index, dtype=jnp.int32),
            'scale_pft_applicable': observations['active'][:, canopy_plan.scale_pft_index]
            & ~bare[:, canopy_plan.scale_pft_index],
        }
    if litterpart_plan is not None:
        def leaf(state):
            return _state_leaf(state, litterpart_plan.leaf)

        eligible = jax.vmap(lambda active: _pft_eligible(active, litterpart_plan))(observations['active'])
        litterpart = jax.vmap(
            lambda predicted, teacher, predicted_defined, teacher_defined, eligible: feedback_state_mse(
                predicted=predicted,
                teacher=teacher,
                predicted_defined=predicted_defined,
                teacher_defined=teacher_defined,
                eligible=eligible,
                scale=litterpart_plan.scale,
            )
        )(
            jax.vmap(leaf)(observations['state']),
            jax.vmap(leaf)(targets['state']),
            jax.vmap(leaf)(observations['defined']),
            jax.vmap(leaf)(targets['defined']),
            eligible,
        )
        result['litterpart_feedback'] = {
            **litterpart._asdict(),
            'weight': jnp.asarray(litterpart_plan.weight, dtype=jnp.float64),
            'scale': litterpart_plan.scale,
        }
    return result


def feedback_hard_counts(feedback):
    """Apply the optional-loss validity requirements to one free rollout window."""

    if not feedback['enabled']:
        return {}
    hard = {}
    canopy = feedback.get('canopy_stock_change')
    if canopy is not None:
        row_valid = np.asarray(canopy['row_valid'], dtype=bool)
        scale_valid = bool(np.asarray(canopy['scale_valid'], dtype=bool))
        applicable = np.asarray(canopy['scale_pft_applicable'], dtype=bool)
        active_slots = np.asarray(canopy['active_slot_count'])
        hard['canopy_change_invalid'] = int(np.count_nonzero(~row_valid) + (not scale_valid) + np.count_nonzero(~applicable))
        hard['canopy_change_empty'] = int(np.count_nonzero(active_slots == 0))
    litterpart = feedback.get('litterpart_feedback')
    if litterpart is not None:
        hard['litterpart_feedback_invalid'] = int(np.sum(np.asarray(litterpart['invalid_count'])))
        hard['litterpart_feedback_empty'] = int(np.count_nonzero(np.asarray(litterpart['expected_count']) == 0))
    return hard


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
        expected_update, expected_pool_protocol = expected_training_boundary(args)
        model_protocol = method['model_protocol']
        report['checkpoint_update'] = expected_update
        report['expected_pool_protocol'] = (None if expected_pool_protocol is None else str(args.expected_pool_protocol))
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
            first = jax.tree_util.tree_map(lambda v:v[0],days[0])
            assembled = head.input_assembler.assemble(first)
            if args.shape_only:
                if h not in executables:
                    _, observation_shapes = jax.eval_shape(forward,*arguments)
                    jax.eval_shape(
                        lambda observations: feedback_diagnostics(
                            jax, jnp, runtime, head, model_protocol, initial, assembled,
                            observations, targets, days[0],
                        ),
                        observation_shapes,
                    )
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
            feedback = feedback_diagnostics(
                jax, jnp, runtime, head, model_protocol, initial, assembled, obs, targets, days[0]
            )
            hard.update(feedback_hard_counts(feedback))
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
                'feedback_diagnostics':feedback,
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
    parser.add_argument('--expected-update',type=int)
    parser.add_argument('--expected-pool-protocol',type=Path)
    run(parser.parse_args())
