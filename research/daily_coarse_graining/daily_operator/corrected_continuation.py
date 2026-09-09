"""Corrected-model experiment plan and atomic continuation state (no process logic)."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

from .target_oriented_coupled_gpp_artifact_aware_direction import SCHEDULED_NON_GPP_SOURCES
from .target_oriented_coupled_gpp_conditioned_attention_persistence import _atomic_pickle
from .target_oriented_coupled_gpp_conditioned_attention_persistent_training import (
    ConditionedAttentionPersistentTrainingContract,
    _artifact_sources,
    load_conditioned_attention_persistent_training_contract,
)
from .target_oriented_coupled_gpp_persistent_scale_training import build_persistent_scale_training_schedule

DEFAULT_PLAN = Path(__file__).resolve().parents[3] / 'manifests/coarse_graining/conditioned_snow_free_continuation_v1.json'


@dataclass(frozen=True)
class CorrectedContinuationContract(ConditionedAttentionPersistentTrainingContract):
    plan: dict

    @property
    def logical_updates(self):
        return tuple(self.plan['logical_updates'])

    @property
    def train_panels(self):
        return tuple(tuple(tuple(g) for g in self.plan['train_panels'][str(u)]) for u in self.logical_updates)

    @property
    def source_registries(self):
        return tuple(_artifact_sources(panel) + SCHEDULED_NON_GPP_SOURCES for panel in self.train_panels)


def load_contract(path=DEFAULT_PLAN):
    plan = json.loads(Path(path).read_text())
    base = load_conditioned_attention_persistent_training_contract()
    version = plan['schema_version']
    if version not in ('conditioned_snow_free_continuation_v1', 'conditioned_snow_free_continuation_v2'):
        raise ValueError('unknown corrected-continuation plan')
    start, stop = (4, 14) if version.endswith('_v1') else (14, 34)
    if (plan['logical_updates'] != list(range(start + 1, stop + 1)) or plan['starting_logical_update'] != start
            or plan['warm_snow_free_boundary'] is not True or tuple(plan['monitor_updates']) != base.monitor_updates):
        raise ValueError('incorrect corrected-continuation plan')
    if version.endswith('_v1'):
        schedule = build_persistent_scale_training_schedule()
        expected_panels = [[list(schedule.groups[i - 1]) for i in (u, *range(4*u + 3, 4*u + 7))]
                           for u in plan['logical_updates']]
    else:
        schedule = build_persistent_scale_training_schedule(seed=20260910, maximum_updates=20, macro_batch_size=40)
        expected_panels = [[list(block[i:i + 8]) for i in range(0, 40, 8)] for block in schedule.groups]
    for u, expected in zip(plan['logical_updates'], expected_panels, strict=True):
        if plan['train_panels'][str(u)] != expected:
            raise ValueError('continuation panel differs from approved schedule')
    artifacts = [v for groups in plan['train_panels'].values() for group in groups for v in group]
    if (version.endswith('_v1') and len(set(artifacts)) != 400) or not set(artifacts).isdisjoint(base.monitor_updates):
        raise ValueError('continuation artifact coverage or monitor overlap')
    return CorrectedContinuationContract(base.payload, Path(path), base.sha256, plan)


def read_pickle(path):
    with Path(path).open('rb') as handle:
        return pickle.load(handle)


def save_state(path, state):
    _atomic_pickle(Path(path), state)


def read_state(path, plan):
    path = Path(path)
    if not path.exists():
        return None
    state = read_pickle(path)
    if state.get('schema_version') != 'corrected_continuation_state_v1' or state.get('plan') != plan:
        raise ValueError('recovery belongs to a different continuation')
    completed = state['completed_logical_update']
    start = plan['starting_logical_update']
    if completed not in range(start, plan['logical_updates'][-1] + 1) or [r['logical_update'] for r in state['records']] != list(range(start + 1, completed + 1)):
        raise ValueError('noncontiguous corrected recovery progress')
    if len(state['monitors']) != len(state['records']):
        raise ValueError('corrected recovery is missing accepted monitors')
    return state


def continue_training(contract, state, directory, *, propose, monitor, transfer, host, emit, max_updates=None):
    """Resume pending monitoring before computing another proposal; commit only accepted steps."""
    directory = Path(directory)
    pending_path, recovery_path = directory / 'pending.pkl', directory / 'recovery.pkl'
    done_this_call = 0
    if state.get('stopped_reason'):
        return state
    for update, panel, registry in zip(contract.logical_updates, contract.train_panels, contract.source_registries, strict=True):
        if update <= state['completed_logical_update']:
            continue
        pending = read_pickle(pending_path) if pending_path.exists() else None
        if pending is not None and pending['logical_update'] > state['completed_logical_update']:
            if pending['logical_update'] != update or pending['plan'] != contract.plan:
                raise ValueError('pending candidate does not follow accepted recovery')
            emit('resume_pending_monitor', {'logical_update': update})
        else:
            result = propose(state['trainable'], update, panel, registry)
            record = result['record']
            if not result['gradient_admitted'] or not record['selection']['accepted']:
                state['stopped_reason'] = 'gradient_score' if not result['gradient_admitted'] else 'train_only'
                state['rejected'] = {'record': record, 'gradient_evidence': result['gradient_evidence']}
                save_state(recovery_path, state)
                emit('stopped', {'reason': state['stopped_reason'], 'rejected': state['rejected']})
                break
            pending = {'plan': contract.plan, 'logical_update': update, 'record': record,
                       'trainable': host(result['updated_trainable']), 'gradient_evidence': result['gradient_evidence']}
            save_state(pending_path, pending)
            emit('pending_saved', {'logical_update': update})
        measured = monitor(pending['trainable'], update)
        before = state['monitors'][-1]['panel'] if state['monitors'] else state['initial_monitor']
        local_gate, cumulative_gate = transfer(before, measured), transfer(state['initial_monitor'], measured)
        origin_gate = transfer(state.get('original_monitor', state['initial_monitor']), measured)
        if not (local_gate['passed'] and cumulative_gate['passed'] and origin_gate['passed']):
            state['stopped_reason'] = 'monitor_transfer'
            state['rejected'] = {'logical_update': update, 'local_gate': local_gate,
                                 'cumulative_gate': cumulative_gate, 'original_gate': origin_gate, 'panel': measured}
            save_state(recovery_path, state)
            emit('stopped', {'reason': state['stopped_reason'], 'rejected': state['rejected']})
            break
        state['trainable'] = pending['trainable']
        state['completed_logical_update'] = update
        state['records'].append(pending['record'])
        state['monitors'].append({'logical_update': update, 'panel': measured,
            'local_gate': local_gate, 'cumulative_gate': cumulative_gate, 'original_gate': origin_gate,
            'gradient_evidence': pending['gradient_evidence']})
        save_state(recovery_path, state)
        emit('update_retained', {'logical_update': update, 'new_updates': update - contract.plan['starting_logical_update'],
                                'completed_logical_update': update,
                                'local_gate': local_gate, 'cumulative_gate': cumulative_gate})
        if update in contract.plan['milestones']:
            emit('milestone', {'new_updates': update - contract.plan['starting_logical_update']})
        done_this_call += 1
        if max_updates is not None and done_this_call >= max_updates:
            break
    return state
