"""Validate and freeze metadata inputs for bounded mixed-learning runs."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Mapping

from scripts.hpc.run_fresh_start_one_day_pool import pool_plan, sampling_identity
from scripts.hpc.trained_screen_common import validate_selection

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 'mixed_learning_plan_v1'
EXPECTED_UPDATES = 200
EXPECTED_EPOCH_UPDATES = 5415
EXPECTED_EPOCH_SOURCE_DRAWS = 346539
EXPECTED_EPOCH_DRAWS = 346560
EXPECTED_EPOCH_PADDING_DRAWS = 21
EXPECTED_RECURRENT_SEED = 20260914
EXPECTED_SAMPLING = {
    'seed':20260912,
    'batch_size':64,
    'microbatch_size':8,
    'max_artifacts_per_batch':5,
    'batch_packing':'fixed_replay_pad',
}


def _resolved_reference(value, *, relative_to):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = (ROOT/path, relative_to.parent/path, relative_to.parent/path.name)
    return next((candidate.resolve() for candidate in candidates if candidate.exists()),
                candidates[0].resolve())


def _metadata_sha256(value):
    payload = json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _day_keys(rows):
    return {(row['landpoint_id'],int(row['year']),int(day))
            for row in rows for day in range(int(row['start_row']),
                                             int(row['start_row'])+int(row['length']))}


def _validate_configuration(configuration):
    schema_version = configuration.get('schema_version')
    expected_updates = {
        'mixed_learning_v1':EXPECTED_UPDATES,
        'mixed_learning_epoch_v1':EXPECTED_EPOCH_UPDATES,
        'mixed_learning_fresh_litter_v1':EXPECTED_EPOCH_UPDATES,
    }.get(schema_version)
    if expected_updates is None:
        raise ValueError('mixed-learning protocol schema drift')
    if configuration.get('updates') != expected_updates:
        raise ValueError(
            f'mixed-learning protocol must declare exactly {expected_updates} updates'
        )
    if configuration.get('sampling') != EXPECTED_SAMPLING:
        raise ValueError('mixed-learning daily sampling protocol drift')
    if configuration.get('recurrent_seed') != EXPECTED_RECURRENT_SEED:
        raise ValueError('mixed-learning recurrent seed drift')
    return expected_updates


def _validate_recurrent_schedule(
        schedule, development, monitor_rows, *, updates=EXPECTED_UPDATES):
    if (schedule.get('status') != 'passed'
            or schedule.get('schema_version') != 'mixed_learning_recurrent_schedule_v1'):
        raise ValueError('recurrent training schedule is not accepted')
    if (schedule.get('role') != 'train_only_mixed_learning_schedule'
            or schedule.get('not_validation') is not True
            or schedule.get('spatial_temporal_split') != 'train/train'):
        raise ValueError('recurrent schedule is not a train-only learning schedule')
    if (schedule.get('updates') != updates
            or schedule.get('seed') != EXPECTED_RECURRENT_SEED
            or schedule.get('windows_per_update') != {'T7':1,'T30':1}):
        raise ValueError('recurrent schedule count, seed or horizon multiplicity drift')
    if (schedule.get('teacher_payload_reads') != 0
            or schedule.get('teacher_regenerated') is not False
            or schedule.get('sealed_test_accessed') is not False
            or schedule.get('parameter_updates') != 0):
        raise ValueError('recurrent schedule preparation crossed its metadata-only boundary')

    rows = schedule.get('update_rows')
    if not isinstance(rows,list) or len(rows) != updates:
        raise ValueError(f'recurrent schedule must contain {updates} ordered update rows')
    monitor_days = {(row['landpoint_id'],int(row['year']),int(row['start_row']))
                    for row in monitor_rows}
    development_days = _day_keys(development['windows'])
    excluded = monitor_days | development_days
    identities,semantic_identities,ordered_table = set(),set(),[]
    for expected_update,entry in enumerate(rows,start=1):
        if entry.get('update') != expected_update or set(entry) != {'update','T7','T30'}:
            raise ValueError('recurrent update order or row keys drift')
        update_days = {}
        for horizon,length,component in (('T7',7,'seven_day'),('T30',30,'thirty_day')):
            row = entry.get(horizon)
            if not isinstance(row,Mapping) or row.get('length') != length or row.get('component') != component:
                raise ValueError('recurrent component/horizon mismatch')
            semantic = tuple((row.get('landpoint_id'),row.get('year'),day)
                             for day in range(row.get('start_row'),row.get('start_row')+length))
            if tuple(map(tuple,row.get('teacher_day_keys',()))) != semantic:
                raise ValueError('recurrent Teacher-day identities differ from semantic window')
            identity = (row.get('artifact_id'),component,row.get('window_index'))
            if identity in identities:
                raise ValueError('recurrent window identity is repeated')
            identities.add(identity)
            semantic_identity = (row.get('landpoint_id'),row.get('year'),row.get('start_row'),length)
            if semantic_identity in semantic_identities:
                raise ValueError('recurrent semantic window identity is repeated')
            semantic_identities.add(semantic_identity)
            update_days[horizon] = set(semantic)
            if update_days[horizon] & excluded:
                raise ValueError('recurrent training window overlaps a fixed exclusion')
            ordered_table.append({'update':expected_update,'horizon':horizon,'window':row})
        if update_days['T7'] & update_days['T30']:
            raise ValueError('T7 and T30 windows overlap within one update')
    if schedule.get('window_table') != ordered_table:
        raise ValueError('recurrent flat window table differs from ordered update rows')
    exclusions = schedule.get('exclusions',{})
    if (exclusions.get('monitor_day_count') != len(monitor_days)
            or exclusions.get('development_window_day_count') != len(development_days)
            or exclusions.get('combined_day_count') != len(excluded)):
        raise ValueError('recurrent exclusion counts differ from loaded panels')
    return rows


def _epoch_daily_coverage(batches, sampler, *, batch_size):
    """Return epoch draws and unique source days after validating terminal replay."""
    if not batches:
        raise ValueError('daily epoch contains no batches')
    if any(batch.draw_count != batch_size for batch in batches):
        raise ValueError('daily epoch contains a non-fixed-size batch')
    if any(batch.padding_draw_count for batch in batches[:-1]):
        raise ValueError('daily epoch padding is not confined to the terminal batch')

    source_rows = tuple(
        row for batch in batches for row in batch.rows[:batch.source_draw_count]
    )
    padding_rows = tuple(
        row for batch in batches for row in batch.rows[batch.source_draw_count:]
    )
    daily_rows = tuple(row for batch in batches for row in batch.rows)
    source_days = {row.day_key for row in source_rows}
    if (len(source_rows) != sampler.source_draw_count
            or len(source_days) != sampler.source_draw_count):
        raise ValueError('daily epoch does not cover every source Teacher day exactly once')
    if (len(daily_rows) != sampler.draw_count
            or len(padding_rows) != sampler.padding_draw_count
            or batches[-1].padding_draw_count != sampler.padding_draw_count):
        raise ValueError('daily epoch draw or declared padding count drift')
    terminal_source_days = {
        row.day_key for row in batches[-1].rows[:batches[-1].source_draw_count]
    }
    if any(row.day_key not in terminal_source_days for row in padding_rows):
        raise ValueError('daily epoch padding is not a replay of terminal source rows')
    return daily_rows,source_days


def load_plan(args, configuration):
    """Load, validate and bind a complete bounded metadata-only plan."""
    updates = _validate_configuration(configuration)
    epoch_protocol = configuration['schema_version'] in (
        'mixed_learning_epoch_v1', 'mixed_learning_fresh_litter_v1')
    report_path = Path(args.catalogue_report).resolve()
    schedule_path = Path(args.recurrent_schedule).resolve()
    selection_path = Path(args.selection).resolve()

    report = _read_json(report_path)
    if (report.get('status') != 'passed'
            or report.get('schema_version') != 'expanded_one_day_merge_v1'):
        raise ValueError('expanded one-day catalogue report is not accepted')
    catalogue_path = _resolved_reference(report['catalogue'],relative_to=report_path)
    with gzip.open(catalogue_path,'rt',encoding='utf-8') as handle:
        catalogue = json.load(handle)
    if catalogue.get('source_identity') != report.get('source_identity'):
        raise ValueError('catalogue and merge-report source identities differ')
    daily_source = _resolved_reference(catalogue['source_manifest'],relative_to=catalogue_path)
    daily_manifest = _read_json(daily_source)
    if daily_manifest.get('status') != 'passed' or daily_manifest.get('identity') != catalogue['source_identity']:
        raise ValueError('catalogue and prepared daily source differ')

    sampler,panels = pool_plan(catalogue,configuration)
    batches,epoch_batches,max_artifacts = [],0,0
    for batch in sampler.iter_epoch(0):
        epoch_batches += 1
        max_artifacts = max(max_artifacts,len(batch.artifact_ids))
        if epoch_protocol or len(batches) < updates:
            batches.append(batch)
    if len(batches) != updates or max_artifacts > EXPECTED_SAMPLING['max_artifacts_per_batch']:
        raise ValueError(f'daily pool cannot supply the declared {updates} bounded batches')
    if epoch_protocol:
        daily_rows,daily_days = _epoch_daily_coverage(
            batches,sampler,batch_size=EXPECTED_SAMPLING['batch_size'],
        )
    else:
        daily_rows = tuple(row for batch in batches for row in batch.rows)
        daily_days = {row.day_key for row in daily_rows}
        if (len(daily_rows) != updates*EXPECTED_SAMPLING['batch_size']
                or len(daily_days) != len(daily_rows)):
            raise ValueError('first 200 daily batches are not 12,800 unique Teacher days')

    development = _read_json(selection_path)
    if development.get('status') != 'passed':
        raise ValueError('development selection is not accepted')
    validate_selection(development)
    schedule = _read_json(schedule_path)
    schedule_rows = _validate_recurrent_schedule(
        schedule,development,catalogue['monitor_rows'],updates=updates,
    )
    recurrent_source = _resolved_reference(schedule['source_manifest'],relative_to=schedule_path)
    recurrent_manifest = _read_json(recurrent_source)
    if recurrent_manifest.get('status') != 'passed' or recurrent_manifest.get('identity') != schedule['source_identity']:
        raise ValueError('recurrent schedule and prepared source differ')
    if development.get('source_identity') != schedule['source_identity']:
        raise ValueError('development and recurrent prepared-source identities differ')

    exclusion_paths = schedule.get('exclusions',{})
    if (_resolved_reference(exclusion_paths.get('expanded_catalogue',''),relative_to=schedule_path) != catalogue_path
            or _resolved_reference(exclusion_paths.get('development_selection',''),relative_to=schedule_path) != selection_path):
        raise ValueError('recurrent exclusion source paths differ from loaded inputs')

    summary = {
        'updates':updates,
        'daily_draws':len(daily_rows),
        'daily_unique_days':len(daily_days),
        'daily_unique_sites':len({row.landpoint_id for row in daily_rows}),
        'daily_unique_years':len({row.year for row in daily_rows}),
        'daily_unique_point_years':len({(row.landpoint_id,row.year) for row in daily_rows}),
        'daily_epoch_batches':epoch_batches,
        'daily_epoch_source_rows':sampler.source_draw_count,
        'daily_epoch_draws':sampler.draw_count,
        'daily_epoch_padding_draws':sampler.padding_draw_count,
        'daily_epoch_max_artifacts_per_batch':max_artifacts,
        'recurrent_windows':2*len(schedule_rows),
        'monitor_days':len(catalogue['monitor_rows']),
        'development_windows':len(development['windows']),
    }
    if epoch_protocol:
        if report.get('counts',{}).get('distinct_train_days') != EXPECTED_EPOCH_SOURCE_DRAWS:
            raise ValueError('epoch protocol requires the formal expanded daily pool')
        expected = {
            'daily_draws':EXPECTED_EPOCH_DRAWS,
            'daily_unique_days':EXPECTED_EPOCH_SOURCE_DRAWS,
            'daily_epoch_batches':EXPECTED_EPOCH_UPDATES,
            'daily_epoch_source_rows':EXPECTED_EPOCH_SOURCE_DRAWS,
            'daily_epoch_draws':EXPECTED_EPOCH_DRAWS,
            'daily_epoch_padding_draws':EXPECTED_EPOCH_PADDING_DRAWS,
            'daily_epoch_max_artifacts_per_batch':5,
        }
        if any(summary[key] != value for key,value in expected.items()):
            raise ValueError('formal expanded-pool epoch coverage differs from metadata admission')
    elif report.get('counts',{}).get('distinct_train_days') == EXPECTED_EPOCH_SOURCE_DRAWS:
        expected = {
            'daily_draws':12800,
            'daily_unique_days':12800,
            'daily_unique_sites':372,
            'daily_unique_years':44,
            'daily_epoch_max_artifacts_per_batch':5,
        }
        if any(summary[key] != value for key,value in expected.items()):
            raise ValueError('formal expanded-pool 200-update coverage differs from metadata admission')
    identities = {
        'daily_source':catalogue['source_identity'],
        'catalogue_order':sampling_identity(catalogue),
        'recurrent_source':schedule['source_identity'],
        'recurrent_schedule':_metadata_sha256(schedule_rows),
        'development_selection':_metadata_sha256({
            'source_identity':development['source_identity'],
            'windows':development['windows'],
        }),
    }
    paths = {
        'catalogue_report':report_path,
        'catalogue':catalogue_path,
        'daily_source':daily_source,
        'recurrent_schedule':schedule_path,
        'recurrent_source':recurrent_source,
        'development_selection':selection_path,
    }
    return {
        'catalogue':catalogue,
        'catalogue_path':catalogue_path,
        'daily_source':daily_source,
        'sampler':sampler,
        'batches':tuple(batches),
        'schedule':schedule,
        'monitor_rows':tuple(panels['excluded_days']),
        'summary':summary,
        'identities':identities,
        'paths':paths,
    }


def run(args):
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    configuration = _read_json(args.mixed_protocol)
    plan = load_plan(args,configuration)
    report = {
        'status':'passed',
        'schema_version':SCHEMA_VERSION,
        'configuration':configuration,
        'summary':plan['summary'],
        'identities':plan['identities'],
        'paths':{key:str(value) for key,value in plan['paths'].items()},
        'teacher_payload_reads':0,
        'parameter_updates':0,
    }
    _atomic_json(Path(args.output).resolve(),report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('catalogue-report','recurrent-schedule','selection','mixed-protocol','output'):
        result.add_argument('--'+name,type=Path,required=True)
    return result


if __name__ == '__main__':
    run(parser().parse_args())
