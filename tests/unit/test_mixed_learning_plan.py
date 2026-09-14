import gzip
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.hpc.mixed_learning_plan import EXPECTED_SAMPLING, load_plan


IDENTITY = {
    'dataset_manifest_sha256':'dataset',
    'schedule_sha256':'schedule',
    'training_contract_sha256':'contract',
}


def _window(prefix,artifact_id,length,start=0):
    component = {7:'seven_day',30:'thirty_day'}[length]
    return {
        'artifact_id':artifact_id,
        'component':component,
        'window_index':0,
        'length':length,
        'landpoint_id':prefix,
        'year':2000,
        'start_row':start,
        'teacher_day_keys':[[prefix,2000,day] for day in range(start,start+length)],
    }


def _fixture(tmp_path):
    daily_identity = {'training_contract_sha256':'daily'}
    daily_source = tmp_path/'daily-manifest.json'
    daily_source.write_text(json.dumps({
        'status':'passed','identity':daily_identity,'entries':[],
    }))
    entries = []
    for identifier in range(201):
        entries.append({
            'update':identifier+1,
            'runtime_year':2000,
            'components':{'one_day':[{
                'landpoint_id':f'daily-{identifier:03d}',
                'year':2000,
                'start_rows':list(range(64)),
            }]},
        })
    monitor = [
        {'artifact_id':index+1,'batch_row':0,
         'landpoint_id':f'daily-{index:03d}','year':2000,'start_row':0}
        for index in range(64)
    ]
    catalogue = {
        'source_manifest':str(daily_source),
        'source_identity':daily_identity,
        'one_day_entries':entries,
        'monitor_rows':monitor,
    }
    catalogue_path = tmp_path/'catalogue.json.gz'
    with gzip.open(catalogue_path,'wt',encoding='utf-8') as handle:
        json.dump(catalogue,handle)
    report_path = tmp_path/'merge-report.json'
    report_path.write_text(json.dumps({
        'status':'passed','schema_version':'expanded_one_day_merge_v1',
        'catalogue':str(catalogue_path),'source_identity':daily_identity,
    }))

    recurrent_source = tmp_path/'recurrent-manifest.json'
    recurrent_source.write_text(json.dumps({
        'status':'passed','identity':IDENTITY,'entries':[],
    }))
    development_rows = []
    for length in (7,30):
        for index in range(8):
            development_rows.append(_window(
                f'development-{length}-{index}',100+index,length,10,
            ))
    selection_path = tmp_path/'selection.json'
    selection_path.write_text(json.dumps({
        'status':'passed','source_manifest':str(recurrent_source),
        'source_identity':IDENTITY,'windows':development_rows,
    }))

    update_rows,window_table = [],[]
    for update in range(1,201):
        t7 = _window(f'train-7-{update}',1000+update,7,20)
        t30 = _window(f'train-30-{update}',2000+update,30,100)
        update_rows.append({'update':update,'T7':t7,'T30':t30})
        window_table.extend((
            {'update':update,'horizon':'T7','window':t7},
            {'update':update,'horizon':'T30','window':t30},
        ))
    schedule = {
        'status':'passed','schema_version':'mixed_learning_recurrent_schedule_v1',
        'role':'train_only_mixed_learning_schedule','not_validation':True,
        'spatial_temporal_split':'train/train','seed':20260914,'updates':200,
        'windows_per_update':{'T7':1,'T30':1},
        'teacher_payload_reads':0,'teacher_regenerated':False,
        'sealed_test_accessed':False,'parameter_updates':0,
        'source_manifest':str(recurrent_source),'source_identity':IDENTITY,
        'update_rows':update_rows,'window_table':window_table,
        'exclusions':{
            'expanded_catalogue':str(catalogue_path),
            'development_selection':str(selection_path),
            'monitor_day_count':64,'development_window_day_count':296,
            'combined_day_count':360,
        },
    }
    schedule_path = tmp_path/'schedule.json'
    schedule_path.write_text(json.dumps(schedule))
    args = SimpleNamespace(
        catalogue_report=report_path,
        recurrent_schedule=schedule_path,
        selection=selection_path,
    )
    configuration = {
        'schema_version':'mixed_learning_v1','updates':200,
        'sampling':dict(EXPECTED_SAMPLING),'recurrent_seed':20260914,
    }
    return args,configuration,schedule_path


def test_load_plan_freezes_200_daily_batches_and_recurrent_pairs(tmp_path):
    args,configuration,_ = _fixture(tmp_path)
    plan = load_plan(args,configuration)

    assert len(plan['batches']) == 200
    assert len(plan['schedule']['update_rows']) == 200
    assert len(plan['monitor_rows']) == 64
    assert plan['summary'] == {
        'updates':200,
        'daily_draws':12800,
        'daily_unique_days':12800,
        'daily_unique_sites':201,
        'daily_unique_years':1,
        'daily_unique_point_years':201,
        'daily_epoch_batches':200,
        'daily_epoch_source_rows':12800,
        'daily_epoch_draws':12800,
        'daily_epoch_padding_draws':0,
        'daily_epoch_max_artifacts_per_batch':2,
        'recurrent_windows':400,
        'monitor_days':64,
        'development_windows':16,
    }
    assert plan['identities']['daily_source'] == {'training_contract_sha256':'daily'}
    assert len(plan['identities']['catalogue_order']) == 64
    assert len(plan['identities']['recurrent_schedule']) == 64
    assert plan['daily_source'] == plan['paths']['daily_source']


@pytest.mark.parametrize('drift,match',[
    ('teacher_days','Teacher-day identities'),
    ('repeat','window identity is repeated'),
    ('semantic_repeat','semantic window identity is repeated'),
    ('overlap','overlap within one update'),
    ('flat','flat window table differs'),
])
def test_load_plan_rejects_recurrent_schedule_drift(tmp_path,drift,match):
    args,configuration,schedule_path = _fixture(tmp_path)
    schedule = json.loads(schedule_path.read_text())
    if drift == 'teacher_days':
        schedule['update_rows'][0]['T7']['teacher_day_keys'][0][2] += 1
    elif drift == 'repeat':
        schedule['update_rows'][1]['T7']['artifact_id'] = schedule['update_rows'][0]['T7']['artifact_id']
    elif drift == 'semantic_repeat':
        first = schedule['update_rows'][0]['T7']
        second = schedule['update_rows'][1]['T7']
        second.update(
            landpoint_id=first['landpoint_id'],year=first['year'],start_row=first['start_row'],
            teacher_day_keys=deepcopy(first['teacher_day_keys']),
        )
    elif drift == 'overlap':
        schedule['update_rows'][0]['T30'] = deepcopy(schedule['update_rows'][0]['T30'])
        schedule['update_rows'][0]['T30'].update(
            landpoint_id=schedule['update_rows'][0]['T7']['landpoint_id'],start_row=20,
            teacher_day_keys=[
                [schedule['update_rows'][0]['T7']['landpoint_id'],2000,day]
                for day in range(20,50)
            ],
        )
    else:
        schedule['window_table'] = schedule['window_table'][:-1]
    schedule_path.write_text(json.dumps(schedule))

    with pytest.raises(ValueError,match=match):
        load_plan(args,configuration)
