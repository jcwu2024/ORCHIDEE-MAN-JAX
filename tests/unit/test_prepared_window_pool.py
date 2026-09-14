import gzip
import json
import pickle
from types import SimpleNamespace

import pytest

from scripts.hpc.trained_screen_common import PreparedWindowPool, WindowPool


def _write_artifact(path,identifier):
    artifact = SimpleNamespace(schedule_entry={'update':identifier})
    with gzip.open(path,'wb') as handle:
        pickle.dump(artifact,handle)


def test_prepared_window_pool_reads_explicit_artifacts_with_bounded_lru_cache(tmp_path):
    identity = {'training_contract_sha256':'contract'}
    entries = []
    for identifier in (11,12,13):
        path = tmp_path/f'artifact-{identifier}.pkl.gz'
        _write_artifact(path,identifier)
        entries.append({'update':identifier,'path':path.name})
    manifest_path = tmp_path/'prepared_manifest.json'
    manifest_path.write_text(json.dumps({
        'status':'passed','identity':identity,'entries':entries,
    }))

    pool = PreparedWindowPool(manifest_path,cache_size=2)
    assert pool.identity == identity
    assert pool.load(11).schedule_entry['update'] == 11
    assert pool.load(12).schedule_entry['update'] == 12
    assert list(pool.cache) == [11,12]
    pool.load(11)
    pool.load(13)
    assert list(pool.cache) == [11,13]


def test_prepared_window_pool_does_not_relax_strict_development_selection(tmp_path):
    manifest_path = tmp_path/'prepared_manifest.json'
    manifest_path.write_text(json.dumps({
        'status':'passed','identity':{},'entries':[],
    }))
    selection_path = tmp_path/'selection.json'
    selection_path.write_text(json.dumps({
        'status':'passed','source_manifest':str(manifest_path),
        'source_identity':{},'windows':[],
    }))

    assert PreparedWindowPool(manifest_path).identity == {}
    with pytest.raises(ValueError,match='exactly eight windows'):
        WindowPool(selection_path)


def test_prepared_window_pool_rejects_bad_cache_and_duplicate_artifacts(tmp_path):
    manifest_path = tmp_path/'prepared_manifest.json'
    manifest_path.write_text(json.dumps({
        'status':'passed','identity':{},
        'entries':[{'update':1,'path':'a'},{'update':1,'path':'b'}],
    }))

    with pytest.raises(ValueError,match='positive cache size'):
        PreparedWindowPool(manifest_path,cache_size=0)
    with pytest.raises(ValueError,match='duplicate prepared artifact'):
        PreparedWindowPool(manifest_path)


@pytest.mark.parametrize('row,match',[
    ({'artifact_id':1,'component':'thirty_day','window_index':0,'length':7,
      'landpoint_id':'site','year':2000,'start_row':10,
      'teacher_day_keys':[['site',2000,day] for day in range(10,17)]},
     'component/horizon'),
    ({'artifact_id':1,'component':'seven_day','window_index':0,'length':7,
      'landpoint_id':'site','year':2000,'start_row':10,
      'teacher_day_keys':[['site',2000,day] for day in range(11,18)]},
     'Teacher-day identities'),
])
def test_prepared_window_pool_checks_explicit_row_before_payload_access(row,match):
    pool = object.__new__(PreparedWindowPool)
    pool.load = lambda _identifier: pytest.fail('invalid metadata reached payload access')

    with pytest.raises(ValueError,match=match):
        pool.case(None,row)
