import gzip
import json
import pickle
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool


class Input(NamedTuple):
    continuous_state: object


class Batch(NamedTuple):
    inputs: object
    target: object


def test_pool_preserves_row_order_and_caches_only_one_day(tmp_path):
    entries = []
    for identifier in (10,20,30):
        batch = Batch(Input(np.arange(6).reshape(3,2)+identifier), np.arange(3)+identifier)
        artifact = SimpleNamespace(schedule_entry={'update': identifier}, training_contract_sha256='data-contract',
                                   batch=SimpleNamespace(one_day=batch, seven_day=np.zeros(10000)))
        path = tmp_path/f'{identifier}.pkl.gz'
        with gzip.open(path, 'wb') as handle:
            pickle.dump(artifact, handle)
        entries.append({'update': identifier, 'path': path.name})
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'status':'passed', 'identity':{'training_contract_sha256':'data-contract'}, 'entries':entries}))
    pool = PreparedOneDayPool(manifest, cache_size=2)
    selected = pool.batch([(20,2),(10,0),(20,0)])
    np.testing.assert_array_equal(selected.target, [22,10,20])
    np.testing.assert_array_equal(selected.inputs.continuous_state, [[24,25],[10,11],[20,21]])
    assert pool.loads == 2
    pool.batch([(10,1),(20,1)])
    assert pool.loads == 2
    pool.load(30)
    assert len(pool.cache) == 2
    assert pool.statistics()['cached_one_day_bytes'] == 2*(6+3)*8
