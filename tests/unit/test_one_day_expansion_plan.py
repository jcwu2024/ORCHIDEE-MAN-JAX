from types import SimpleNamespace as NS

import pytest

from research.daily_coarse_graining.daily_operator.one_day_expansion_plan import expansion_plan


def inputs():
    ref = NS(parent=NS(landpoint_id='p1', year=2000, sha256='parent',
                       spatial_split='train', temporal_split='train'), sha256='typed', day_count=366)
    record = {'landpoint_id': 'p1', 'year': 2000, 'parent_shard_sha256': 'parent',
              'typed_shard_sha256': 'typed', 'start_rows': [3, 5, 8]}
    catalogue = {'source_manifest': 'prepared.json', 'source_identity': {},
                 'one_day_entries': [{'update': 9, 'components': {'one_day': [record]}}],
                 'monitor_rows': [{'landpoint_id': 'p1', 'year': 2000, 'start_row': 5}]}
    return catalogue, [ref], [{'teacher_day_keys': [('p1', 2000, r) for r in range(20, 50)]}]


def test_adds_new_dates_and_preserves_exact_retained_source():
    args = inputs()
    plan = expansion_plan(*args)
    assert plan == expansion_plan(*args)
    rows = plan['entries'][0]['new_start_rows']
    assert len(set(rows)) == 16
    assert not set(rows) & ({0, 3, 5, 8} | set(range(20, 50)))
    assert min(rows) < 80 and max(rows) > 300
    assert plan['entries'][0]['retained_source'] == {'artifact_id': 9, 'batch_row': 0, 'reference_row': 3}
    assert plan['counts']['combined_unique_train_days'] == 18
    assert {3, 8}.issubset(plan['entries'][0]['start_rows'])
    assert 5 not in plan['entries'][0]['start_rows']
    assert all(plan['checks'].values())


def test_never_admits_nontrain_or_overlapping_rollout():
    catalogue, refs, windows = inputs()
    refs[0].parent.temporal_split = 'test'
    with pytest.raises(ValueError, match='train/train'):
        expansion_plan(catalogue, refs, windows)
    refs[0].parent.temporal_split = 'train'
    windows[0]['teacher_day_keys'].append(('p1', 2000, 3))
    with pytest.raises(ValueError, match='overlaps'):
        expansion_plan(catalogue, refs, windows)
