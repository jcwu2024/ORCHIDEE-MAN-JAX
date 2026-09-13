"""Select additional immutable Teacher rows; never execute the Teacher."""
from __future__ import annotations

import numpy as np


def expansion_plan(catalogue, references, windows, *, seed=20260913, rows_per_point_year=16):
    """Add stratified dates at each existing point-year, preserving exclusions.

    Retained daily inputs are reused from an exact matching prepared point-year.
    The existing one-day pool stays available; every selected new date is absent
    from both that pool and the fixed development panels.
    """
    if rows_per_point_year < 1:
        raise ValueError('rows_per_point_year must be positive')
    lookup = {}
    for ref in references:
        parent = ref.parent
        if (parent.spatial_split, parent.temporal_split) != ('train', 'train'):
            raise ValueError('expansion only accepts train/train references')
        lookup[(str(parent.landpoint_id), int(parent.year))] = ref
    monitor = {(str(row['landpoint_id']), int(row['year']), int(row['start_row']))
               for row in catalogue['monitor_rows']}
    old, sources = set(), {}
    for entry in sorted(catalogue['one_day_entries'], key=lambda e: int(e['update'])):
        offset = 0
        for record in entry['components']['one_day']:
            key = (str(record['landpoint_id']), int(record['year']))
            ref = lookup[key]
            if (record['parent_shard_sha256'], record['typed_shard_sha256']) != (ref.parent.sha256, ref.sha256):
                raise ValueError('prepared and raw Teacher point-year identities differ')
            for local_row, raw_row in enumerate(record['start_rows']):
                if (*key, int(raw_row)) not in monitor:
                    sources.setdefault(key, {'artifact_id': int(entry['update']), 'batch_row': offset+local_row,
                                             'reference_row': int(raw_row)})
                    break
            old.update((*key, int(row)) for row in record['start_rows'])
            offset += len(record['start_rows'])
    if not monitor <= old:
        raise ValueError('monitor dates are not from the declared base catalogue')
    rollout = {tuple(key) for window in windows for key in window['teacher_day_keys']}
    if (old - monitor) & rollout:
        raise ValueError('existing training pool overlaps rollout development days')
    excluded = old | monitor | rollout
    base_by_point_year = {}
    excluded_by_point_year = {}
    for point, year, row in old - monitor:
        base_by_point_year.setdefault((point, year), []).append(row)
    for point, year, row in excluded:
        excluded_by_point_year.setdefault((point, year), set()).add(row)
    entries, new = [], set()
    for number, key in enumerate(sorted(sources), start=1):
        ref = lookup[key]
        candidates = np.array([row for row in range(1, int(ref.day_count))
                               if row not in excluded_by_point_year.get(key, set())], dtype=np.int64)
        if len(candidates) < rows_per_point_year:
            raise ValueError(f'insufficient new dates at {key}')
        # Equal-size strata of the remaining chronologically ordered dates.
        rng = np.random.default_rng(np.random.SeedSequence([seed, key[1], *key[0].encode('utf-8')]))
        rows = sorted(int(rng.choice(group)) for group in np.array_split(candidates, rows_per_point_year))
        new.update((*key, row) for row in rows)
        existing_rows = sorted(base_by_point_year.get(key, ()))
        entries.append({'id': number, 'landpoint_id': key[0], 'year': key[1],
                        'new_start_rows': rows, 'start_rows': sorted(existing_rows + rows),
                        'day_count': int(ref.day_count),
                        'parent_shard_sha256': ref.parent.sha256, 'typed_shard_sha256': ref.sha256,
                        'retained_source': sources[key]})
    base = old - monitor
    eligible = sum(int(ref.day_count) - 1 for ref in lookup.values())
    return {'schema_version': 'one_day_expansion_plan_v1', 'status': 'planned',
            'seed': seed, 'rows_per_point_year': rows_per_point_year,
            'source_manifest': catalogue['source_manifest'], 'source_identity': catalogue['source_identity'],
            'entries': entries, 'counts': {
                'base_unique_train_days': len(base), 'additional_unique_train_days': len(new),
                'combined_unique_train_days': len(base | new), 'point_years': len(entries),
                'landpoints': len({key[0] for key in sources}),
                'years': sorted({key[1] for key in sources}), 'eligible_train_days': eligible,
                'combined_eligible_fraction': len(base | new) / eligible,
                'monitor_excluded': len(monitor), 'rollout_excluded': len(rollout)},
            'checks': {'new_disjoint_from_base_and_panels': not bool(new & excluded),
                       'unique_new_rows': len(new) == len(entries) * rows_per_point_year},
            'teacher_regenerated': False, 'sealed_test_accessed': False, 'training_updates': 0,
            'preparation_complete': False}
