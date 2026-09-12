"""Bounded host cache of one-day views from accepted prepared-v6 packages."""
from __future__ import annotations

import gzip
import json
import pickle
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


class PreparedOneDayPool:
    def __init__(self, manifest_path, *, cache_size=2):
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text())
        if manifest['status'] != 'passed' or cache_size < 1:
            raise ValueError('one-day pool requires an accepted manifest and positive cache size')
        self.records = {int(record['update']): record for record in manifest['entries']}
        if len(self.records) != len(manifest['entries']):
            raise ValueError('duplicate prepared artifact identity')
        self.identity = manifest['identity']
        self.cache_size, self.cache = cache_size, OrderedDict()
        self.loads, self.read_seconds, self.compressed_bytes_read = 0, 0., 0

    def load(self, artifact_id):
        if artifact_id in self.cache:
            self.cache.move_to_end(artifact_id)
            return self.cache[artifact_id]
        record = self.records[artifact_id]
        path = (self.manifest_path.parent/record['path']).resolve()
        path.relative_to(self.manifest_path.parent)
        started = time.perf_counter()
        with gzip.open(path, 'rb') as handle:
            artifact = pickle.load(handle)
        if int(artifact.schedule_entry['update']) != artifact_id:
            raise ValueError('prepared artifact identity differs from manifest')
        if artifact.training_contract_sha256 != self.identity['training_contract_sha256']:
            raise ValueError('prepared artifact uses a different data contract')
        value = (artifact.batch.one_day, artifact.schedule_entry)
        self.read_seconds += time.perf_counter()-started
        self.loads += 1
        self.compressed_bytes_read += path.stat().st_size
        self.cache[artifact_id] = value
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def batch(self, rows):
        """Stack explicit (artifact_id,row_index) identities without resampling."""
        import jax
        rows = tuple(rows)
        if not rows:
            raise ValueError('cannot materialize an empty one-day batch')
        batches = {identifier: self.load(identifier)[0] for identifier in dict.fromkeys(row[0] for row in rows)}
        selected = []
        for identifier, index in rows:
            batch = batches[identifier]
            if not 0 <= index < len(batch.inputs.continuous_state):
                raise IndexError('one-day row index is outside its artifact')
            selected.append(jax.tree_util.tree_map(lambda value: np.asarray(value)[index], batch))
        return jax.tree_util.tree_map(lambda *values: np.stack(values), *selected)

    def statistics(self):
        import jax
        cached_bytes = sum(np.asarray(leaf).nbytes for batch, _ in self.cache.values()
                           for leaf in jax.tree_util.tree_leaves(batch))
        return {'artifact_loads': self.loads, 'read_seconds': self.read_seconds,
                'compressed_bytes_read': self.compressed_bytes_read,
                'cached_artifacts': list(self.cache), 'cached_one_day_bytes': cached_bytes}
