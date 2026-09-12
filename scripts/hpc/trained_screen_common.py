"""One trained-head and window-input boundary shared by forward and AD probes."""
import gzip
import json
import pickle
from collections import OrderedDict

import numpy as np

from scripts.hpc import run_fresh_start_one_day as training


def json_safe(value):
    if isinstance(value,dict):
        return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value,'dtype'):
        return json_safe(np.asarray(value).tolist())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def load_model(args):
    report = json.loads(args.training_report.read_text())
    if report['status'] != 'completed' or report['completed_updates'] != 1998:
        raise ValueError('screen requires the declared complete update-1998 run')
    runtime,head,template,scales,_,_,_ = training.bootstrap(args,report['model_protocol'])
    from research.daily_coarse_graining.daily_operator.one_day_training_state import load_training_checkpoint

    saved = load_training_checkpoint(args.checkpoint,expected_method=report['method'])
    if saved['completed_updates'] != 1998:
        raise ValueError('checkpoint is not the final declared update')
    parameters = runtime['jax'].device_put(saved['parameters'])
    if sum(v.size for v in runtime['jax'].tree_util.tree_leaves(parameters)) != report['model_protocol']['parameter_count']:
        raise ValueError('trained parameter tree differs from declared architecture')
    return runtime,head,template,scales,parameters,report['method']


class WindowPool:
    def __init__(self,selection_path,expected_source_identity=None):
        from pathlib import Path

        self.selection = json.loads(selection_path.read_text())
        if self.selection['status'] != 'passed':
            raise ValueError('window selection is not accepted')
        validate_selection(self.selection)
        if expected_source_identity is not None and self.selection['source_identity'] != expected_source_identity:
            raise ValueError('evaluation pool differs from the declared training source')
        source = Path(self.selection['source_manifest'])
        manifest = json.loads(source.read_text())
        if manifest['identity'] != self.selection['source_identity']:
            raise ValueError('selection and prepared source identities differ')
        self.root = source.parent
        self.records = {int(r['update']):r for r in manifest['entries']}
        self.cache = OrderedDict()

    def load(self,identifier):
        if identifier not in self.cache:
            path = (self.root/self.records[identifier]['path']).resolve()
            path.relative_to(self.root.resolve())
            with gzip.open(path,'rb') as handle:
                artifact = pickle.load(handle)
            if int(artifact.schedule_entry['update']) != identifier:
                raise ValueError('prepared artifact identity differs')
            self.cache[identifier] = artifact
            while len(self.cache)>2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(identifier)
        return self.cache[identifier]

    def case(self,jax,row):
        from research.daily_coarse_graining.daily_operator.canonical_rollout import prepared_retained_days

        artifact = self.load(row['artifact_id'])
        component = row['component']
        records = [(r,start) for r in artifact.schedule_entry['components'][component] for start in r['start_rows']]
        record,start = records[row['window_index']]
        if (record['landpoint_id'],record['year'],start) != (row['landpoint_id'],row['year'],row['start_row']):
            raise ValueError('metadata selection does not match the actual window row')
        window = getattr(artifact.batch,component)
        k,h = row['window_index'],row['length']
        templates = jax.tree_util.tree_map(lambda v:v[k,:h],window.inputs)
        if len(templates.continuous_state) != h or not np.all(np.diff(np.asarray(templates.day_index).reshape(-1))==1):
            raise ValueError('prepared window does not contain the declared consecutive days')
        retained = prepared_retained_days(window.day_inputs,k,h)
        initial = templates.continuous_state[0]
        discrete = jax.tree_util.tree_map(lambda v:v[0],templates.discrete_state)
        targets = {'state':window.expected_next_state.other[k,:h],
                   'defined':window.expected_next_state_defined.other[k,:h],
                   'gpp':window.expected_fast_gpp[k,:h],
                   'gpp_defined':window.expected_fast_gpp_defined[k,:h],
                   'fraction':templates.pft.pft_fraction,'active':templates.pft.active_pft_mask}
        expected_discrete = jax.tree_util.tree_map(lambda v:v[k,:h],window.expected_next_discrete_state)
        return initial,discrete,(templates,retained),targets,expected_discrete


def validate_selection(selection):
    rows = selection['windows']
    if len(rows)!=16 or {h:sum(r['length']==h for r in rows) for h in (7,30)}!={7:8,30:8}:
        raise ValueError('screen requires exactly eight windows of each horizon')
    used,artifacts = set(),set()
    for r in rows:
        if r['component'] != {7:'seven_day',30:'thirty_day'}[r['length']]:
            raise ValueError('component/horizon mismatch')
        days = [(r['landpoint_id'],r['year'],i) for i in range(r['start_row'],r['start_row']+r['length'])]
        if [tuple(v) for v in r['teacher_day_keys']] != days or used.intersection(days):
            raise ValueError('window day identities differ or overlap')
        key = (r['artifact_id'],r['length'])
        if key in artifacts:
            raise ValueError('more than one window per artifact/horizon')
        used.update(days)
        artifacts.add(key)
