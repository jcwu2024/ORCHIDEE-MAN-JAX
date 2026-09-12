"""Exercise the real runner, AdamW, sampler and recovery with a tiny daily model."""
import gzip
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import fresh_start_training, one_day_training_state, prepared_one_day
from scripts.hpc import run_fresh_start_one_day_pool as runner


class Inputs(NamedTuple):
    day_index: object


class Sample(NamedTuple):
    inputs: Inputs
    target: object


def test_full_loop_resume_matches_uninterrupted_update_and_sampling(tmp_path,monkeypatch):
    root = Path(__file__).resolve().parents[2]
    configuration = json.loads((root/'manifests/coarse_graining/fresh_start_one_day_pool_v1.json').read_text())
    model = json.loads((root/configuration['model_protocol']).read_text())
    model['parameter_count'] = 2
    model_path = tmp_path/'model.json'
    model_path.write_text(json.dumps(model))
    configuration['model_protocol'] = str(model_path)
    configuration['sampling'].update(batch_size=4,microbatch_size=2,max_artifacts_per_batch=1)
    configuration['monitoring']['updates'] = [0,1]
    configuration['checkpoint']['every_updates'] = 1
    config_path = tmp_path/'configuration.json'
    config_path.write_text(json.dumps(configuration))
    source = tmp_path/'source.json'
    source.write_text('{}')
    catalogue_path = tmp_path/'catalogue.json.gz'
    identity = {'data':'test'}
    catalogue = {'source_manifest':str(source),'source_identity':identity,
                 'one_day_entries':[{'update':1,'runtime_year':2000,'components':{'one_day':[
                     {'landpoint_id':'a','year':2000,'start_rows':list(range(9))}]}}],
                 'monitor_rows':[{'artifact_id':1,'batch_row':8,'landpoint_id':'a','year':2000,'start_row':8}]}
    with gzip.open(catalogue_path,'wt') as handle:
        json.dump(catalogue,handle)
    catalogue_report = tmp_path/'catalogue_report.json'
    catalogue_report.write_text(json.dumps({'status':'passed','catalogue':str(catalogue_path)}))

    def bootstrap(args,protocol):
        return ({'jax':jax,'devices':jax.devices()},None,None,None,
                {'w':jnp.array(.1,dtype=jnp.float32),'b':jnp.array(.2,dtype=jnp.float32)},None,None)

    def objective(p,sample):
        prediction = p['w']*sample.inputs.day_index+p['b']
        loss = (prediction-sample.target)**2
        field = {'predicted':prediction[None], 'teacher':sample.target[None],
                 'defined':jnp.ones((1,),dtype=bool)}
        return loss,{'hard':{'nonfinite':jnp.sum(~jnp.isfinite(prediction))},
                     'flux_loss':loss,'state_loss':loss,'process_loss':loss,
                     'physical':{'fluxes':{'gpp':field},'stocks':{}}}

    class Pool:
        def __init__(self,*args,**kwargs):
            self.identity = identity

        def batch(self,rows):
            x = np.array([r[1]+1 for r in rows],dtype=np.float32)
            return Sample(Inputs(x),2*x)

        def statistics(self):
            return {}

    monkeypatch.setattr(runner.prototype,'bootstrap',bootstrap)
    monkeypatch.setattr(fresh_start_training,'make_objective',lambda *a,**k:objective)
    monkeypatch.setattr(prepared_one_day,'PreparedOneDayPool',Pool)

    def args(output,resume=None):
        return SimpleNamespace(pool_protocol=config_path,catalogue_report=catalogue_report,output=output,
                               resume=resume,smoke_updates=None,shape_only=False,admission_only=False,cpu_forward_only=False)

    straight_path = tmp_path/'straight.json'
    straight = runner.run(args(straight_path))
    assert straight['status'] == 'completed'
    assert straight['completed_updates'] == 2
    assert straight['consumed_row_draws'] == 8
    assert straight['consumed_unique_teacher_days'] == 8
    assert [m['update'] for m in straight['monitors']] == [0,1,2]
    with straight_path.with_suffix('.checkpoint.pkl').open('rb') as handle:
        expected = pickle.load(handle)

    actual_save = one_day_training_state.save_training_checkpoint

    def interrupt_after_first(*a,**kw):
        result = actual_save(*a,**kw)
        if kw['completed_updates'] == 1:
            raise RuntimeError('simulated interruption after durable update')
        return result

    interrupted_path = tmp_path/'interrupted.json'
    monkeypatch.setattr(one_day_training_state,'save_training_checkpoint',interrupt_after_first)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        runner.run(args(interrupted_path))
    monkeypatch.setattr(one_day_training_state,'save_training_checkpoint',actual_save)
    resumed = runner.run(args(interrupted_path,interrupted_path.with_suffix('.checkpoint.pkl')))
    assert resumed['completed_updates'] == 2
    assert [r['batch_index'] for r in resumed['updates']] == [0,1]
    assert [m['update'] for m in resumed['monitors']] == [0,1,2]
    with interrupted_path.with_suffix('.checkpoint.pkl').open('rb') as handle:
        actual = pickle.load(handle)
    assert actual['cursor'] == expected['cursor'] == {'epoch':0,'batch_offset':2}
    for left,right in zip(jax.tree_util.tree_leaves((actual['parameters'],actual['optimizer_state'])),
                          jax.tree_util.tree_leaves((expected['parameters'],expected['optimizer_state'])),strict=True):
        np.testing.assert_array_equal(left,right)
