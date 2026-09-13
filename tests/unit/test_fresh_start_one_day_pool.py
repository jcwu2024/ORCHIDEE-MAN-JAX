import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.hpc.run_fresh_start_one_day_pool import (
    feedback_loss_plan,
    hard_pass,
    pool_plan,
    retained_checkpoint_path,
    retained_checkpoint_updates,
    sampling_identity,
    validate_recovery,
)


def test_pool_plan_excludes_same_teacher_day_in_every_artifact_and_resumes():
    def entry(identifier, starts):
        return {'update':identifier, 'runtime_year':2000,
                'components':{'one_day':[{'landpoint_id':'site','year':2000,'start_rows':starts}]}}

    catalogue = {'one_day_entries':[entry(1,[0,1,2]),entry(2,[0,3,4])],
                 'monitor_rows':[{'artifact_id':1,'batch_row':0,'landpoint_id':'site','year':2000,'start_row':0}]}
    config = {'sampling':{'seed':5,'batch_size':3,'microbatch_size':2,'max_artifacts_per_batch':2}}
    sampler, panels = pool_plan(catalogue,config)
    batches = list(sampler.iter_epoch(0))
    assert [batch.draw_count for batch in batches] == [3,1]
    assert sampler.draw_count == 4
    assert all(row.start_row != 0 for batch in batches for row in batch.rows)
    assert panels['training'] == [row.as_metadata() for row in batches[0].rows]
    assert panels['excluded_days'] == catalogue['monitor_rows']
    assert list(sampler.resume(batches[0].next_cursor)) == batches[1:]
    saved = {'cursor':{'epoch':0,'batch_offset':1},'completed_updates':1,
             'optimizer_state':SimpleNamespace(step=1)}
    validate_recovery(saved,sampler,1)
    with pytest.raises(ValueError,match='disagree'):
        validate_recovery({**saved,'completed_updates':2},sampler,1)
    with pytest.raises(ValueError,match='outside'):
        validate_recovery({**saved,'cursor':{'epoch':0,'batch_offset':3}},sampler,1)
    changed = {**catalogue,'one_day_entries':[entry(1,[0,2,1]),entry(2,[0,3,4])]}
    assert sampling_identity(changed) != sampling_identity(catalogue)
    assert sampling_identity({**catalogue,'one_day_entries':list(reversed(catalogue['one_day_entries']))}) == sampling_identity(catalogue)


def test_hard_pass_requires_every_sample_to_pass():
    assert hard_pass({'finite':np.array([0,0]),'physical':np.array([0,0])})
    assert not hard_pass({'finite':np.array([0,1]),'physical':np.array([0,0])})


def test_frozen_method_keeps_existing_v3_loss_and_full_network_initialization():
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root/'manifests/coarse_graining/fresh_start_one_day_pool_v1.json').read_text())
    model = json.loads((root/config['model_protocol']).read_text())
    assert model['parameter_count'] == 109464375
    assert config['trainable_scope'] == model['trainable_scope'] == 'all_neural_parameters'
    assert config['sampling']['batch_size'] == 32
    assert config['sampling']['microbatch_size'] == 4
    assert model['loss'] == {'independent_daily_flux_mse':1.,'one_day_state_anchor':.1,'necessary_process_supervision':.01}
    assert not config['execution']['paid_submission_authorized']
    assert feedback_loss_plan(model) is None
    assert retained_checkpoint_updates(config) == ()


def test_feedback_and_retained_checkpoint_extensions_fail_closed():
    protocol = {
        'loss': {'litterpart_feedback_mse': 2.5e-5},
        'canopy_carbon_stock_change': {'weight': 3e-9, 'scale_source_pft_index': 13},
    }
    assert feedback_loss_plan(protocol)['coefficients'] == {
        'base': 1., 'canopy_stock_change': 3e-9, 'litterpart_feedback': 2.5e-5}
    with pytest.raises(ValueError,match='enabled together'):
        feedback_loss_plan({'loss': {'litterpart_feedback_mse': 2.5e-5}})
    assert retained_checkpoint_updates(
        {'checkpoint': {'retained_updates': [100,500,1998]}}) == (100,500,1998)
    with pytest.raises(ValueError,match='unique increasing'):
        retained_checkpoint_updates({'checkpoint': {'retained_updates': [500,100]}})
    assert retained_checkpoint_path(Path('run.json'),100).name == 'run.update-000100.checkpoint.pkl'
