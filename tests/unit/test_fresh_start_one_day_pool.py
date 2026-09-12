import json
from pathlib import Path

import numpy as np

from scripts.hpc.run_fresh_start_one_day_pool import hard_pass, pool_plan


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
