import json
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.forward_screen_metrics import (
    complete_metric_gate,
    physical_window,
    summarize_windows,
)
from research.daily_coarse_graining.daily_operator.long_horizon_objective import MaskedTrajectory
from scripts.hpc.probe_trained_checkpointed_grad import rademacher_direction
from scripts.hpc.trained_screen_common import WindowPool, json_safe, validate_selection


class PFT(NamedTuple):
    pft_fraction: object
    active_pft_mask: object


class Inputs(NamedTuple):
    continuous_state: object
    discrete_state: object
    day_index: object
    pft: object


def test_actual_window_index_and_per_window_retained_bundle_are_preserved():
    states = np.arange(2*7*3).reshape(2,7,3).astype(float)
    inputs = Inputs(states,{'flag':np.zeros((2,7),dtype=int)},
                    np.array([np.arange(11,18),np.arange(21,28)]),
                    PFT(np.ones((2,7,2))/2,np.ones((2,7,2),dtype=bool)))
    window = SimpleNamespace(inputs=inputs,day_inputs={'annual':np.arange(10).reshape(2,5)},
        expected_next_state=SimpleNamespace(other=states+1),
        expected_next_state_defined=SimpleNamespace(other=np.ones_like(states,dtype=bool)),
        expected_fast_gpp=np.ones((2,7,1,2)),expected_fast_gpp_defined=np.ones((2,7,1,2),dtype=bool),
        expected_next_discrete_state={'flag':np.zeros((2,7),dtype=int)})
    artifact = SimpleNamespace(batch=SimpleNamespace(seven_day=window),schedule_entry={
        'components':{'seven_day':[{'landpoint_id':'a','year':2000,'start_rows':[10,20]}]}})
    pool = object.__new__(WindowPool)
    pool.load = lambda identifier:artifact
    row = {'artifact_id':1,'component':'seven_day','window_index':1,'length':7,
           'landpoint_id':'a','year':2000,'start_row':20}
    initial,_,days,targets,_ = pool.case(jax,row)
    np.testing.assert_array_equal(initial,states[1,0])
    np.testing.assert_array_equal(days[1]['annual'],np.tile(np.arange(5,10),(7,1)))
    np.testing.assert_array_equal(targets['state'],states[1]+1)


def test_cumulative_flux_cancellation_and_biomass_changes_are_reported_separately():
    def trajectory(values):
        a = np.asarray(values,dtype=float)[:,None]
        return MaskedTrajectory(a,np.ones_like(a,dtype=bool))

    pflux,tflux = {'gpp':trajectory([2,0])},{'gpp':trajectory([1,1])}
    pstock,tstock = {'agb':trajectory([101,103])},{'agb':trajectory([102,105])}
    result = physical_window(pflux,tflux,pstock,tstock,{'agb':trajectory([100])})
    summary = summarize_windows([result])
    assert summary['fluxes']['gpp']['daily_gC_m2_day']['rmse'] == 1.
    assert summary['fluxes']['gpp']['cumulative_gC_m2']['rmse'] == 0.
    assert summary['stocks']['agb']['change_gC_m2']['bias'] == -2.
    np.testing.assert_array_equal(result['stocks']['agb']['predicted_change'],[3.])
    encoded = json.dumps(json_safe({'curves':np.array([1.,np.nan,np.inf]),'metrics':summary}),allow_nan=False)
    assert json.loads(encoded)['curves'] == [1.,None,None]


def test_rademacher_direction_is_seeded_exact_and_respects_parameter_tree():
    parameters = {'w':jnp.ones((8,4),dtype=jnp.float32),'bias':jnp.array(0.),'absent':None}
    a,count = rademacher_direction(jax,parameters,20260913)
    b,_ = rademacher_direction(jax,parameters,20260913)
    c,_ = rademacher_direction(jax,parameters,20260914)
    assert count == 33 and a['absent'] is None
    np.testing.assert_array_equal(a['w'],b['w'])
    assert set(np.unique(a['w'])) == {-1.,1.}
    assert not np.array_equal(a['w'],c['w'])


def test_truncated_or_overlapping_panels_are_not_accepted():
    rows = []
    for h in (7,30):
        for i in range(8):
            start = 10 if h==7 else 100
            rows.append({'length':h,'component':{7:'seven_day',30:'thirty_day'}[h],
                'artifact_id':i,'landpoint_id':f'p{i}','year':2000,'start_row':start,
                'teacher_day_keys':[(f'p{i}',2000,d) for d in range(start,start+h)]})
    validate_selection({'windows':rows})
    with pytest.raises(ValueError):
        validate_selection({'windows':rows[:-1]})
    with pytest.raises(ValueError):
        validate_selection({'windows':rows[:8]+rows[:8]})


def test_metric_gate_rejects_filtered_nonfinite_or_missing_population():
    p = {'fluxes':{f'f{i}':{'cumulative_gC_m2':{'count':8,'nonfinite_defined':0},
                          'daily_gC_m2_day':{'count':56,'nonfinite_defined':0}} for i in range(8)},
         'stocks':{f's{i}':{'final':{'count':8,'nonfinite_defined':0},
                            'change':{'count':8,'nonfinite_defined':0}} for i in range(5)}}
    science = {'keys':{str(i):{'evaluated_values':8} for i in range(13)}}
    assert complete_metric_gate(p,science,7,8)
    p['fluxes']['f0']['daily_gC_m2_day'].update(count=55,nonfinite_defined=1)
    assert not complete_metric_gate(p,science,7,8)
    p['fluxes']['f0']['daily_gC_m2_day'].update(count=56,nonfinite_defined=0)
    science['keys']['0']['evaluated_values'] = 7
    assert not complete_metric_gate(p,science,7,8)
