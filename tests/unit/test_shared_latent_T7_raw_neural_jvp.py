"""Analytic prefix composition, JVP isolation and diagnostic evidence contracts."""
import json
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.hpc import shared_latent_T7_raw_neural_jvp as raw


def fixture_prefix(log=None):
    log = [] if log is None else log
    fields = (SimpleNamespace(name='tensor',shape=(3,)), SimpleNamespace(name='scalar',shape=()))
    head = SimpleNamespace(contract=object(), learned_schema=SimpleNamespace(fields=fields),
        layout=object(), input_normalization=object(), shared_config=object(),carbon_state_layout=object())
    def rebuild(template,state,discrete,contract):
        log.append('rebuild')
        assert contract is head.contract
        return state,discrete,template
    def assemble(value):
        log.append('assemble')
        return value
    def context(parameters,inputs):
        log.append('context')
        return parameters['scale']
    def provider(parameters,context,inputs,**kwargs):
        log.append('raw')
        assert kwargs == dict(schema=head.learned_schema,layout=head.layout,
            input_normalization=head.input_normalization,config=head.shared_config,
            carbon_state_layout=head.carbon_state_layout)
        state,discrete,template = inputs
        values = context*state.astype(jnp.float32)+template['forcing'].astype(jnp.float32)
        return {'tensor':values,'scalar':jnp.sum(values)}, {'tensor':template['mask'],'scalar':jnp.asarray(True)}
    head.input_assembler = SimpleNamespace(assemble=assemble)
    head.assemble_context = context
    head.learned_outputs = lambda *args: pytest.fail('physical outputs must not be called')
    return raw.make_raw_prefix(head,rebuild=rebuild,raw_outputs=provider),head,provider


def inputs():
    jax.config.update('jax_enable_x64',True)
    states = jnp.arange(12,dtype=jnp.float64).reshape(4,3)/10
    discretes = jnp.arange(4,dtype=jnp.int32)
    days = ({'forcing':jnp.ones((4,3),jnp.float64),
             'mask':jnp.asarray([[True,False,True]]*4)},jnp.zeros((4,),jnp.int32))
    direction = jnp.asarray([0.,1.,0.],jnp.float64)
    directions = jnp.zeros_like(states).at[3].set(direction)
    return states,discretes,days,direction,directions


def results():
    prefix,_,_ = fixture_prefix()
    b1,b4 = raw.route_functions(jax,prefix)
    states,discretes,days,direction,directions = inputs()
    parameters = {'scale':jnp.asarray(.5,jnp.float32)}
    single_day = jax.tree_util.tree_map(lambda x:x[3],days)
    one = jax.jit(b1)(parameters,states[3],discretes[3],single_day,direction)
    many = jax.jit(b4)(parameters,states,discretes,days,directions)
    return one,many


def test_prefix_orders_source_composition_and_uses_same_patched_global(monkeypatch):
    log = []
    prefix,head,provider = fixture_prefix(log)
    from research.daily_coarse_graining.daily_operator import shared_process_head
    monkeypatch.setattr(shared_process_head,'shared_transformer_raw_outputs',provider)
    # Exercise the production provider selection while keeping rebuild analytic.
    selected = raw.make_raw_prefix(head,rebuild=lambda t,s,d,c:(s,d,t))
    states,discretes,days,_,_ = inputs()
    day = jax.tree_util.tree_map(lambda x:x[3],days)
    prefix({'scale':jnp.asarray(.5,jnp.float32)},states[3],discretes[3],day)
    assert log == ['rebuild','assemble','context','raw']
    log.clear()
    selected({'scale':jnp.asarray(.5,jnp.float32)},states[3],discretes[3],day)
    assert log == ['assemble','context','raw']


def test_vmapped_jvp_preserves_natural_scale_and_every_masked_raw_coordinate():
    one,many = results()
    report = raw.compare_results(jax,one,many,sample_index=3)
    assert report['passed'] and report['exact_match']
    np.testing.assert_array_equal(one.tangent['tensor'],np.asarray([0.,.5,0.],np.float32))
    np.testing.assert_array_equal(many.tangent['tensor'][:3],np.zeros((3,3),np.float32))
    assert one.tangent['scalar'] == .5
    assert one.primal['tensor'].shape == (3,) and not one.auxiliary['tensor'][1]
    assert one.primal['tensor'][1] != 0  # inactive raw data is retained, not hidden
    assert not hasattr(one,'next_continuous')
    isolation=raw.other_sample_tangent_isolation(jax,many)
    assert isolation['passed'] and isolation['nonzero_coordinates']==0
    assert all(row['checked_coordinates']>0 for row in isolation['leaves'])


@pytest.mark.parametrize('value',[1e-12,float('nan')])
def test_other_sample_tangent_leakage_and_nonfinite_fail_exact_isolation(value):
    _,many=results()
    changed=many._replace(tangent={**many.tangent,'tensor':many.tangent['tensor'].at[0,1].set(value)})
    report=raw.other_sample_tangent_isolation(jax,changed)
    assert not report['passed']
    if np.isfinite(value):
        assert report['nonzero_coordinates']==1 and report['nonfinite_coordinates']==0
    else:
        assert report['nonzero_coordinates']==0 and report['nonfinite_coordinates']==1
    json.dumps(report,allow_nan=False)


def test_small_passing_difference_still_records_worst_values_and_exact_repeat_difference():
    one,_ = results()
    changed = one._replace(tangent={**one.tangent,'tensor':one.tangent['tensor'].at[0].set(jnp.float32(1e-7))})
    report = raw.compare_results(jax,changed,one)
    row = next(x for x in report['float_leaves'] if x['path']=="tangent['tensor']")
    assert report['passed'] and not report['exact_match']
    assert row['worst_position']==[0] and row['max_absolute_error']>0
    assert row['max_error_to_allowance']>0 and row['worst_actual_value']>0
    assert row['worst_reference_value']==0
    json.dumps(report,allow_nan=False)


def test_natural_tolerance_failure_is_not_hidden_by_dividing_by_four():
    result = raw.RawNeuralJVPResult({'field':np.zeros(2,np.float32)},
        {'field':np.zeros(2,np.float32)},{'field':np.ones(2,bool)})
    changed = result._replace(tangent={'field':np.asarray([0.,2e-6],np.float32)})
    report = raw.compare_results(jax,changed,result)
    assert not report['passed'] and report['failed_float_coordinates']==1
    assert raw.compare_results(jax,changed._replace(tangent={'field':changed.tangent['field']/4}),result)['passed']
    json.dumps(report,allow_nan=False)


def test_nonfinite_in_inactive_raw_output_fails_and_json_remains_strict():
    one,_ = results()
    changed = one._replace(primal={**one.primal,'tensor':one.primal['tensor'].at[1].set(jnp.nan)})
    report = raw.compare_results(jax,changed,changed)
    assert not report['passed'] and report['nonfinite_float_coordinates']==1
    row = next(x for x in report['float_leaves'] if x['path']=="primal['tensor']")
    assert row['worst_position']==[1] and row['worst_actual_value'] is None
    json.dumps(report,allow_nan=False)


def test_mask_mismatch_and_batch_axis_fail_closed():
    one,many = results()
    mismatch = one._replace(auxiliary={**one.auxiliary,'tensor':~one.auxiliary['tensor']})
    assert not raw.compare_results(jax,mismatch,one)['passed']
    missing_batch = many._replace(primal={'tensor':many.primal['tensor'][3],'scalar':many.primal['scalar']})
    with pytest.raises(ValueError,match='batch axis'):
        raw.compare_results(jax,one,missing_batch,sample_index=3)
    with pytest.raises(ValueError,match='dtype'):
        raw.compare_results(jax,one._replace(tangent={'tensor':np.zeros(3,np.float64),'scalar':one.tangent['scalar']}),one)


def test_bad_state_direction_dtype_is_rejected_before_prefix():
    prefix,_,_ = fixture_prefix()
    b1,_ = raw.route_functions(jax,prefix)
    states,discretes,days,direction,_ = inputs()
    with pytest.raises(ValueError,match='FP64'):
        b1({},states[3],discretes[3],days,direction.astype(jnp.float32))


@pytest.mark.parametrize('bad', ['missing_field','nonboolean_mask','reduced_field_shape'])
def test_raw_provider_contract_cannot_drop_fields_masks_or_full_shape(bad):
    _,head,_ = fixture_prefix()
    def provider(*args,**kwargs):
        values = {'tensor':jnp.ones(3,jnp.float32),'scalar':jnp.asarray(1.,jnp.float32)}
        active = {'tensor':jnp.ones(3,bool),'scalar':jnp.asarray(True)}
        if bad=='missing_field':
            del values['scalar']
        elif bad=='nonboolean_mask':
            active['tensor']=jnp.ones(3,jnp.int32)
        else:
            values['tensor'],active['tensor']=jnp.ones(2,jnp.float32),jnp.ones(2,bool)
        return values,active
    prefix = raw.make_raw_prefix(head,rebuild=lambda *args:None,raw_outputs=provider)
    # analytic assembler/context tolerate None; no production model is called
    head.input_assembler = SimpleNamespace(assemble=lambda _:None)
    with pytest.raises(ValueError):
        prefix({'scale':jnp.asarray(.5,jnp.float32)},None,None,(None,None))
