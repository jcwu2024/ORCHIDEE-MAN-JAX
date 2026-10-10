"""Preparation scope, false approval, zero model calls and truthful raw receipts."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.hpc import probe_shared_latent_T7_raw_neural_jvp as probe
from scripts.hpc import run_shared_latent_T7_raw_neural_jvp as launcher


def arguments(tmp_path):
    command = launcher.child_command(launcher.parser().parse_args(['--stage','execute','--plan-only']))
    args = probe.parser().parse_args(command[3:])
    args.approval_report=tmp_path/'approval.json'
    proposal=probe.proposal_facts(probe.DIAG/(probe.PREFIX+'.result.json'))
    proposal.update(approved=False,approval_origin=None,approval_note='')
    args.approval_report.write_text(json.dumps(proposal))
    return args,proposal


def test_distinct_raw_prefix_scope_caps_and_preflight_paths(tmp_path):
    args,_=arguments(tmp_path)
    probe.validate_caps(args)
    assert launcher.PREFIX=='T7-raw-neural-jvp-20261010'
    assert args.output.name==launcher.PREFIX+'.result.json'
    assert args.runtime_preflight.name==launcher.PREFIX+'.runtime-preflight.json'
    assert Path(launcher.preflight_command()[2])==launcher.CODE/'scripts/hpc/check_daily_training_runtime.py'
    assert args.backend_seconds==1800 and args.route_seconds==2100
    assert args.wall_seconds==4800 and args.gpu_seconds_cap==5400
    assert args.nvml_memory_log.name.endswith(launcher.PREFIX+'.memory.csv')


def test_false_approval_rejects_before_assets_or_backend(tmp_path,monkeypatch):
    args,_=arguments(tmp_path)
    args.plan_only=False
    monkeypatch.setattr(probe,'_assets',lambda *args:pytest.fail('no assets or model allowed'))
    with pytest.raises(PermissionError):
        probe.run(args)


@pytest.mark.parametrize('key,value', [('funding_scope','latent_T7_first_day_jvp_20261009'),
    ('gpu_count',True),('vjp_calls',1),('result_schema','fake_next_state'),('prior_funding_reuse',True)])
def test_new_scope_cannot_borrow_previous_funding_or_change_boundary(tmp_path,key,value):
    args,proposal=arguments(tmp_path)
    proposal[key]=value
    args.approval_report.write_text(json.dumps(proposal))
    with pytest.raises(ValueError):
        probe.validate_proposal(args,execution=False)


def test_plan_only_reads_no_model_and_preserves_truthful_result_schema(tmp_path,monkeypatch):
    args,_=arguments(tmp_path)
    monkeypatch.setattr(probe,'_assets',lambda *args:pytest.fail('no assets allowed'))
    report=probe.run(args)
    assert not report['GPU_executed'] and report['JVP_calls']==0 and report['planned_JVP_calls']==24
    assert report['raw_prefix_only'] and 'RawNeuralJVPResult' in report['result_schema']


@pytest.mark.parametrize('status',[1,3,124])
def test_launcher_preserves_actual_child_exit_without_running_preflight_in_plan(tmp_path,monkeypatch,status):
    monkeypatch.setattr(launcher,'CODE',tmp_path)
    monkeypatch.setattr(launcher,'OVERLAY',tmp_path)
    monkeypatch.setattr(launcher.os,'chdir',lambda *args:None)
    calls=[]
    monkeypatch.setattr(launcher.subprocess,'run',lambda command,**kwargs:(calls.append(command) or SimpleNamespace(returncode=status)))
    assert launcher.main(['--stage','execute','--plan-only'])==status
    assert len(calls)==1 and '--plan-only' in calls[0]


def test_preservation_binds_complete_raw_arrays_actual_direction_and_identity(tmp_path):
    captured=[]
    owner=SimpleNamespace(atomic_pickle=lambda path,value:captured.append(value))
    raw={'full_tensor':np.arange(8,dtype=np.float32)}
    direction=np.zeros((4,3854),np.float64);direction[3,434]=1
    probe.preserve_raw(owner,tmp_path/'result.pkl',raw,route='B4',cycle=1,coordinate=434,
        direction=direction,identity={'scope':probe.SCOPE})
    payload=captured[0]
    assert payload['result'] is raw and payload['scope']==probe.SCOPE
    np.testing.assert_array_equal(payload['direction'],direction)
    assert len(payload['identity_sha256'])==64 and payload['tangent_scale']=='natural_transition_no_div4'


def test_numerical_failure_remains_exit3(tmp_path,monkeypatch):
    args,_=arguments(tmp_path)
    command=launcher.child_command(launcher.parser().parse_args(['--stage','execute','--plan-only']))
    monkeypatch.setattr(probe,'run',lambda *args:{'status':'numerical_failed'})
    assert probe.main(command[3:])==3
