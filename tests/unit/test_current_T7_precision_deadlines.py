"""No model execution: controlled clocks and short timeout subprocesses."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
from scripts.hpc import probe_shared_current_T7_precision_admission as probe
from scripts.hpc import run_shared_current_T7_precision_admission as runner


class Expired(Exception):
    pass


class Timer:
    intervals = []
    def __init__(self, seconds, action):
        self.seconds = seconds
        self.action = action
        self.intervals.append(seconds)
    def start(self):
        pass
    def cancel(self):
        pass


class Base:
    def __init__(self, output, seconds):
        self.output = output
        self.started = probe.time.monotonic()
        self.report = {'stage_events': [], 'results': []}
        self.timer = Timer(seconds, lambda: None)
    def expire(self, phase):
        self.report.update(status='incomplete_timeout', deadline_phase=phase)
        self.emit('deadline_exceeded', phase=phase)
        raise Expired(phase)


@pytest.fixture
def controlled(monkeypatch, tmp_path):
    clock = [0.]
    Timer.intervals = []
    monkeypatch.setattr(probe.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(probe.threading, 'Timer', Timer)
    journal = probe.strict_journal_class(Base)(tmp_path/'result.json', 4980)
    return clock, journal


def test_lower_over_300_compile_only_gets_shared_remainder(controlled):
    clock, j = controlled
    clock[0] = 300  # preparation, imports already subtracted from outer wall
    j.begin_graph()
    end = j.graph_deadline
    with pytest.raises(RuntimeError, match='cannot be renewed'):
        j.begin_graph()
    with j.deadline(probe.CAPS['graph'], 'lower'):
        clock[0] += 700
    with j.deadline(probe.CAPS['graph'], 'compile'):
        assert Timer.intervals[-1] == 2600
        assert j.graph_deadline == end
        clock[0] += 2500
    assert clock[0] == 3500
    assert j.work_deadline - j.graph_deadline == probe.CAPS['calls']
    with j.deadline(probe.CAPS['calls'], 'native_and_fd_calls'):
        assert Timer.intervals[-1] == 1380


@pytest.mark.parametrize('phase', ['lower', 'compile'])
def test_graph_exhaustion_preserves_partial_result_and_phase(controlled, phase):
    clock, j = controlled
    j.report['results'] = [{'existing': 'preserved'}]
    j.begin_graph()
    if phase == 'compile':
        with j.deadline(probe.CAPS['graph'], 'lower'):
            clock[0] = 500
    with pytest.raises(Expired, match=phase):
        with j.deadline(probe.CAPS['graph'], phase):
            clock[0] = j.graph_deadline
    saved = json.loads(j.output.read_text())
    assert saved['deadline_phase'] == phase
    assert saved['results'] == [{'existing': 'preserved'}]


def test_late_graph_start_reserves_calls_and_cannot_enter_after_budget(controlled):
    clock, j = controlled
    clock[0] = 1000
    j.begin_graph()
    assert j.graph_deadline == 3600  # 4980 - 1380
    with j.deadline(3300, 'lower'):
        assert Timer.intervals[-1] == 2600
        clock[0] = 3599
    clock[0] = 3600
    with pytest.raises(Expired, match='compile'):
        with j.deadline(3300, 'compile'):
            pytest.fail('exhausted compile body must not run')


def test_calls_cannot_run_past_outer_work_end(controlled):
    clock, j = controlled
    clock[0] = 4900
    with pytest.raises(Expired, match='native_and_fd_calls'):
        with j.deadline(1380, 'native_and_fd_calls'):
            assert Timer.intervals[-1] == 80
            clock[0] = 4980


def test_import_startup_and_outer_phase_deadline_are_not_renewed():
    env = {'CURRENT_T7_SHELL_STARTED_EPOCH': '1000',
           'CURRENT_T7_PROBE_DEADLINE_EPOCH': '6100'}
    assert probe.remaining_probe_seconds(env, now=1200) == 4900
    assert probe.remaining_probe_seconds(env, now=1400) == 4700
    env['CURRENT_T7_PROBE_DEADLINE_EPOCH'] = '7000'
    assert probe.remaining_probe_seconds(env, now=1400) == 4820  # global5220
    assert probe.remaining_probe_seconds(env, now=6300) < 0
    assert runner.CAPS == probe.CAPS
    assert sum(probe.CAPS.values()) == probe.TOTAL_SECONDS == 5400


def test_base_initialization_does_not_renew_wall(controlled, tmp_path):
    clock, _ = controlled
    class SlowBase(Base):
        def __init__(self, *args):
            super().__init__(*args)
            clock[0] += 10
    j = probe.strict_journal_class(SlowBase)(tmp_path/'slow.json', 100)
    assert j.work_deadline == 100
    assert Timer.intervals[-1] == 90


@pytest.mark.parametrize('phase', ['lower', 'compile'])
def test_actual_timer_exits124_and_saves_partial_json(tmp_path, phase):
    output = tmp_path/'timeout.json'
    code = textwrap.dedent("""
        import os,time,threading
        from pathlib import Path
        from scripts.hpc import probe_shared_current_T7_precision_admission as p
        class Base:
            def __init__(self, path, seconds):
                self.output=path; self.started=time.monotonic()
                self.report={'stage_events': [], 'results': [{'partial': True}]}
                self.timer=threading.Timer(seconds, lambda: self.expire('wall'))
                self.timer.daemon=True; self.timer.start()
            def expire(self, phase):
                self.report.update(status='incomplete_timeout', deadline_phase=phase)
                try: self.emit('deadline_exceeded', phase=phase)
                finally: os._exit(124)
        j=p.strict_journal_class(Base)(Path(OUTPUT), 5)
        p.CAPS=dict(p.CAPS, graph=.2, calls=1)
        j.begin_graph()
        if PHASE == 'compile':
            with j.deadline(.2, 'lower'): time.sleep(.05)
        with j.deadline(.2, PHASE): time.sleep(2)
    """).replace('OUTPUT', repr(str(output))).replace('PHASE', repr(phase))
    r = subprocess.run([sys.executable, '-B', '-c', code], cwd=Path(__file__).resolve().parents[2],
                       capture_output=True, text=True, timeout=15,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES=''))
    assert r.returncode == 124, r.stderr
    saved = json.loads(output.read_text())
    assert saved['deadline_phase'] == phase
    assert saved['results'] == [{'partial': True}]


def test_fresh_approval_shared_budget_and_exact_args(monkeypatch, tmp_path):
    from types import SimpleNamespace
    argv = ['--stage', 'execute']
    args = SimpleNamespace(approval_report=tmp_path/'approval.json', output=tmp_path/'result.json')
    approval = dict(schema_version='current_T7_precision_approval_v2', approved=True,
        approval_origin='explicit_human_user', gpu_count=1, gpu_model='V100', slurm_cpus=4,
        nonexclusive=True, wall_seconds=5400, memory_limit_gib=24, model_compile_cap=1,
        retry=False, worst_case_cost_cny=3.72, working_directory=str(probe.ROOT),
        entry_script=str(Path(probe.__file__).resolve()), entry_sha256=probe.sha(probe.__file__),
        output=str(args.output), stage_caps=probe.CAPS, argv=argv, approval_note='test only')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
    def write():
        args.approval_report.write_text(json.dumps(approval))
    write()
    assert probe.check_approval(args, argv)['stage_caps']['graph'] == 3300
    with pytest.raises(PermissionError, match='exact command'):
        probe.check_approval(args, argv+['different'])
    approval['stage_caps'] = dict(load=300, preflight=240, lower=300, compile=3000, calls=1380, reserve=180)
    write()
    with pytest.raises(PermissionError, match='stage_caps'):
        probe.check_approval(args, argv)
