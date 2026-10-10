"""CPU native AD/public AdamW schedule checks and actual private pipes."""
from collections import namedtuple
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc import probe_shared_original_full_state_schedule as runner
from scripts.hpc import probe_shared_original_full_state_schedule_worker as worker_module
from scripts.hpc.shared_full_state_ad_execution import FullStateADEvaluator
from scripts.hpc.probe_shared_serialized_dual import RemoteDualExecutor
from scripts.hpc.probe_shared_dual import make_microbatch_daily_branch, make_recurrent_branch, make_component_combiner
from scripts.hpc.run_shared_learning import make_tree_operations
from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.target_oriented_training import initialize_target_oriented_adamw

Inputs = namedtuple("Inputs", "continuous_state")
Samples = namedtuple("Samples", "inputs")


def setup_chains(*, state_mismatch=False, hard_failure=False):
    device = jax.local_devices(backend="cpu")[0]
    parameters = {"weight": jnp.array([2.], dtype=jnp.float32)}
    optimizer = initialize_target_oriented_adamw(parameters)
    operations = make_tree_operations(jax, jnp, 32)
    update = make_update({"optimizer": {"learning_rate": .02, "clip_norm": 1., "beta1": .9,
        "beta2": .999, "epsilon": 1e-8, "weight_decay": 0.}})
    def daily(p, state, sample):
        loss = p["weight"][0]**2*jnp.mean(state[:, 0])
        return loss, {"hard": {"ok": jnp.int32(0)}, "flux": loss/3, "stock": loss/3}
    def recurrent(p, state, *case):
        loss = p["weight"][0]**2*jnp.mean(state)
        return loss, {"hard": {"ok": jnp.int32(0)}, "flux": loss/2, "stock": loss/2}
    def evaluators():
        return {name: FullStateADEvaluator(jax, jax.value_and_grad(
            daily if name == "daily" else recurrent, argnums=(0, 1), has_aux=True), name)
            for name in ("daily", "T7", "T30")}
    parent, remote = evaluators(), evaluators()
    daily_branch = runner.full_state_branch(make_microbatch_daily_branch(jax=jax, jnp=jnp,
        device=device, evaluator=parent["daily"], operations=operations, microbatch_size=4,
        expected_count=32, preserve_auxiliary=True), {"daily": parent["daily"]}, daily=True)
    def recurrent_branch(ev, horizons):
        return runner.full_state_branch(make_recurrent_branch(jax=jax, device=device,
            evaluators={h: ev[f"T{h}"] for h in horizons}, operations=operations,
            horizons=horizons, preserve_auxiliary=True), {f"T{h}": ev[f"T{h}"] for h in horizons})
    branches = {frozenset(hs): recurrent_branch(remote, hs) for hs in ((7, 30), (30,))}
    seen = []
    class Worker:
        def evaluate(self, jax_module, current, cases):
            command = {"kind": "evaluate", "parameters": jax_module.device_get(current),
                       "cases": jax_module.device_get(cases), "parameter_version": self.parameter_version}
            seen.append((frozenset(cases), self.parameter_version, command["parameters"]["weight"].copy(),
                         float(command["cases"][30][0][0])))
            response = worker_module.evaluate_command(jax_module, device, branches, command)
            if state_mismatch and set(cases) == {30}:
                response["result"][30]["initial_state_gradient"][0] += .01
            if hard_failure:
                response["result"][30]["hard"]["ok"] = 1
            self.last_parameter_version = response["executed_parameter_version"]
            self.last_device_memory = response["device_memory"]
            return response["result"], response["recurrent_seconds"], response["worker_total_seconds"]
    worker = Worker()
    kwargs = dict(jax=jax, parameters=parameters, optimizer=optimizer, daily_branch=daily_branch,
        worker=worker, combine=make_component_combiner(jax), update=update, finite=operations["finite"],
        preserve_components=True, audit_parameter_versions=True)
    old = RemoteDualExecutor(**kwargs)
    new = RemoteDualExecutor(**kwargs, local_recurrent_branch=recurrent_branch(parent, (7,)))
    samples = Samples(Inputs(jnp.broadcast_to(jnp.arange(1, 33, dtype=jnp.float32)[:, None], (32, 3854))))
    payloads = [{"daily": samples, "cases": {
        h: (jnp.ones(3854, dtype=jnp.float32)*h, *[jnp.ones(1)]*4) for h in (7, 30)}}]
    return old, new, payloads, seen


@pytest.mark.parametrize("mismatch", [False, True])
def test_public_old_new_current_chain_full_state_and_strict_failure_continues(monkeypatch, mismatch, tmp_path):
    old, new, payloads, seen = setup_chains(state_mismatch=mismatch)
    monkeypatch.setattr(runner, "require_memory", lambda *_: None)  # CPU has no GPU stats.
    saved, events = [], []
    def save_pair(index, values, comparison):
        path = tmp_path/f"pair-{index}.pkl"
        runner.atomic_pickle(path, runner.parameter_snapshot(jax, values, comparison))
        return str(path)
    try:
        rows, diagnostics, attempts, strict = runner.execute_pairs(jax=jax, old=old, new=new,
            payloads=payloads, artifact_ids=(12,), args=SimpleNamespace(updates=4, warmup_proposals=0),
            deadline=SimpleNamespace(may_start_update=lambda: True),
            emit=lambda stage, **fields: events.append((stage, fields)), save_states=lambda value: saved.append(len(value)),
            save_pair=save_pair)
        assert len(rows) == len(diagnostics) == 4 and attempts == {"old": 4, "new": 4}
        assert strict is (not mismatch)
        assert int(old.optimizer.step) == int(new.optimizer.step) == 4
        assert [x[1] for x in seen] == [0, 0, 1, 1, 2, 2, 3, 3]
        assert [x[0] for x in seen] == [frozenset((7, 30)), frozenset((30,)),
            frozenset((30,)), frozenset((7, 30))]*2
        assert not np.array_equal(seen[0][2], seen[-1][2])
        for item in diagnostics:
            state = item["new"]["daily"]
            assert state["full_state_ad_calls"] == 8
            assert state["initial_state_gradient"].shape == (32, 3854)
            np.testing.assert_allclose(state["initial_state_gradient"],
                state["initial_state_gradients_micro4_mean"].reshape(32, 3854)/8)
        if mismatch:
            assert all(not row["strict_comparison"]["comparisons"]["T30"]["initial_state_gradient"]["passed"] for row in rows)
            assert all(row["strict_comparison"]["comparisons"]["parameters"]["passed"] for row in rows)
        summary = runner.timing_summary(rows)
        assert summary["minimum_three_pairs_complete"] and summary["post_first_pairs"]["paired_samples"] == 3
        assert saved == [1, 2, 3, 4]
        import pickle
        with Path(rows[0]["parameter_gradient_artifact"]).open("rb") as handle:
            retained = pickle.load(handle)
        np.testing.assert_allclose(retained["old"]["mixed_gradient"]["weight"], [214.])
        assert isinstance(retained["old"]["component_gradients"]["T7"]["weight"], np.ndarray)
        np.testing.assert_allclose(retained["old"]["component_gradients"]["T7"]["weight"], [28.])
    finally:
        old.close(); new.close()


def test_physical_failure_fatal(monkeypatch):
    old, new, payloads, _ = setup_chains(hard_failure=True)
    monkeypatch.setattr(runner, "require_memory", lambda *_: None)
    try:
        with pytest.raises(ValueError, match="physical counters"):
            runner.execute_pairs(jax=jax, old=old, new=new, payloads=payloads, artifact_ids=(1,),
                args=SimpleNamespace(updates=4, warmup_proposals=0),
                deadline=SimpleNamespace(may_start_update=lambda: True), emit=lambda *a, **kw: None,
                save_states=lambda _: None, save_pair=lambda *args: "cpu-test-pair.pkl")
        assert int(old.optimizer.step) == int(new.optimizer.step) == 0
    finally:
        old.close(); new.close()


def test_budget_stops_between_pair_and_accounts_warmup(monkeypatch):
    old, new, payloads, _ = setup_chains()
    monkeypatch.setattr(runner, "require_memory", lambda *_: None)
    decisions = iter([True, True, True, True, False])
    events = []
    try:
        rows, diagnostics, attempts, strict = runner.execute_pairs(jax=jax, old=old, new=new,
            payloads=payloads, artifact_ids=(1,), args=SimpleNamespace(updates=4, warmup_proposals=1),
            deadline=SimpleNamespace(may_start_update=lambda: next(decisions)),
            emit=lambda stage, **kw: events.append((stage, kw)), save_states=lambda _: None,
            save_pair=lambda *args: "partial-pair.pkl")
        assert rows == diagnostics == [] and strict
        assert attempts == {"old": 2, "new": 1}
        assert int(old.optimizer.step) == 1 and int(new.optimizer.step) == 0
        assert events[-1][0] == "partial_pair_budget_stop"
        assert not runner.timing_summary(rows)["minimum_three_pairs_complete"]
    finally:
        old.close(); new.close()


def test_actual_private_pipe_transfers_changed_cases_params_versions(tmp_path):
    with runner.worker_class()(tmp_path/"unused.json", cuda_device="1",
            parameters={"weight": np.ones(1)}, cases={7: (), 30: ()}, echo_test=True) as worker:
        for version, horizons in ((0, (7, 30)), (1, (30,))):
            worker.parameter_version = version
            result, *_ = worker.evaluate(jax, {"weight": np.array([1.+version])},
                {h: (np.array([h+version]),) for h in horizons})
            assert result["parameter_version"] == worker.last_parameter_version == version
            np.testing.assert_array_equal(result["parameters"]["weight"], [1.+version])
            assert set(result["cases"]) == set(horizons)
            for h in horizons: np.testing.assert_array_equal(result["cases"][h][0], [h+version])


@pytest.mark.parametrize("start_step", [0, 1])
def test_backend_precedes_initial_unpickle_and_both_binaries_loaded(monkeypatch, start_step):
    import scripts.hpc.probe_shared_dual as branches
    import scripts.hpc.run_shared_learning as learning
    events = []
    fake_jax = SimpleNamespace(config=SimpleNamespace(update=lambda *_: None), device_put=lambda value, *_: value)
    device = SimpleNamespace(platform="gpu", memory_stats=lambda: {"peak_bytes_in_use": 1024})
    def initialize():
        events.append("backend")
        return fake_jax, (device,)
    def load(*args):
        assert args[-1] == ("T7", "T30")
        events.append("both_loaded")
        return {"T7": object(), "T30": object()}, {"T7": {}, "T30": {}}
    parent = SimpleNamespace(load_adjacent=lambda _: SimpleNamespace(initialize_gpu_backend=initialize),
        load_overlays=lambda: events.append("overlays"),
        read_manifest=lambda _: {"resume": {"completed_pairs": start_step}} if start_step else {},
        read_reference=lambda _: ({}, {}), validate_original_assets=lambda *_: {}, load_evaluators=load,
        full_state_branch=lambda branch, _: branch, require_memory=lambda *_: None)
    monkeypatch.setattr(worker_module, "load_parent", lambda: parent)
    monkeypatch.setattr(learning, "make_tree_operations", lambda *_: {})
    built = []
    monkeypatch.setattr(branches, "make_recurrent_branch", lambda **kw: built.append(kw["horizons"]))
    commands = iter([{"kind": "initialize", "parameters": {}, "cases": {7: (), 30: ()}, "parameter_version": start_step}, {"kind": "close"}])
    def receive():
        assert events[:2] == ["backend", "overlays"]
        events.append("receive")
        return next(commands)
    sent = []
    worker_module.serve(SimpleNamespace(manifest=Path("unused.json")), receive, sent.append)
    assert built == [(7, 30), (30,)] and "both_loaded" in events
    assert [message["status"] for message in sent] == ["ready", "closed"]
    assert sent[0]["initial_parameter_version"] == start_step


@pytest.mark.parametrize("cases,version", [({7: ()}, 0), ({30: ()}, None), ({30: ()}, True)])
def test_worker_rejects_wrong_horizons_or_versions(cases, version):
    with pytest.raises(ValueError):
        worker_module.evaluate_command(None, None, {}, {"kind": "evaluate", "cases": cases, "parameter_version": version})


def fixture_manifest(tmp_path):
    root = Path(__file__).resolve().parents[2]/"outputs/diagnostics/shared-latent-reader-20261004/original-admissions"
    names = {"daily": "shared-chunk-vmap-daily-dual-v4-20261001-reference.json",
        "T7": "shared-recurrent-checkpoint-policy-20260930-v2-T7-reference.json",
        "T30": "shared-original-t30-dual-reference-20261001.json"}
    identity = json.loads((root/names["daily"]).read_text())["identity"]
    reference = {"status": "passed", "completed_updates": 2, "technical_updates": 2,
        "source_git_head": identity.get("source_head", identity.get("source_git_head")), "method": identity["method"]}
    ref = tmp_path/"reference.json"; ref.write_text(json.dumps(reference))
    profile = tmp_path/"profile.json"; profile.write_text(json.dumps({"status": "passed",
        "executables": {key: {} for key in names}, "technical_update_report": str(ref)}))
    manifest = {"schema_version": runner.SCHEMA, "source_profile": str(profile), "components": {}}
    for key, name in names.items():
        report = json.loads((root/name).read_text())
        executable = tmp_path/Path(report.get("reference_executable", report.get("executable"))).name
        executable.write_bytes(b"metadata-only test, must not deserialize")
        manifest["components"][key] = {"report": str(root/name), "executable": str(executable)}
    for key in ("input_bundle", "original_checkpoint"):
        file = tmp_path/(key+".pkl"); file.write_bytes(b"must not unpickle")
        manifest[key] = str(file)
    path = tmp_path/"manifest.json"; path.write_text(json.dumps(manifest))
    return path, manifest


def test_actual_cli_metadata_dry_run_without_jax_import_or_unpickling(tmp_path):
    path, _ = fixture_manifest(tmp_path)
    code = "import runpy,sys; runpy.run_path(sys.argv.pop(1),run_name='__main__')"
    # sitecustomize blocks even an accidental JAX import in this fresh process.
    (tmp_path/"sitecustomize.py").write_text("import sys\nclass Guard:\n def find_spec(self,name,*a):\n  if name == 'jax' or name.startswith('jax.'):\n   raise RuntimeError('metadata dryrun imported JAX')\nsys.meta_path.insert(0,Guard())\n")
    import os
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run([sys.executable, "-B", "-c", code, str(Path(runner.__file__)),
        "--manifest", str(path), "--output", str(tmp_path/"out.json"),
        "--parent-cuda-device", "0", "--worker-cuda-device", "1", "--dry-run"],
        capture_output=True, text=True, env=env, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "dry_run_passed" and report["GPU_executed"] is False
    assert not (tmp_path/"out.json").exists()


def test_original_manifest_and_full_admission_reject_other_routes(tmp_path):
    path, manifest = fixture_manifest(tmp_path)
    manifest["candidate_parameters"] = "forbidden"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="candidate"): runner.read_manifest(path)
    manifest.pop("candidate_parameters")
    report = json.loads(Path(manifest["components"]["T7"]["report"]).read_text())
    report.pop("parameter_and_initial_state_gradients", None)
    report["scope"] = "parameters_only"
    bad = tmp_path/"bad.json"; bad.write_text(json.dumps(report))
    manifest["components"]["T7"]["report"] = str(bad)
    _, reference = runner.read_reference(manifest)
    with pytest.raises(ValueError, match="full-state"): runner.validate_original_assets(manifest, reference)


@pytest.mark.parametrize("memory", [None, {}, {"peak_bytes_in_use": float("nan")}, {"peak_bytes_in_use": 25*2**30}])
def test_memory_unavailable_or_over_limit_fatal(memory):
    with pytest.raises(MemoryError): runner.require_memory(memory, "test")


def test_argument_caps_include_discarded_warmups(tmp_path):
    args = runner.parser().parse_args(["--manifest", "unused", "--output", str(tmp_path/"out.json"),
        "--parent-cuda-device", "0", "--worker-cuda-device", "1", "--updates", "16", "--warmup-proposals", "1"])
    with pytest.raises(ValueError, match="cap"): runner.validate_arguments(args)
    args.updates = 4; args.wall_seconds = 501
    with pytest.raises(ValueError, match="wall cap"): runner.validate_arguments(args)
    assert not runner.timing_summary([{"chain_step": i, "old": {"complete_seconds": 2.},
        "new": {"complete_seconds": 1.}} for i in (1, 2)])["minimum_three_pairs_complete"]


def test_deadline_preserves_strict_failure_and_kills_child(tmp_path):
    args = SimpleNamespace(output=tmp_path/"timeout.json", wall_seconds=480, save_reserve_seconds=30)
    exits, killed = [], []
    deadline = runner.ExecutionDeadline(args, started=0, clock=lambda: 480., exit_process=exits.append)
    waited = []
    deadline.register(SimpleNamespace(poll=lambda: None, kill=lambda: killed.append(True),
        wait=lambda timeout: waited.append(timeout)))
    import os
    owned = tmp_path/f".timeout.pair-001-gradients.pkl.{os.getpid()}.tmp"
    completed = tmp_path/"timeout.pair-001-gradients.pkl"
    unrelated = tmp_path/"prior-result.tmp"
    for path in (owned, completed, unrelated): path.write_bytes(b"preserve or clean")
    deadline.capture({"strict_parity_passed": False, "changing_pairs": [{"chain_step": 1}],
        "owned_temporary_artifacts": [str(owned), str(unrelated)]})
    assert not deadline.may_start_update()
    deadline.expire()
    report = json.loads(args.output.read_text())
    assert exits == [124] and killed == [True]
    assert waited == [.5] and not owned.exists() and completed.exists() and unrelated.exists()
    assert report["timeout_removed_temporary_artifacts"] == [str(owned)]
    assert report["timeout_cleanup_errors"][0]["error"] == "refused nonexclusive temporary path"
    assert report["status"] == "strict_failed" and report["hard_deadline_reached"]
    assert report["changing_pairs"] == [{"chain_step": 1}] and report["adoption"] is False


def test_next_pair_budget_uses_measured_save_and_compare_cost(tmp_path):
    args = SimpleNamespace(output=tmp_path/"out.json", wall_seconds=480, save_reserve_seconds=30)
    deadline = runner.ExecutionDeadline(args, started=0, clock=lambda: 400.)
    assert deadline.may_start_update()
    assert deadline.may_start_pair(20.)
    assert not deadline.may_start_pair(40.)


def test_nonfinite_state_is_fatal_before_attachment():
    parameters = {"weight": jnp.ones(1, dtype=jnp.float32)}
    sample = Samples(Inputs(jnp.ones((4, 3854), dtype=jnp.float32)))
    evaluator = FullStateADEvaluator(jax,
        lambda *args: ((jnp.float32(1), {"hard": {"ok": 0}}),
                      ({"weight": jnp.ones(1, dtype=jnp.float32)}, jnp.full((4, 3854), jnp.nan))), "daily")
    with pytest.raises(ValueError, match="finite gate"):
        evaluator(parameters, sample)


def test_parameter_snapshot_owned_full_gradient_arrays_and_failed_optimizer(tmp_path):
    import pickle
    array = np.array([1., 2.], dtype=np.float32)
    optimizer = {"step": np.int32(1), "m": array.copy(), "v": array.copy()}
    item = {"gradient": {"weight": array}, "parameters": {"weight": array}, "optimizer": optimizer,
        "components": {name: {"gradient": {"weight": array}, "initial_state_gradient": array,
            "initial_state_gradients_micro4_mean": array} for name in ("daily", "T7", "T30")}}
    comparison = {"comparisons": {key: {"passed": key != "optimizer"}
        for key in ("gradient", "parameters", "optimizer")}}
    owned = runner.parameter_snapshot(jax, {"old": item, "new": item}, comparison)
    path = tmp_path/"pair.pkl"
    runner.atomic_pickle(path, owned)
    array[:] = 99
    optimizer["m"][:] = 77
    del item
    with path.open("rb") as handle: restored = pickle.load(handle)
    for name in ("old", "new"):
        np.testing.assert_array_equal(owned[name]["mixed_gradient"]["weight"], [1, 2])
        for component in ("daily", "T7", "T30"):
            np.testing.assert_array_equal(restored[name]["component_gradients"][component]["weight"], [1, 2])
            np.testing.assert_array_equal(restored[name]["component_initial_state_gradients"][component], [1, 2])
        np.testing.assert_array_equal(restored[name]["parameters"]["weight"], [1, 2])
        np.testing.assert_array_equal(restored[name]["optimizer"]["m"], [1, 2])
        assert restored[name]["optimizer"]["step"] == 1
    passed = {"comparisons": {key: {"passed": True} for key in ("gradient", "parameters", "optimizer")}}
    minimal = runner.parameter_snapshot(jax, {"old": {"gradient": {"weight": jnp.ones(1)},
        "components": {name: {"gradient": {"weight": jnp.ones(1)}, "initial_state_gradient": jnp.ones(1),
            "initial_state_gradients_micro4_mean": jnp.ones(1)} for name in ("daily", "T7", "T30")}}}, passed)
    assert set(minimal["old"]) == {"mixed_gradient", "component_gradients", "component_initial_state_gradients",
                                  "daily_initial_state_gradients_micro4_mean"}


def resume_fixture(tmp_path):
    path, manifest = fixture_manifest(tmp_path)
    report_path = Path(__file__).resolve().parents[2]/"outputs/diagnostics/shared-latent-reader-20261004/shared-original-full-state-schedule-20261005.json"
    prior = json.loads(report_path.read_text())
    resume = {"old": str(tmp_path/"old.pkl"), "new": str(tmp_path/"new.pkl"),
              "completed_pairs": 1, "prior_report": str(tmp_path/"prior.json")}
    for name in ("old", "new"): Path(resume[name]).write_bytes(b"dryrun must not unpickle")
    prior["retained_diagnostic_chains"] = {key: resume[key] for key in ("old", "new")}
    for name, item in manifest["components"].items():
        prior["admissions"][name].update(report=item["report"], executable=item["executable"])
    Path(resume["prior_report"]).write_text(json.dumps(prior))
    manifest["resume"] = resume
    path.write_text(json.dumps(manifest))
    return path, manifest, prior


def test_resume_actual_prior_metadata_sticky_strict_and_independent_array_validation(tmp_path):
    _path, manifest, prior = resume_fixture(tmp_path)
    _, reference = runner.read_reference(manifest)
    admission = runner.validate_resume_metadata(manifest, reference)
    assert admission["start_step"] == 1 and not admission["prior_strict_parity_passed"]
    assert admission["prior_optimizer_attempts"] == {"old": 1, "new": 1}
    parameters = {"weight": np.array([2.], dtype=np.float32)}
    optimizer = initialize_target_oriented_adamw(parameters)
    saved = {"scope": "original_full_state_schedule_diagnostic_not_formal_training", "schedule": "old",
        "manifest": {key: value for key, value in manifest.items() if key != "resume"},
        "completed_diagnostic_updates": 1, "parameters": deepcopy(parameters),
        "optimizer": optimizer._replace(step=np.int32(1)), "adoption": False, "scientific_acceptance": False}
    old, old_adam = runner.validate_resume_chain(jax, saved, manifest=manifest, schedule="old",
        original_parameters=parameters, original_optimizer=optimizer)
    saved_new = deepcopy(saved)
    saved_new["schedule"] = "new"
    saved_new["parameters"]["weight"][0] += np.float32(2e-7)
    new, new_adam = runner.validate_resume_chain(jax, saved_new, manifest=manifest, schedule="new",
        original_parameters=parameters, original_optimizer=optimizer)
    assert old is saved["parameters"] and new is saved_new["parameters"]
    assert not np.array_equal(old["weight"], new["weight"])
    assert int(old_adam.step) == int(new_adam.step) == 1
    for mutation in ("schedule", "manifest", "step", "nonfinite", "shape", "dtype", "negative_second"):
        bad = deepcopy(saved)
        if mutation == "schedule": bad["schedule"] = "new"
        if mutation == "manifest": bad["manifest"]["original_checkpoint"] = "other"
        if mutation == "step": bad["optimizer"] = bad["optimizer"]._replace(step=np.int32(0))
        if mutation == "nonfinite": bad["parameters"]["weight"][0] = np.nan
        if mutation == "shape": bad["parameters"]["weight"] = np.ones(2, dtype=np.float32)
        if mutation == "dtype": bad["parameters"]["weight"] = np.ones(1, dtype=np.float64)
        if mutation == "negative_second": bad["optimizer"] = bad["optimizer"]._replace(second={"weight": np.array([-1.], dtype=np.float32)})
        with pytest.raises(ValueError):
            runner.validate_resume_chain(jax, bad, manifest=manifest, schedule="old",
                original_parameters=parameters, original_optimizer=optimizer)


def test_resume_public_dispatch_step1_to5_continuation_artifact275(monkeypatch, tmp_path):
    old, new, payloads, seen = setup_chains()
    monkeypatch.setattr(runner, "require_memory", lambda *_: None)
    try:
        runner.execute_pairs(jax=jax, old=old, new=new, payloads=payloads, artifact_ids=(742,),
            args=SimpleNamespace(updates=1, warmup_proposals=0), deadline=SimpleNamespace(may_start_update=lambda: True),
            emit=lambda *a, **kw: None, save_states=lambda _: None, save_pair=lambda *a: "prior.pkl")
        # Both saved chains retain their own values, including admitted small FP32 differences.
        new.parameters = {"weight": new.parameters["weight"]+jnp.float32(2e-7)}
        second = {"daily": payloads[0]["daily"], "cases": {
            h: (case[0]+2, *case[1:]) for h, case in payloads[0]["cases"].items()}}
        seen.clear()
        stored = []
        def save(index, values, comparison):
            stored.append(index)
            path = tmp_path/f"pair-{index}.pkl"
            runner.atomic_pickle(path, runner.parameter_snapshot(jax, values, comparison))
            return str(path)
        rows, states, attempts, strict = runner.execute_pairs(jax=jax, old=old, new=new,
            payloads=[payloads[0], second], artifact_ids=(742, 275),
            args=SimpleNamespace(updates=4, warmup_proposals=0, start_step=1, initial_strict_parity_passed=False,
                                 prior_optimizer_attempts={"old": 1, "new": 1}),
            deadline=SimpleNamespace(may_start_update=lambda: True), emit=lambda *a, **kw: None,
            save_states=lambda _: None, save_pair=save)
        assert stored == [2, 3, 4, 5] and len(states) == 4
        assert [row["artifact_id"] for row in rows] == [275, 742, 275, 742]
        assert [entry[1] for entry in seen] == [1, 1, 2, 2, 3, 3, 4, 4]
        assert [entry[3] for entry in seen] == [32., 32., 30., 30., 32., 32., 30., 30.]
        assert seen[0][0] == frozenset((30,)) and seen[1][0] == frozenset((7, 30))
        assert not np.array_equal(seen[0][2], seen[1][2])
        assert attempts == {"old": 5, "new": 5} and not strict
        assert int(old.optimizer.step) == int(new.optimizer.step) == 5
        summary = runner.timing_summary(rows)
        assert summary["first_step_boundary_index"] == 2
        assert summary["post_first_pairs"]["pair_indices"] == [3, 4, 5]
        assert summary["minimum_three_post_first_pairs_complete"]
    finally:
        old.close(); new.close()


def test_resume_actual_metadata_cli_external900_and_no_pickle_or_jax(tmp_path):
    path, manifest, _ = resume_fixture(tmp_path)
    import os
    (tmp_path/"sitecustomize.py").write_text("import sys\nclass Guard:\n def find_spec(self,name,*a):\n  if name == 'jax' or name.startswith('jax.'):\n   raise RuntimeError('resume dryrun imported JAX')\nsys.meta_path.insert(0,Guard())\n")
    code = "import runpy,sys; runpy.run_path(sys.argv.pop(1),run_name='__main__')"
    command = [sys.executable, "-B", "-c", code, str(Path(runner.__file__)), "--manifest", str(path),
        "--output", str(tmp_path/"resume-out.json"), "--parent-cuda-device", "0", "--worker-cuda-device", "1",
        "--external-wall-cap-seconds", "900", "--wall-seconds", "880", "--dry-run"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["resume"]["start_step"] == 1 and report["end_step_if_completed"] == 5
    assert report["optimizer_attempt_upper_bound_per_chain"] == 5
    assert report["resume"]["array_identity_verified"] is False
    assert report["external_wall_cap_seconds"] == 900 and report["GPU_executed"] is False
    assert not (tmp_path/"resume-out.json").exists()
    args = runner.parser().parse_args(command[5:])
    args.external_wall_cap_seconds = 500
    with pytest.raises(ValueError, match="wall cap"): runner.validate_arguments(args, start_step=1)
    args.external_wall_cap_seconds = 900
    with pytest.raises(ValueError, match="cap"): runner.validate_arguments(args, start_step=1, previous_attempts=13)


@pytest.mark.parametrize("failure", ["partial", "assets", "cursor", "attempts", "parameter_parity"])
def test_resume_prior_metadata_rejects_unadmitted_associations(tmp_path, failure):
    _, manifest, prior = resume_fixture(tmp_path)
    if failure == "partial": prior["final_optimizer_steps"]["new"] = 2
    if failure == "assets": prior["admissions"]["T7"]["executable"] = "candidate.pkl"
    if failure == "cursor": prior["changing_pairs"][-1]["artifact_id"] = 275
    if failure == "attempts": prior.pop("optimizer_attempts")
    if failure == "parameter_parity": prior["changing_pairs"][-1]["strict_comparison"]["comparisons"]["parameters"]["passed"] = False
    Path(manifest["resume"]["prior_report"]).write_text(json.dumps(prior))
    _, reference = runner.read_reference(manifest)
    with pytest.raises(ValueError): runner.validate_resume_metadata(manifest, reference)


def test_real_settled_report_metadata_matches_resume_manifest(monkeypatch):
    root = Path(__file__).resolve().parents[2]/"outputs/diagnostics/shared-latent-reader-20261004"
    manifest = runner.read_manifest(root/"original-full-state-schedule-resume.manifest.json")
    manifest["resume"]["prior_report"] = str(root/"shared-original-full-state-schedule-20261005.json")
    original_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: path.as_posix().startswith("/WORK/") or original_is_file(path))
    report = json.loads(Path(manifest["resume"]["prior_report"]).read_text())
    reference = {"source_git_head": report["source_git_head"], "method": report["admissions"]["daily"]["identity"]["method"]}
    admission = runner.validate_resume_metadata(manifest, reference)
    assert admission["start_step"] == 1 and admission["prior_strict_parity_passed"] is False
    assert admission["chain_paths"] == report["retained_diagnostic_chains"]
