"""Candidate screen wiring and unchanged numerical gates; no model compile."""

import json
import pickle
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from scripts.hpc import probe_shared_neural_precision as neural
from scripts.hpc import probe_shared_union_micro4 as micro4
from scripts.hpc import probe_shared_union_neural_precision as screen


def prior(identity):
    return {"status": "bounded_validity_passed", "mode": "union", "identity": identity,
            "precision": "unchanged_FP32_neural_FP64_physical", "neural_remat": True, "loss_multiplier": 1.0,
            "compile_units": [{"ordinal": i, "module": name, "status": "completed",
                               "main_signature": {"result_leaves": 1}} for i, name in enumerate(screen.STAGE_NAMES)]}


def test_requires_completed_six_units_and_same_union_reference():
    identity = {"artifact_id": 742}
    assert screen.validate_prior_report(prior(identity), identity)["mandatory_cache_hit_ordinals"] == [1, 2, 3, 4]
    for change in ({"mode": "field"}, {"neural_remat": False}, {"loss_multiplier": 2.0},
                   {"precision": "mixed"}, {"status": "partial"}, {"identity": {"artifact_id": 1}}):
        with pytest.raises(ValueError):
            screen.validate_prior_report(prior(identity) | change, identity)
    bad = prior(identity)
    bad["compile_units"][3]["status"] = "compiling"
    with pytest.raises(ValueError, match="unit identity"):
        screen.validate_prior_report(bad, identity)
    with pytest.raises(ValueError, match="six"):
        screen.validate_prior_report(prior(identity) | {"compile_units": []}, identity)


def test_cache_copy_preserves_source_and_refuses_reuse(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "candidate"
    source.mkdir()
    for name in ("jit_transition_stage-abc-cache", "jit_loss_stage-def-cache"):
        (source / name).write_bytes(b"preserved")
    result = screen.clone_cache(source, destination)
    assert result["files"] == 2 and result["bytes"] == 18
    assert result["reference_cache_modified"] is False
    assert (source / "jit_transition_stage-abc-cache").read_bytes() == b"preserved"
    with pytest.raises(FileExistsError):
        screen.clone_cache(source, destination)
    with pytest.raises(ValueError, match="separate"):
        screen.clone_cache(source, source / "inside")


@pytest.mark.parametrize('name,mode', [('jit_transition_stage', 'candidate'), ('jit_neural_stage', 'union')])
def test_mandatory_stage_cache_miss_stops_before_backend(monkeypatch, tmp_path, name, mode):
    from jax._src import compiler
    backend_calls = []
    monkeypatch.setattr(compiler, "backend_compile", lambda *a, **kw: backend_calls.append(1), raising=False)
    monkeypatch.setattr(compiler, "compile_or_get_cached", lambda *a, **kw: compiler.backend_compile())
    signature = {'result_leaves': 1, 'result_nominal_bytes': 4}
    monkeypatch.setattr(neural, "module_signature", lambda module: signature)
    monkeypatch.setattr(neural, "deadline", lambda *a, **kw: nullcontext())
    previous = {"identity": {"x": 1}, "neural_remat": True, "mode": "union",
                "compile_units": [{"module": name, "status": "completed", "main_signature": signature}]}
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(previous))
    args = SimpleNamespace(mode=mode, neural_remat=True, reuse_stage_report=path,
                           required_cache_ordinals=(0,), output=tmp_path / "out.json", memory_limit_gib=24,
                           final_neural_compile_seconds=600)
    report = {"identity": {"x": 1}}
    helper = SimpleNamespace(atomic_pickle=lambda *a: None)
    module = SimpleNamespace(operation=SimpleNamespace(attributes={"sym_name": '"'+name+'"'}, get_asm=lambda: "ir"))
    with neural.stage_compilation_evidence(args, helper, lambda *a, **kw: None, report):
        with pytest.raises(RuntimeError, match="refusing to recompile"):
            compiler.compile_or_get_cached(None, module)
    assert backend_calls == []
    assert report["compile_units"][0]["status"] == "completed_unit_cache_miss"


def test_cache_compatibility_mode_is_fp32_and_requires_all_six_caches():
    policy = screen.execution_policy(True)
    assert policy == {'mode': 'union', 'required_cache_ordinals': (0, 1, 2, 3, 4, 5),
                      'mixed': False, 'precision': 'unchanged_FP32_neural_FP64_physical'}
    mixed = screen.execution_policy(False)
    assert mixed['required_cache_ordinals'] == (1, 2, 3, 4) and mixed['mixed'] is True
    assert screen.parser().get_default('cache_compatibility_only') is False


def test_cache_only_refuses_unexpected_seventh_model_backend(monkeypatch, tmp_path):
    from jax._src import compiler
    backend_calls = []
    monkeypatch.setattr(compiler, 'backend_compile', lambda *a, **kw: backend_calls.append(1), raising=False)
    signature = {'result_leaves': 1, 'result_nominal_bytes': 4}
    monkeypatch.setattr(neural, 'module_signature', lambda module: signature)
    monkeypatch.setattr(neural, 'deadline', lambda *a, **kw: nullcontext())
    lookups = []
    def lookup(*arguments, **keywords):
        lookups.append(1)
        return SimpleNamespace(get_compiled_memory_stats=lambda: None) if len(lookups) <= 6 else compiler.backend_compile()
    monkeypatch.setattr(compiler, 'compile_or_get_cached', lookup)
    prior_report = {'identity': {}, 'neural_remat': False, 'mode': 'union',
                    'compile_units': [{'module': name, 'status': 'completed', 'main_signature': signature}
                                      for name in screen.STAGE_NAMES]}
    path = tmp_path/'prior.json'
    path.write_text(json.dumps(prior_report))
    args = SimpleNamespace(mode='union', neural_remat=False, reuse_stage_report=path,
                           required_cache_ordinals=tuple(range(6)), forbid_named_model_backend=True,
                           output=tmp_path/'out.json', memory_limit_gib=24, final_neural_compile_seconds=600)
    report = {'identity': {}}
    with neural.stage_compilation_evidence(args, SimpleNamespace(atomic_pickle=lambda *a: None), lambda *a, **kw: None, report):
        for name in (*screen.STAGE_NAMES, 'jit_neural_stage'):
            module = SimpleNamespace(operation=SimpleNamespace(attributes={'sym_name': '"'+name+'"'}, get_asm=lambda: 'IR'))
            if len(lookups) < 6:
                compiler.compile_or_get_cached(None, module)
            else:
                with pytest.raises(RuntimeError, match='refusing to recompile'):
                    compiler.compile_or_get_cached(None, module)
    assert backend_calls == []
    assert report['compile_units'][6]['status'] == 'model_backend_forbidden'


def test_interface_mismatch_is_retained_before_native_lookup(monkeypatch, tmp_path):
    from jax._src import compiler
    lookups = []
    monkeypatch.setattr(compiler, 'compile_or_get_cached', lambda *a, **kw: lookups.append(1))
    monkeypatch.setattr(compiler, 'backend_compile', lambda *a, **kw: None, raising=False)
    signature = {'argument_leaves': 1, 'argument_nominal_bytes': 4, 'result_leaves': 1,
                 'result_nominal_bytes': 4, 'argument_types': ['tensor<f32>'], 'result_types': ['tensor<f32>']}
    monkeypatch.setattr(neural, 'module_signature', lambda module: signature)
    path = tmp_path/'prior.json'
    old = signature | {'argument_types': ['tensor<i32>']}
    path.write_text(json.dumps({'identity': {}, 'neural_remat': False, 'mode': 'union',
                               'compile_units': [{'module': 'jit_transition_stage', 'status': 'completed', 'main_signature': old}]}))
    args = SimpleNamespace(mode='union', neural_remat=False, reuse_stage_report=path,
                           required_cache_ordinals=(0,), output=tmp_path/'out.json')
    saved = []
    report = {'identity': {}}
    module = SimpleNamespace(operation=SimpleNamespace(attributes={'sym_name': '"jit_transition_stage"'}, get_asm=lambda: 'newIR'))
    with neural.stage_compilation_evidence(args, SimpleNamespace(atomic_pickle=lambda path, value: saved.append(value)), lambda *a, **kw: None, report):
        with pytest.raises(ValueError, match='ordered interface'):
            compiler.compile_or_get_cached(None, module)
    assert lookups == [] and saved == ['newIR']
    event = report['compile_units'][0]
    assert event['status'] == 'completed_unit_interface_mismatch'
    assert event['main_signature'] == signature and event['prior_main_signature'] == old


def test_cache_only_reads_preserved_ordered_interface(monkeypatch, tmp_path):
    from jax._src import compiler
    def transition_stage(value):
        return value * value
    module = jax.jit(transition_stage).lower(np.ones(2, np.float32)).compiler_ir()
    old_path = tmp_path/'old.hlo.pkl'
    with old_path.open('wb') as handle:
        pickle.dump(module.operation.get_asm(), handle)
    signature = neural.module_signature(module)
    prior_path = tmp_path/'prior.json'
    prior_path.write_text(json.dumps({'identity': {}, 'neural_remat': False, 'mode': 'union',
        'compile_units': [{'module': 'jit_transition_stage', 'status': 'completed',
                           'main_signature': {key: value for key, value in signature.items() if not key.endswith('_types')},
                           'stablehlo_artifact': str(old_path)}]}))
    backend_calls = []
    monkeypatch.setattr(compiler, 'backend_compile', lambda *a, **kw: backend_calls.append(1), raising=False)
    monkeypatch.setattr(compiler, 'compile_or_get_cached', lambda *a, **kw: compiler.backend_compile())
    args = SimpleNamespace(mode='union', neural_remat=False, reuse_stage_report=prior_path,
                           required_cache_ordinals=(0,), forbid_named_model_backend=True, require_saved_ir_interface=True,
                           output=tmp_path/'out.json', memory_limit_gib=24)
    report = {'identity': {}}
    with neural.stage_compilation_evidence(args, SimpleNamespace(atomic_pickle=lambda *a: None), lambda *a, **kw: None, report):
        with pytest.raises(RuntimeError, match='refusing to recompile'):
            compiler.compile_or_get_cached(None, module)
    assert backend_calls == []
    event = report['compile_units'][0]
    assert event['prior_interface_evidence']['ordered_prior_types'] == 'recorded_and_equal'
    assert event['prior_stablehlo_artifact'] == str(old_path)


def test_cache_replay_strictly_compares_full_auxiliary_and_gradients():
    reference = {'loss': np.float64(1), 'gradients': ({'p': np.array([1., 2.], np.float32)}, np.ones(2)),
                 'diagnostics': {'outputs': {'x': np.ones(2)}, 'mask': np.array([True, False]),
                                 'auxiliary': {'expected': np.array([1., np.nan])}}}
    assert screen.strict_saved_result_comparison(jax, reference, reference)['passed']
    changed = reference | {'loss': np.float64(1.0001)}
    assert not screen.strict_saved_result_comparison(jax, changed, reference)['passed']
    changed = reference | {'gradients': ({'p': np.array([1., 2.001], np.float32)}, np.ones(2))}
    assert not screen.strict_saved_result_comparison(jax, changed, reference)['passed']
    changed = reference | {'diagnostics': reference['diagnostics'] | {'mask': np.array([False, False])}}
    assert not screen.strict_saved_result_comparison(jax, changed, reference)['passed']
    changed = reference | {'loss': np.float32(1)}
    with pytest.raises(ValueError, match='shape/dtype'):
        screen.strict_saved_result_comparison(jax, changed, reference)


def test_cache_admission_never_unlocks_mixed_micro4(tmp_path):
    path = tmp_path/'cache.json'
    identity = {'artifact_id': 742}
    path.write_text(json.dumps({'status': 'cache_compatibility_passed', 'cache_admission_eligible': True,
                                'screening_passed': False, 'second_stage_eligible': False,
                                'identity': identity, 'contract': neural.CONTRACT}))
    with pytest.raises(ValueError, match='full parameter/state'):
        micro4.validate_precision_screen(path, neural, identity)


def test_neural_patch_restored_after_exception():
    def original(value):
        return value
    module = SimpleNamespace(_linear=original, _ffn=original)
    with pytest.raises(RuntimeError):
        with neural.neural_policy([module], mixed=True):
            assert module._linear is neural.algorithm_linear
            raise RuntimeError("stop")
    assert module._linear is original and module._ffn is original


def test_micro4_requires_passing_full_gradient_report(tmp_path):
    path = tmp_path / "screen.json"
    identity = {"x": 1}
    report = {"status": "screening_passed", "screening_passed": True, "second_stage_eligible": True,
              "identity": identity, "contract": neural.CONTRACT, "loss_multiplier": 1.0,
              "candidate_gradient_contract": {"parameter_leaves": 300, "parameter_coordinates": 64_884_481,
                  "continuous_state_coordinates": 3854, "all_measured_values_finite": True,
                  "finite_scope": screen.FINITE_SCOPE},
              "scope": "saved_union_row0_full_parameter_and_state_AD_precision_screen"}
    path.write_text(json.dumps(report))
    assert micro4.validate_precision_screen(path, neural, identity)["full_parameter_and_state_screening_passed"]
    for change in ({"screening_passed": False}, {"status": "screening_failed"}, {"loss_multiplier": 2.0},
                   {"contract": neural.CONTRACT | {"gradient_relative_l2_limit": 0.1}}):
        path.write_text(json.dumps(report | change))
        with pytest.raises(ValueError, match="full parameter/state"):
            micro4.validate_precision_screen(path, neural, identity)


def test_parameter_engineering_gate_reuses_original_thresholds():
    reference = ((np.asarray(2.0), {}), {"root": np.array([1.0, 2.0], np.float32)})
    actual = ((np.asarray(2.001), {}), {"root": np.array([1.001, 2.002], np.float32)})
    result = micro4.precision_comparisons(neural, jax, actual, reference)
    assert result["screening_passed"]
    assert result["contract"] is neural.CONTRACT
    bad = (actual[0], {"root": np.array([1.1, 2.2], np.float32)})
    assert not micro4.precision_comparisons(neural, jax, bad, reference)["screening_passed"]


def test_candidate_is_saved_before_gate_and_has_no_warm_replay():
    source = Path(screen.__file__).read_text()
    assert source.index("helper.atomic_pickle(args.result, saved)") < source.index("gradient_contract = validate_full_result")
    assert "range(args.repeats)" not in source
    assert source.index('os.environ["JAX_COMPILATION_CACHE_DIR"]') < source.index("probe.load_helper")
    assert screen.parser().get_default("process_seconds") == 1200
    assert screen.parser().get_default("neural_forward_compile_seconds") == 300
    assert screen.parser().get_default("final_neural_compile_seconds") == 600
    assert 'host_sample = jax.tree_util.tree_map(lambda value: np.asarray(value)[0]' in source
    assert 'sample = jax.tree_util.tree_map(lambda value: jnp.asarray(value)[0]' in source
    assert 'state = sample.inputs.continuous_state' in source


def test_refuses_incomplete_native_gradient_pair():
    saved = {"identity": {}, "diagnostic_sample": "prepared_micro4_row0", "route": "union",
             "native_ad": "all_parameters_and_continuous_state", "gradients": ({},)}
    with pytest.raises(ValueError, match="gradient pair"):
        screen.validate_full_result(jax, saved, {}, {}, np.zeros(3854), reference=True)


def test_full_validator_accepts_structured_ragged_boolean_masks(monkeypatch):
    monkeypatch.setattr(screen, "PARAMETER_LEAVES", 2)
    monkeypatch.setattr(screen, "PARAMETER_COUNT", 3)
    monkeypatch.setattr(screen, "STATE_COUNT", 4)
    parameters = {"a": np.ones(1, np.float32), "b": np.ones(2, np.float32)}
    masks = {"other": np.array([True, False]), "annual": (np.ones(1, bool), np.ones(3, bool))}
    state_values = jax.tree_util.tree_map(lambda mask: np.where(mask, 1.0, np.nan), masks)
    diagnostics = {"outputs": {"out": np.ones(3, np.float64)}, "raw_masks": {"out": np.ones(3, bool)},
        "auxiliary": {"hard": {"invalid": np.asarray(0)},
                      "state_diagnostics": {"predicted_defined": masks, "expected_defined": masks,
                                            "expected": state_values, "predicted": state_values}}}
    saved = {"identity": {}, "diagnostic_sample": "prepared_micro4_row0", "route": "union",
             "native_ad": "all_parameters_and_continuous_state", "loss": np.asarray(1.0),
             "gradients": (parameters, np.ones(4)), "diagnostics": diagnostics}
    result = screen.validate_full_result(jax, saved, {}, parameters, np.ones(4), reference=True)
    assert result["mask_schema"]["predicted_defined"] == {"leaves": 3, "coordinates": 6}
    assert result["finite_scope"] == screen.FINITE_SCOPE
    assert result["state_diagnostic_finiteness"]["expected"] == {"nonfinite": 1, "nonfinite_defined": 0, "nonfinite_undefined": 1}
    state_values["other"][0] = np.nan
    with pytest.raises(screen.NumericalFailure, match="declared-defined"):
        screen.validate_full_result(jax, saved, {}, parameters, np.ones(4), reference=True)
    state_values["other"][0] = 1.0
    diagnostics["outputs"]["out"][0] = np.nan
    with pytest.raises(screen.NumericalFailure, match="physical output"):
        screen.validate_full_result(jax, saved, {}, parameters, np.ones(4), reference=True)
    diagnostics["outputs"]["out"][0] = 1.0
    masks["other"] = np.ones(2, np.float32)
    with pytest.raises(ValueError, match="only boolean leaves"):
        screen.validate_full_result(jax, saved, {}, parameters, np.ones(4), reference=True)
