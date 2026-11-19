"""CHECK_CWRR resolver, complete-owner rejection and cache admission contracts.

No full timestep or model data is executed. Numeric residual gates live in the
separate portable PB-03 component tests.
"""

import inspect
from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace as NS

import pytest

from jax_orchidee.parameters.soil import (
    SoilParameters,
    config_soil_parameters,
    require_supported_hydrol_check_cwrr,
    resolve_hydrol_check_cwrr,
)


@pytest.mark.parametrize("value", [False, "FALSE", ".false.", "n", "NO", "0"])
def test_explicit_false_and_default_are_effective_false(value):
    assert resolve_hydrol_check_cwrr() is False
    assert resolve_hydrol_check_cwrr({" check_cwrr ": value}) is False
    assert require_supported_hydrol_check_cwrr(value) is False


@pytest.mark.parametrize("value", [True, "TRUE", ".true.", "y", "YES", "1"])
def test_true_is_resolved_then_refused_at_complete_owner(value):
    assert resolve_hydrol_check_cwrr({"CHECK_CWRR": value}) is True
    with pytest.raises(NotImplementedError, match="effective CHECK_CWRR=true.*conservation and ET"):
        require_supported_hydrol_check_cwrr(value, entrypoint="test complete owner")


@pytest.mark.parametrize("value", [None, "maybe", "", 2])
def test_invalid_logical_never_becomes_false(value):
    with pytest.raises(ValueError, match="invalid Fortran logical"):
        resolve_hydrol_check_cwrr({"CHECK_CWRR": value})


@pytest.mark.parametrize(
    "ok_sechiba,impose_param,hydrol_cwrr", [(False, False, False), (True, False, True), (True, True, True)]
)
def test_shared_soil_resolver_resets_saved_default_and_keeps_override(ok_sechiba, impose_param, hydrol_cwrr):
    controls = dict(ok_sechiba=ok_sechiba, impose_param=impose_param, hydrol_cwrr=hydrol_cwrr)
    prior = replace(SoilParameters(), check_cwrr=True)
    assert config_soil_parameters({}, initial=prior, **controls).check_cwrr is False
    assert config_soil_parameters({"CHECK_CWRR": "FALSE"}, initial=prior, **controls).check_cwrr is False
    assert config_soil_parameters({"CHECK_CWRR": "TRUE"}, **controls).check_cwrr is True


def test_public_prepare_guards_changed_same_path_before_cache_hit(monkeypatch, tmp_path):
    from jax_orchidee.driver import orchestration as o

    path = tmp_path / "used_run.def"
    materializations = []

    @lru_cache(maxsize=8)
    def fake_cached(*args):
        materializations.append(args)
        return NS(run_def_values={}, hydrol_check_cwrr=False)

    monkeypatch.setattr(o, "_prepare_paper_1961_clock_context_cached", fake_cached)
    config = tmp_path / "case.yaml"
    path.write_text("", encoding="utf-8")
    implicit = o.prepare_paper_1961_driver_context(config, used_run_def_path=path)
    path.write_text("CHECK_CWRR=FALSE\n", encoding="utf-8")
    explicit = o.prepare_paper_1961_driver_context(config, used_run_def_path=path)
    assert implicit is explicit
    assert len(materializations) == 1
    prior_hits = fake_cached.cache_info().hits
    path.write_text("CHECK_CWRR=TRUE\n", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="CHECK_CWRR=true"):
        o.prepare_paper_1961_driver_context(config, used_run_def_path=path)
    assert fake_cached.cache_info().hits == prior_hits
    assert len(materializations) == 1


@pytest.mark.parametrize(
    "run_mode,run_id",
    [("new_native_run", None), ("new_native_run", "explicit-native-binding"), ("same_run_resume", "resume-binding")],
    ids=["cold-cached", "ordinary-native", "restart"],
)
def test_public_prepare_rejects_true_for_cold_ordinary_restart_before_io(tmp_path, monkeypatch, run_mode, run_id):
    from jax_orchidee.driver import orchestration as o

    path = tmp_path / "used_run.def"
    path.write_text("CHECK_CWRR=TRUE\n", encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported true reached a prepared context cache")

    monkeypatch.setattr(o, "_prepare_paper_1961_clock_context_cached", forbidden)
    monkeypatch.setattr(o, "_prepare_paper_1961_driver_context_cached", forbidden)
    with pytest.raises(NotImplementedError, match="CHECK_CWRR=true"):
        o.prepare_paper_1961_driver_context(
            tmp_path / "case.yaml",
            used_run_def_path=path,
            run_mode=run_mode,
            run_id=run_id,
        )


def _required_none_arguments(function):
    return {
        name: None
        for name, p in inspect.signature(function).parameters.items()
        if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


@pytest.mark.parametrize("compiled", [False, True])
def test_complete_module_refuses_true_before_math(compiled):
    from jax_orchidee.sechiba import hydrol as h

    function = h._hydrol_module_explicit_step_jit if compiled else h.hydrol_module_explicit_step
    arguments = _required_none_arguments(h.hydrol_module_explicit_step)
    with pytest.raises(NotImplementedError, match="CHECK_CWRR=true"):
        function(**arguments, check_cwrr=True)
    assert inspect.signature(h.hydrol_soil_all_tiles_explicit_step).parameters["check_cwrr"].default is True


def test_complete_wrapper_refuses_payload_true_before_state_validation():
    from jax_orchidee.sechiba import hydrol as h

    precall = NS(payload={"check_cwrr": True}, missing_inputs=(), ok=True)
    with pytest.raises(NotImplementedError, match="CHECK_CWRR=true"):
        h.run_hydrol_first_step_module_from_precall(precall, humcste=None, dz_mm=None, dh_mm=None, zz_mm=None)


@pytest.mark.parametrize("entry", ["_paper_compiled_sechiba_scan", "_paper_compiled_later_day_block_executable"])
def test_compiled_cache_admission_refuses_true_before_layout_or_lookup(entry):
    from jax_orchidee.driver import orchestration as o

    function = getattr(o, entry)
    arguments = _required_none_arguments(function)
    arguments["context"] = NS(run_def_values={"CHECK_CWRR": "TRUE"}, hydrol_check_cwrr=True)
    with pytest.raises(NotImplementedError, match="CHECK_CWRR=true"):
        function(**arguments)
