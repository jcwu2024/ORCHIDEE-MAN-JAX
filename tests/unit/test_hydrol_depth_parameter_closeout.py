import pickle
from dataclasses import FrozenInstanceError

import jax
import numpy as np
import pytest

from jax_orchidee.sechiba import hydrol as h

CUSTOM = {
    "CWRR_NKS_N0": 0.85,
    "CWRR_NKS_POWER": 0.45,
    "CWRR_AKS_A0": 0.0002,
    "CWRR_AKS_POWER": 0.65,
    "KFACT_DECAY_RATE": 1.5,
    "KFACT_STARTING_DEPTH": 0.25,
    "KFACT_MAX": 20.0,
}


def _initialization_inputs():
    z_m = np.array([0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0])
    d_m = np.r_[z_m[0], np.diff(z_m)]
    dh_m = np.r_[(d_m[:-1] + d_m[1:]) / 2.0, d_m[-1] / 2.0]
    return {
        "veget": [[0.0, 1.0]], "veget_max": [[0.0, 1.0]],
        "soiltile": [[0, 0, 1, 0, 0, 0]], "njsc": [2],
        "znh_m": z_m, "dnh_m": d_m, "dlh_m": dh_m,
        "humcste": [1.0, 3.0], "altmax": [[0.0, 0.0]], "peat_hydro": True,
    }


@pytest.fixture(scope="module")
def table_pair():
    payload = {"njsc": [2], "mcs": [h.USDA_MCS[1]], "zz_mm": np.array([1.0, 550.0, 2000.0])}
    soil = h.hydrol_soil_parameters()
    baseline = h.hydrol_runtime_static_tables_from_payload(payload, soil_parameters=soil, peat_hydro=True)
    depth = h.hydrol_depth_parameters(CUSTOM)
    custom = h.hydrol_runtime_static_tables_from_payload(
        payload, soil_parameters=soil, depth_parameters=depth, peat_hydro=True,
    )
    return payload, soil, baseline, custom


def test_depth_contract_defaults_guards_and_legacy_schema():
    depth = h.hydrol_depth_parameters()
    assert depth.as_config_values() == {
        "altmax_for_nroot_thresh": 3.0, "CWRR_NKS_N0": 0.95, "CWRR_NKS_POWER": 0.34,
        "CWRR_AKS_A0": 0.00012, "CWRR_AKS_POWER": 0.53, "KFACT_DECAY_RATE": 2.0,
        "KFACT_STARTING_DEPTH": 0.3, "KFACT_MAX": 10.0,
    }
    assert h.hydrol_depth_parameters(depth) is depth
    assert len(jax.tree_util.tree_leaves(depth)) == 8
    assert len(jax.tree_util.tree_leaves(h.hydrol_soil_parameters())) == 10
    with pytest.raises(FrozenInstanceError):
        depth.n0 = 1.0
    restored = pickle.loads(pickle.dumps(depth))
    assert type(restored) is h.HydrolDepthParameters
    assert restored.as_config_values() == depth.as_config_values()
    for legacy in (h.HydrolRuntimeStaticTables(None, None),
                   h.HydrolRuntimeStaticTables(None, None, h.hydrol_soil_parameters())):
        assert pickle.loads(pickle.dumps(legacy)).depth_parameters is None
    selected = h.hydrol_depth_parameters_from_run_def({**CUSTOM, "KFACT_DECAY_RATE": "1.5D0", "UNRELATED": "x"})
    assert selected.as_config_values() == {**depth.as_config_values(), **CUSTOM}
    # hydrol.f90:4000 checks nk_rel, not the freshly read ak_rel. This is
    # intentional source parity, rather than a new non-negative-ak guard.
    assert float(h.hydrol_depth_parameters({"CWRR_AKS_POWER": -0.2}).ak_rel) == -0.2
    with pytest.raises(ValueError, match="unsupported hydrol_var_init"):
        h.hydrol_depth_parameters({"UNDECLARED": 1.0})
    with pytest.raises(ValueError, match="scalar"):
        h.hydrol_depth_parameters({"KFACT_DECAY_RATE": [1.0, 2.0]})


@pytest.mark.parametrize("key,value", [
    ("CWRR_NKS_N0", -0.1), ("CWRR_NKS_POWER", -0.1), ("CWRR_AKS_A0", -0.1),
    ("KFACT_DECAY_RATE", -0.1), ("KFACT_STARTING_DEPTH", 0.0), ("KFACT_MAX", 9.0),
])
def test_depth_contract_rejects_the_source_fatal_ranges(key, value):
    with pytest.raises(ValueError):
        h.hydrol_depth_parameters({key: value})


def test_custom_depth_initialization_and_actual_runtime_tables_agree():
    depth = h.hydrol_depth_parameters(CUSTOM)
    soil = h.hydrol_soil_parameters()
    initialized = h.hydrol_var_init(**_initialization_inputs(), soil_parameters=soil, depth_parameters=depth)
    runtime = h.hydrol_runtime_static_tables_from_payload(
        {"njsc": [2], "mcs": initialized.mcs, "zz_mm": initialized.zz},
        soil_parameters=soil, depth_parameters=depth, peat_hydro=True,
    )
    assert runtime.depth_parameters is depth
    assert runtime.soil_parameters is soil
    assert initialized.parameters == depth.as_config_values()
    for lhs, rhs in ((initialized.mineral_tables, runtime.mineral), (initialized.peat_tables, runtime.peat)):
        for field in ("kfact", "nfact", "afact", "nvan_mod", "avan_mod", "mc_lin", "k_lin", "a_lin", "b_lin", "d_lin"):
            np.testing.assert_array_equal(getattr(lhs, field), getattr(rhs, field))
    # Independently evaluate the source's depth profile in metres, including
    # the upper and lower clips. Table construction consumes no tile mask.
    z_m = _initialization_inputs()["znh_m"]
    kfact = np.clip(np.exp(-1.5 * (z_m - 0.25)), 1.0 / 20.0, 1.0)
    np.testing.assert_allclose(runtime.mineral.kfact[:, 0], kfact, rtol=1e-14)
    np.testing.assert_allclose(runtime.peat.kfact, kfact, rtol=1e-14)
    np.testing.assert_allclose(runtime.mineral.nvan_mod[:, 0], 0.85 + (h.USDA_NVAN[1] - 0.85) * kfact**0.45, rtol=1e-14)
    np.testing.assert_allclose(runtime.peat.avan_mod, 0.0002 + (h.PEAT_AVAN - 0.0002) * kfact**0.65, rtol=1e-14)
    # The depth family does not replace either mineral backgrounds or fixed
    # VG_* used to derive refSOC-adjusted effective point thresholds.
    np.testing.assert_array_equal(initialized.mcs, [h.USDA_MCS[1]])
    np.testing.assert_array_equal(soil.mcs_mineral, h.USDA_MCS)
    with pytest.raises(ValueError, match="conflict"):
        h.hydrol_var_init(**_initialization_inputs(), depth_parameters=depth,
                         config_values={"KFACT_DECAY_RATE": 2.0})


@pytest.mark.parametrize("key,value,field", [
    ("CWRR_NKS_N0", 1.0, "nvan_mod"), ("CWRR_NKS_POWER", 0.25, "nvan_mod"),
    ("CWRR_AKS_A0", 0.0002, "avan_mod"), ("CWRR_AKS_POWER", 0.4, "avan_mod"),
    ("KFACT_DECAY_RATE", 1.5, "kfact"), ("KFACT_STARTING_DEPTH", 0.2, "kfact"),
    ("KFACT_MAX", 20.0, "kfact"),
])
def test_every_active_depth_scalar_partitions_both_real_table_caches(table_pair, key, value, field):
    payload, soil, baseline, _ = table_pair
    selected = h.hydrol_runtime_static_tables_from_payload(
        payload, soil_parameters=soil, depth_parameters=h.hydrol_depth_parameters({key: value}), peat_hydro=True,
    )
    assert selected.mineral is not baseline.mineral
    assert selected.peat is not baseline.peat
    assert not np.array_equal(getattr(selected.mineral, field), getattr(baseline.mineral, field))
    assert not np.array_equal(getattr(selected.peat, field), getattr(baseline.peat, field))
    repeated = h.hydrol_runtime_static_tables_from_payload(payload, soil_parameters=soil, peat_hydro=True)
    assert repeated.mineral is baseline.mineral
    assert repeated.peat is baseline.peat


def test_source_inactive_root_threshold_is_metadata_only(table_pair):
    payload, soil, baseline, _ = table_pair
    depth = h.hydrol_depth_parameters({"altmax_for_nroot_thresh": 0.2})
    runtime = h.hydrol_runtime_static_tables_from_payload(
        payload, soil_parameters=soil, depth_parameters=depth, peat_hydro=True,
    )
    assert float(runtime.depth_parameters.altmax_for_nroot_thresh) == 0.2
    assert runtime.mineral is baseline.mineral
    assert runtime.peat is baseline.peat
    normal = h.hydrol_var_init(**_initialization_inputs())
    configured = h.hydrol_var_init(**_initialization_inputs(), depth_parameters=depth)
    np.testing.assert_array_equal(normal.nroot, configured.nroot)
    np.testing.assert_array_equal(normal.humcste_use, configured.humcste_use)


@pytest.mark.parametrize("override,expected_decay", [(None, 1.5), ({"KFACT_DECAY_RATE": 1.25}, 1.25)])
def test_actual_initialization_owner_resolves_run_def_and_explicit_override(override, expected_decay):
    inputs = _initialization_inputs()
    inputs.pop("peat_hydro")
    initialized = h.hydrol_init_pft14_owner(
        **inputs, pref_soil_veg=[3, 3],
        run_def_values={**CUSTOM, "KFACT_DECAY_RATE": "1.5D0", "UNRELATED": "not a depth control"},
        hydrol_var_config=override,
    )
    expected = h.hydrol_depth_parameters({**CUSTOM, "KFACT_DECAY_RATE": expected_decay})
    assert initialized.restart_static.parameters == expected.as_config_values()
    rebuilt = h.hydrol_runtime_static_tables_from_payload(
        {"njsc": [2], "mcs": initialized.restart_static.mcs, "zz_mm": initialized.restart_static.zz},
        soil_parameters=initialized.soil_parameters, depth_parameters=expected, peat_hydro=True,
    )
    np.testing.assert_array_equal(initialized.restart_static.mineral_tables.k_lin, rebuilt.mineral.k_lin)
    np.testing.assert_array_equal(initialized.restart_static.peat_tables.d_lin, rebuilt.peat.d_lin)


def test_first_step_wrapper_rebuilds_real_tables_with_the_explicit_depth_contract(monkeypatch, table_pair):
    payload, soil, _, custom = table_pair
    depth = custom.depth_parameters
    built = []
    real_builder = h.hydrol_runtime_static_tables_from_payload

    def capture_builder(table_payload, **kwargs):
        assert kwargs["depth_parameters"] is depth
        runtime = real_builder(table_payload, **kwargs)
        built.append(runtime)
        return runtime

    class TableBoundaryReached(Exception):
        pass

    def stop_before_module(**kwargs):
        raise TableBoundaryReached

    monkeypatch.setattr(h, "hydrol_runtime_static_tables_from_payload", capture_builder)
    monkeypatch.setattr(h, "hydrol_initial_tmc_from_restart_mc", stop_before_module)
    # The deliberately minimal precall reaches real table construction, then
    # stops before any module/snow/soil step. This is an interface test only.
    precall = h.HydrolFirstStepPrecallAssembly(
        payload={**payload, "soiltile": [[0, 0, 1, 0, 0, 0]], "mc": None, "water2infilt": None},
        missing_inputs=(), enerbil_boundary=None, enerbil_inputs=(), restart_inputs=(),
        slowproc_inputs=(), driver_inputs=(), source_kernel_inputs=(), provenance=(), notes=(),
    )
    with pytest.raises(TableBoundaryReached):
        h.run_hydrol_first_step_module_from_precall(
            precall, humcste=[1.0, 3.0], dz_mm=[1.0, 549.0, 1450.0],
            dh_mm=[275.0, 999.5, 725.0], zz_mm=payload["zz_mm"],
            soil_parameters=soil, depth_parameters=depth, peat_hydro=True,
        )
    assert len(built) == 1
    # The module fixture may outlive an LRU entry when the entire CI group
    # runs. This boundary requires exact table values; immediate cache reuse
    # is checked separately by the scalar-partition tests above.
    for actual, expected in ((built[0].mineral, custom.mineral), (built[0].peat, custom.peat)):
        for field in expected._fields:
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field), err_msg=field)


def test_real_mineral_and_peat_consumers_keep_same_shape_depth_changes_under_jit(table_pair):
    _, _, baseline, custom = table_pair

    def consume(runtime):
        mineral = h.hydrol_soil_coef_mineral_profile_from_tables(
            mc=np.array([[0.25, 0.28, 0.31]]), profil_froz=np.array([[0.0, 0.25, 0.5]]),
            kfact_root=np.ones((1, 3)), tables=runtime.mineral,
            mcr=[h.USDA_MCR[1]], mcs=[h.USDA_MCS[1]],
        )
        peat = h.hydrol_soil_coef_peat_profile_from_tables(
            mc=np.array([[0.35, 0.45, 0.55]]), profil_froz=np.array([[0.0, 0.25, 0.5]]),
            kfact_root=np.ones((1, 3)), tables=runtime.peat,
        )
        return mineral.k, mineral.d, peat.k, peat.d, runtime.depth_parameters.f_ks

    compiled = jax.jit(consume)
    expected = [consume(runtime) for runtime in (baseline, custom)]
    actual = [compiled(runtime) for runtime in (baseline, custom)]
    for eager, jitted in zip(expected, actual):
        for lhs, rhs in zip(eager, jitted):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)
            assert np.isfinite(np.asarray(rhs)).all()
    assert compiled._cache_size() == 1
    for lhs, rhs in zip(actual[0][:4], actual[1][:4]):
        assert not np.array_equal(lhs, rhs)
    assert float(actual[0][4]) == 2.0
    assert float(actual[1][4]) == 1.5
