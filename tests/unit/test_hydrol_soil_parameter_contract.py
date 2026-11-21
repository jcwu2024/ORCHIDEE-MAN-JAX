import pickle
from dataclasses import FrozenInstanceError

import jax
import numpy as np
import pytest

from jax_orchidee.sechiba import hydrol as h


def _custom_parameters():
    fields = {name: np.array(value, copy=True) for name, value in h.hydrol_soil_parameters().items()}
    overrides = {
        "nvan": 2.10, "avan": 0.009, "mcr": 0.06, "mcs_mineral": 0.45,
        "ks": 2800.0, "pcent": 0.70, "mcf_mineral": 0.30,
        "mcw_mineral": 0.12, "mc_awet": 0.32, "mc_adry": 0.08,
    }
    for name, value in overrides.items():
        fields[name][1] = value
    return h.hydrol_soil_parameters(fields)


def test_soil_parameter_contract_preserves_defaults_and_serialization():
    contract = h.hydrol_soil_parameters()
    names = ("nvan", "avan", "mcr", "mcs_mineral", "ks", "pcent", "mcf_mineral", "mcw_mineral", "mc_awet", "mc_adry")
    defaults = (h.USDA_NVAN, h.USDA_AVAN, h.USDA_MCR, h.USDA_MCS, h.USDA_KS, h.USDA_PCENT, h.USDA_MCF, h.USDA_MCW, h.USDA_MC_AWET, h.USDA_MC_ADRY)
    assert tuple(contract) == names
    for name, expected in zip(names, defaults):
        np.testing.assert_array_equal(contract[name], expected)
    assert h.hydrol_soil_parameters(contract) is contract
    with pytest.raises(FrozenInstanceError):
        contract.ks = contract.ks * 2.0
    leaves, structure = jax.tree_util.tree_flatten(contract)
    assert len(leaves) == 10
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    decoded = pickle.loads(pickle.dumps(restored))
    assert type(decoded) is h.HydrolSoilParameters
    for name in contract:
        np.testing.assert_array_equal(decoded[name], contract[name])
    # Existing callers and archived two-argument runtime containers remain valid.
    legacy = h.HydrolRuntimeStaticTables(None, None)
    assert legacy.soil_parameters is None
    assert type(pickle.loads(pickle.dumps(legacy))) is h.HydrolRuntimeStaticTables


def test_soil_parameter_run_def_resolves_all_fields_and_source_guards():
    custom = _custom_parameters()
    config_names = (
        "CWRR_N_VANGENUCHTEN", "CWRR_A_VANGENUCHTEN", "VWC_RESIDUAL", "VWC_SAT",
        "CWRR_KS", "WETNESS_TRANSPIR_MAX", "VWC_FC", "VWC_WP",
        "VWC_MIN_FOR_WET_ALB", "VWC_MAX_FOR_DRY_ALB",
    )
    values = {}
    for config_name, field in zip(config_names, custom):
        values[config_name] = ", ".join(str(float(value)) for value in custom[field])
        values[f"{config_name}__00002"] = str(float(custom[field][1]))
    selected = h.hydrol_soil_parameters_from_run_def(values)
    for field in custom:
        np.testing.assert_array_equal(selected[field], custom[field])
    np.testing.assert_array_equal(h.hydrol_soil_parameters_from_run_def({"VWC_MIN_FOR_WET_ALB": "12*3.2D-1"}).mc_awet, 0.32)
    assert float(h.hydrol_soil_parameters_from_run_def({"VWC_MIN_FOR_WET_ALB": "0.4", "VWC_MIN_FOR_WET_ALB__00002": "0.32"}).mc_awet[1]) == 0.32
    with pytest.raises(ValueError, match="unsupported hydrol_init soil"):
        h.hydrol_soil_parameters({"effective_mcs": np.ones(12)})
    with pytest.raises(ValueError, match="shape"):
        h.hydrol_soil_parameters({"ks": np.ones(2)})
    with pytest.raises(ValueError, match="VWC_WP"):
        h.hydrol_soil_parameters({"mcw_mineral": np.ones(12)})
    with pytest.raises(ValueError, match="one or twelve"):
        h.hydrol_soil_parameters_from_run_def({"CWRR_KS": "1, 2"})


def test_refsoc_keeps_effective_point_thresholds_separate_from_background():
    contract = _custom_parameters()
    plain = h.hydrol_refsoc_1d_thresholds(njsc=[2], use_refSOC_hydrol=False, soil_parameters=contract)
    np.testing.assert_array_equal(plain.mcs, [0.45])
    np.testing.assert_array_equal(plain.mcf, [0.30])
    np.testing.assert_array_equal(plain.mcw, [0.12])
    refsoc = 0.4 * h.SOILC_MAX
    mixed = h.hydrol_refsoc_1d_thresholds(njsc=[2], refSOC_1d=[refsoc], soil_parameters=contract)
    saturation = 0.4 * h.POROS_ORG + 0.6 * 0.45
    # hydrol_var_init4071-4088 uses fixed VG_* for refSOC, independently of
    # the configurable nvan/avan CWRR and thermodynamic-freezing parameters.
    def expected(psi):
        return 0.06 + (saturation - 0.06) / (1.0 + (h.USDA_AVAN[1] * psi) ** h.USDA_NVAN[1]) ** h.USDA_VG_M[1]
    np.testing.assert_allclose(mixed.mcs, [saturation], rtol=1e-14)
    np.testing.assert_allclose(mixed.mcw, [expected(h.USDA_VG_PSI_WP[1])], rtol=1e-14)
    np.testing.assert_allclose(mixed.mcf, [expected(h.USDA_VG_PSI_FC[1])], rtol=1e-14)
    np.testing.assert_array_equal(contract.mcs_mineral[1], 0.45)
    np.testing.assert_array_equal(contract.mcf_mineral[1], 0.30)
    np.testing.assert_array_equal(contract.mcw_mineral[1], 0.12)


def test_initialization_uses_the_same_contract_for_thresholds_and_tables():
    soil = _custom_parameters()
    znh = np.array([0.001, 0.01, 0.03, 0.06, 0.12, 0.4, 2.0])
    dnh = np.concatenate((znh[:1], np.diff(znh)))
    dlh = np.concatenate(((dnh[:-1] + dnh[1:]) / 2.0, dnh[-1:] / 2.0))
    result = h.hydrol_var_init(
        veget=[[0.0, 1.0]], veget_max=[[0.0, 1.0]], soiltile=[[1.0]],
        njsc=[2], znh_m=znh, dnh_m=dnh, dlh_m=dlh,
        humcste=[0.0, 3.0], altmax=[[0.0, 0.0]], soil_parameters=soil,
    )
    np.testing.assert_array_equal(result.mcs, [0.45])
    np.testing.assert_array_equal(result.mcf, [0.30])
    np.testing.assert_array_equal(result.mcw, [0.12])
    np.testing.assert_allclose(result.mineral_tables.mc_lin[0], [0.06], rtol=1e-14)
    np.testing.assert_allclose(result.mineral_tables.mc_lin[-1], [0.45], rtol=1e-14)
    litter = h.hydrol_litter_top_diagnostics(
        mc=result.mc, dz_mm=result.dz, dh_mm=result.dh, njsc=[2],
        mcr=soil.mcr, mcs=result.mcs, mcf=result.mcf, mcw=result.mcw,
        mc_awet=soil.mc_awet, mc_adry=soil.mc_adry,
    )
    for name in ("tmc_litter_res", "tmc_litter_wilt", "tmc_litter_field", "tmc_litter_sat", "tmc_litter_awet", "tmc_litter_adry"):
        np.testing.assert_allclose(getattr(result, name), getattr(litter, name), rtol=1e-14)


def test_hydrol_main_forwards_its_bound_soil_contract(monkeypatch):
    from tests.unit.test_hydrol_main import _bucket_main_inputs

    soil = _custom_parameters()
    module, diagnostic, snow, irrigation, routing = _bucket_main_inputs()
    module.update(soil_parameters=soil, mcr=np.array([0.06]), mcs=np.array([0.45]))
    module["mineral_tables"] = h.hydrol_runtime_static_tables_from_payload(
        {"njsc": module["njsc"], "mcs": module["mcs"], "zz_mm": module["zz_mm"]},
        soil_parameters=soil,
    ).mineral
    diagnostic.update(mcs=module["mcs"], mineral_tables=module["mineral_tables"])
    root_factor = h.hydrol_kfact_root_from_vegupd
    run_soil = h.hydrol_module_explicit_step
    captured = {}

    def capture_root_factor(**kwargs):
        captured["ks"] = kwargs["ks"]
        return root_factor(**kwargs)

    def capture_soil(**kwargs):
        captured["point_ks"] = kwargs["ks"]
        return run_soil(**kwargs)

    monkeypatch.setattr(h, "hydrol_kfact_root_from_vegupd", capture_root_factor)
    monkeypatch.setattr(h, "hydrol_module_explicit_step", capture_soil)
    result = h.hydrol_main_step(
        module_inputs=module, diagnostic_inputs=diagnostic, snow_inputs=snow,
        irrigation_inputs=irrigation, routing_inputs=routing,
        humcste=np.ones(module["veget_max"].shape[1]), dlh=diagnostic["dh_mm"],
        tmcs=np.full(module["soiltile"].shape, 900.0), itopmax=2, zmaxh_m=2.0,
        ok_explicitsnow=False,
        alma_inputs={"tot_watveg_beg": np.array([0.1]), "tot_watsoil_beg": np.array([10.0]),
                     "snow_beg": np.array([3.0]), "mx_eau_var": np.array([900.0])},
    )
    assert captured["ks"] is soil.ks
    np.testing.assert_array_equal(captured["point_ks"], np.asarray(soil.ks)[module["njsc"] - 1])
    weights = np.array([module["dz_mm"][1] / 2.0, *(module["dz_mm"][1:-1] + module["dz_mm"][2:]) / 2.0, module["dz_mm"][-1] / 2.0])
    np.testing.assert_allclose(result.diagnostics.layer_by_tile[0].smw_tmp, 0.12 * weights[None, :])
    np.testing.assert_allclose(result.diagnostics.layer_by_tile[0].smf_tmp, 0.30 * weights[None, :])
