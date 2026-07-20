from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.stomate.carbon_kernels import ICARBRES, ILEAF
from jax_orchidee.stomate.source_helpers import (
    DATA_PROVENANCE,
    SORT_ASCENDING_PROVENANCE,
    STOMATE_INIT_PROVENANCE,
    StomateDataConstants,
    sort_ascending_owner,
    stomate_data_owner,
    stomate_init_lifecycle_owner,
)


def _constants():
    return StomateDataConstants(
        alpha_tree=4.0,
        alpha_grass=2.0,
        bm_sapl_leaf=(1.2, 0.7, 0.5, 1.1),
        pipe_tune1=1.3,
        mass_ratio_heart_sap=2.0,
        pipe_k1=0.4,
        bm_sapl_carbres=0.25,
        bm_sapl_sapabove=1.4,
        pipe_density=0.8,
        pipe_tune2=1.1,
        pipe_tune3=2.5,
        dia_coeff=(0.9, 0.5),
        bm_sapl_heartabove=1.7,
        bm_sapl_heartbelow=1.8,
        init_sapl_mass_leaf_nat=3.0,
        init_sapl_mass_leaf_agri=1.5,
        init_sapl_mass_carbres=0.2,
        init_sapl_mass_root=0.6,
        init_sapl_mass_fruit=0.1,
        migrate_tree=12.0,
        migrate_grass=99.0,
        pipe_tune4=3.0,
        maxdia_coeff=(1.2, 0.7),
        cn_sapl_init=0.03,
        lai_initmin_tree=0.4,
        lai_initmin_grass=0.2,
        undef=-9999.0,
        min_stomate=1.0e-8,
    )


def _data_inputs(nvm=5):
    return {
        "bm_sapl": np.zeros((nvm, 12, 1)),
        "migrate": np.full(nvm, -7.0),
        "maxdia": np.full(nvm, -7.0),
        "cn_sapl": np.full(nvm, -7.0),
        "lai_initmin": np.full(nvm, -7.0),
        "sla": np.asarray([1.0, 2.0, 4.0, 5.0, 3.0])[:nvm],
        "is_tree": np.asarray([False, True, False, False, True])[:nvm],
        "pheno_type": np.asarray([1, 1, 2, 2, 2])[:nvm],
        "ok_laidev": np.asarray([False, False, False, True, False])[:nvm],
        "natural": np.asarray([False, True, True, False, True])[:nvm],
        "is_grassland_manag": np.asarray([False, False, False, False, False])[:nvm],
        "sp_densitesem": np.asarray([0.0, 0.0, 0.0, 8.0, 0.0])[:nvm],
        "sp_pgrainmaxi": np.asarray([0.0, 0.0, 0.0, 0.5, 0.0])[:nvm],
        "tmin_crit": np.asarray([-9999.0, -10.0, -9999.0, 2.0, -20.0])[:nvm],
        "tcm_crit": np.asarray([-9999.0, 3.0, -9999.0, 4.0, 5.0])[:nvm],
        "pheno_model": np.asarray(["none", "none", "none", "none", "moigdd"], dtype=object)[:nvm],
        "pheno_gdd_crit": np.ones((nvm, 3)),
        "senescence_type": np.asarray(["none", "none", "dry", "none", "cold"], dtype=object)[:nvm],
        "senescence_temp": np.ones((nvm, 3)),
        "senescence_hum": np.ones(nvm),
        "nosenescence_hum": np.ones(nvm),
        "leafagecrit": np.arange(nvm, dtype=float) * 4.0,
        "nleafages": 4,
        "ok_dgvm": False,
        "use_age_class": False,
        "constants": _constants(),
    }


def test_data_owner_materializes_tree_grass_crop_and_preserves_bare_soil():
    inputs = _data_inputs()
    result = stomate_data_owner(**inputs)
    bm = np.asarray(result.bm_sapl)

    assert bm[1, ILEAF, 0] > 0.0
    assert bm[1, ICARBRES, 0] == 0.0  # evergreen tree, lines 275-280
    assert bm[2, ILEAF, 0] == pytest.approx(3.0 / 4.0)  # natural grass
    assert bm[3, ILEAF, 0] == 0.0  # LAIdev crop
    assert bm[3, ICARBRES, 0] == pytest.approx(4.0)
    np.testing.assert_array_equal(bm[0], 0.0)
    np.testing.assert_allclose(result.migrate[1:], [12.0, 99.0, 99.0, 12.0])
    np.testing.assert_allclose(result.lai_initmin[1:], [0.4, 0.2, 0.2, 0.4])
    assert result.maxdia[2] == -9999.0
    assert result.cn_sapl[2] == 1.0
    assert result.tmin_crit[1] == pytest.approx(263.15)
    assert result.tmin_crit[2] == -9999.0
    np.testing.assert_allclose(result.leaf_timecst, np.arange(5, dtype=float))
    assert "data lines 261-348" in DATA_PROVENANCE[0]


def test_data_owner_age_class_scaling_and_configuration_guards():
    inputs = _data_inputs()
    baseline = stomate_data_owner(**inputs)
    inputs["use_age_class"] = True
    scaled = stomate_data_owner(**inputs)
    np.testing.assert_allclose(np.asarray(scaled.bm_sapl)[1:, :, 0], np.asarray(baseline.bm_sapl)[1:, :, 0] * 0.05)

    inputs = _data_inputs()
    inputs["natural"][3] = True
    with pytest.raises(ValueError, match="ok_LAIdev and natural"):
        stomate_data_owner(**inputs)


@pytest.mark.parametrize(
    "field,index,message",
    [
        ("pheno_gdd_crit", (4, 1), "phenology GDD"),
        ("senescence_temp", (4, 1), "senescence temperature"),
        ("senescence_hum", 2, "senescence_hum"),
        ("nosenescence_hum", 2, "nosenescence_hum"),
    ],
)
def test_data_owner_reproduces_source_parameter_error_guards(field, index, message):
    inputs = _data_inputs()
    inputs[field][index] = -9999.0
    with pytest.raises(ValueError, match=message):
        stomate_data_owner(**inputs)


def test_sort_ascending_reproduces_strict_less_selection_sort_and_nan_position():
    values = np.asarray([[3.0, 1.0, 2.0], [np.nan, 1.0, 0.0]])
    result = np.asarray(sort_ascending_owner(values))
    np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0])
    assert np.isnan(result[1, 0])
    np.testing.assert_allclose(result[1, 1:], [0.0, 1.0])
    assert "sort_ascending lines 12278-12311" in SORT_ASCENDING_PROVENANCE[0]


def test_stomate_init_owner_validates_flags_and_initializes_exact_leak_shapes():
    result = stomate_init_lifecycle_owner(
        npts=2, nvm=17, ndeep=3, npool=4, nelements=2, nlitt=2,
        nstm=5, nslm=6, nflow=2, ncarb=3, diagnostic_index_fortran=9,
    )
    assert result.diagnostic_index_fortran == 2
    assert result.leak_daily_state["soilcarbon_input_DOC_daily"].shape == (2, 17, 3, 4, 2)
    assert result.leak_daily_state["biomass_Cforcing_daily"].shape == (2, 17, 12, 2)
    assert all(np.all(np.asarray(value) == 0.0) for value in result.leak_daily_state.values())
    assert "stomate_init lines 6296-6330" in STOMATE_INIT_PROVENANCE[0]

    with pytest.raises(ValueError, match="DGVM requires STOMATE"):
        stomate_init_lifecycle_owner(
            npts=1, nvm=14, ndeep=1, npool=1, nelements=1, nlitt=1,
            nstm=1, nslm=1, nflow=1, ncarb=1, ok_stomate=False, ok_dgvm=True,
        )
    with pytest.raises(ValueError, match="requires CO2"):
        stomate_init_lifecycle_owner(
            npts=1, nvm=14, ndeep=1, npool=1, nelements=1, nlitt=1,
            nstm=1, nslm=1, nflow=1, ncarb=1, ok_co2=False,
        )

