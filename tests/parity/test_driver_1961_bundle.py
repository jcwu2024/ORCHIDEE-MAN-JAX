from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.bundle import (
    _static_fields_with_local_inputs,
    iter_paper_year_step_bundles,
    load_paper_1961_first_step_bundle,
    load_paper_1961_step_bundle,
)
from jax_orchidee.driver.domain import DomainGrid
from jax_orchidee.driver.orchestration import prepare_paper_1961_driver_context
from jax_orchidee.driver.run_def_materialization import materialize_case_run_def_values, write_materialized_run_def
from jax_orchidee.driver.trace import StaticTraceFields


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
USED_RUN_DEF = ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"
REFERENCE_CASE_RUN_DEF = (
    ROOT
    / "reference"
    / "case_001_071"
    / "OUT"
    / "orc_calibrate_250919_sen"
    / "arg2_1.0"
    / "001.0-071.0"
    / "I10"
    / "S2_63.206_0.0876_0.2019_50.658"
    / "run.def"
)


def test_paper_1961_first_step_bundle_composes_verified_domain_forcing_and_co2():
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961)

    assert bundle.year == 1961
    assert bundle.domain.nbindex == 1
    assert np.allclose(bundle.domain.lon, [[109.0]])
    assert np.allclose(bundle.domain.lat, [[21.0]])
    assert np.allclose(bundle.domain.lalo, [[21.0, 109.0]])
    assert np.allclose(bundle.domain.contfrac_land, [0.5625])
    assert np.allclose(bundle.domain.resolution, [[111106.370838091, 111111.0]])
    assert np.array_equal(bundle.domain.neighbours, np.full((1, 8), -1, dtype=np.int32))
    assert np.allclose(bundle.domain.area, [12345139970.1911])

    assert np.allclose(bundle.forcing.Tair, [[291.3238830566406]])
    assert np.allclose(bundle.forcing.PSurf, [[98422.8671875]])
    assert np.allclose(bundle.forcing.Qair, [[0.008104387670755386]])
    assert np.allclose(bundle.forcing.Wind_E, [[-3.8484580516815186]])
    assert np.allclose(bundle.forcing.Wind_N, [[-0.8622167110443115]])
    assert np.allclose(bundle.forcing.SWdown, [[362.3620910644531]])
    assert np.allclose(bundle.forcing.LWdown, [[338.0463562011719]])

    assert bundle.co2_ppm == 317.27
    assert bundle.ccanopy.shape == (1,)
    assert np.allclose(bundle.ccanopy, [317.27])


def test_paper_1961_step_bundle_reuses_static_sources_and_advances_forcing_only():
    first = load_paper_1961_first_step_bundle(CONFIG, year=1961)
    step0 = load_paper_1961_step_bundle(CONFIG, year=1961, tstep=0)
    step1 = load_paper_1961_step_bundle(CONFIG, year=1961, tstep=1)

    assert step0.year == first.year
    assert step0.tstep == 0
    assert np.allclose(step0.forcing.Tair, [[289.76685333251953]])
    assert np.allclose(step0.forcing.Qair, [[0.009809188079088926]])
    assert np.allclose(step0.forcing.SWdown, [[119.53496445377026]])
    assert np.allclose(step0.forcing.zlevuv, [[10.0]])
    assert step1.tstep == 1
    assert np.allclose(step1.forcing.Tair, [[289.9084014892578]])
    assert np.allclose(step1.forcing.Rainf, [[0.0]])
    assert np.allclose(step1.forcing.zlev, [[2.0]])
    assert np.allclose(step1.forcing.SWdown, [[186.07451159425887]])
    assert np.allclose(step1.forcing.zlevuv, [[10.0]])
    assert np.allclose(step1.ccanopy, first.ccanopy)
    assert np.allclose(step1.vegetation.veget_max, first.vegetation.veget_max)
    assert np.array_equal(step1.run_scalars.pref_soil_veg, first.run_scalars.pref_soil_veg)
    assert step1.restart_anchors is not None
    assert first.restart_anchors is not None
    assert np.array_equal(step1.restart_anchors.njsc, first.restart_anchors.njsc)
    assert np.allclose(step1.restart_anchors.clay_frac, first.restart_anchors.clay_frac)
    assert step1.missing_static_fields == first.missing_static_fields


def test_paper_1961_year_step_iterator_yields_ordered_driver_boundary_only():
    steps = list(iter_paper_year_step_bundles(CONFIG, year=1961, nsteps=2))

    assert [step.tstep for step in steps] == [0, 1]
    assert np.allclose(steps[0].forcing.Tair, [[289.76685333251953]])
    assert np.allclose(steps[1].forcing.Tair, [[289.9084014892578]])
    assert steps[0].domain is steps[1].domain
    assert steps[0].run_scalars is steps[1].run_scalars
    assert steps[0].vegetation is steps[1].vegetation
    assert steps[0].missing_geometry_fields == steps[1].missing_geometry_fields


def test_paper_1961_first_step_bundle_contains_verified_init_and_restart_anchors():
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961)

    assert bundle.run_scalars.nvm == 14
    assert bundle.run_scalars.nstm == 6
    assert bundle.run_scalars.pref_soil_veg[13] == 4
    assert np.allclose(bundle.run_scalars.ext_coeff_vegetfrac, np.ones(14))
    assert np.allclose(bundle.run_scalars.slowproc_height[13], 30.0)

    assert bundle.vegetation.veget_max.shape == (1, 14)
    assert bundle.vegetation.veget.shape == (1, 14)
    assert np.allclose(bundle.vegetation.veget_max[:, :13], 0.0)
    assert np.allclose(bundle.vegetation.veget_max[:, 13], 1.0)
    assert np.allclose(bundle.vegetation.veget[:, :13], 0.0)
    assert np.allclose(bundle.vegetation.veget[:, 13], 1.0)
    assert np.allclose(bundle.vegetation.soiltile, [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])

    assert bundle.restart_anchors is not None
    assert np.array_equal(bundle.restart_anchors.njsc, np.asarray([2], dtype=np.int32))
    assert np.allclose(bundle.restart_anchors.clay_frac, [0.2])
    assert np.allclose(bundle.restart_anchors.sand_frac, [0.4])
    assert np.allclose(bundle.restart_anchors.bulk_dens, [1650.0])
    assert np.allclose(bundle.restart_anchors.soil_ph, [7.0])
    assert np.allclose(bundle.restart_anchors.poor_soils, [0.0])
    assert np.allclose(bundle.restart_anchors.wtp, [0.0])
    assert np.allclose(bundle.restart_anchors.wt_ab_tide, [0.0])


def test_paper_1961_first_step_bundle_marks_missing_geometry_and_static_truth():
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961)

    assert "resolution" not in bundle.missing_geometry_fields
    assert "neighbours" not in bundle.missing_geometry_fields
    assert "corners" not in bundle.missing_geometry_fields
    assert "seglength" not in bundle.missing_geometry_fields
    assert "soilclass" not in bundle.missing_static_fields
    assert "njsc" not in bundle.missing_static_fields
    assert "clay_frac" not in bundle.missing_static_fields
    assert "sand_frac" not in bundle.missing_static_fields
    assert "silt_frac" not in bundle.missing_static_fields
    assert "bulk_dens" not in bundle.missing_static_fields
    assert "soil_ph" not in bundle.missing_static_fields
    assert "poor_soils" not in bundle.missing_static_fields
    assert "soilclass_sub_index" not in bundle.missing_static_fields
    assert "soilclass_sub_area" not in bundle.missing_static_fields
    assert "salinity" not in bundle.missing_static_fields
    assert "tide_height" not in bundle.missing_static_fields
    assert "veget" not in bundle.missing_static_fields
    assert bundle.static_trace_fields.salinity is not None
    assert bundle.static_trace_fields.tide_height is not None
    assert bundle.static_trace_fields.tide_height.shape == (1, 584)
    assert bundle.static_trace_fields.soilclass is not None
    assert bundle.static_trace_fields.soilclass_sub_index is not None
    assert bundle.static_trace_fields.soilclass_sub_area is not None
    assert bundle.static_trace_fields.njsc is not None
    assert np.array_equal(bundle.static_trace_fields.njsc, np.asarray([5], dtype=np.int32))

    assert not hasattr(bundle, "resolution")
    assert not hasattr(bundle, "neighbours")
    assert not hasattr(bundle, "soilclass")
    assert not hasattr(bundle, "salinity")
    assert not hasattr(bundle, "tide_height")


def test_static_local_input_preflight_requires_resolution_for_missing_slowproc_fields():
    domain = DomainGrid(
        iim=1,
        jjm=1,
        nbindex=1,
        kindex=np.asarray([1], dtype=np.int32),
        lon=np.asarray([[109.0]], dtype=np.float64),
        lat=np.asarray([[21.0]], dtype=np.float64),
        lalo=np.asarray([[21.0, 109.0]], dtype=np.float64),
        contfrac=np.asarray([1.0], dtype=np.float64),
        contfrac_land=np.asarray([1.0], dtype=np.float64),
        area=np.asarray([1.0], dtype=np.float64),
        forcing_indices_zero_based=np.asarray([0], dtype=np.int32),
        resolution=None,
    )

    with pytest.raises(ValueError, match="explicit driver resolution"):
        _static_fields_with_local_inputs(
            CONFIG,
            domain=domain,
            static_fields=StaticTraceFields(),
        )


def test_paper_1961_first_step_bundle_keeps_water_table_sequences_as_inputs():
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961)

    assert bundle.water_table.positive.shape == (17520,)
    assert bundle.water_table.differential.shape == (17520,)
    assert np.allclose(bundle.water_table.positive[:5], [308.33, 329.00, 349.67, 347.92, 346.17])
    assert np.allclose(bundle.water_table.differential[:5], [44.83, 20.67, 20.67, -1.75, -1.75])


def test_paper_1961_first_step_bundle_uses_explicit_run_def_for_run_scalars(tmp_path):
    custom_run_def = tmp_path / "used_run_modified.def"
    text = USED_RUN_DEF.read_text(encoding="utf-8")
    text = re.sub(
        r"(SLOWPROC_HEIGHT__00014\s*=\s*)[^\r\n]+",
        r"\g<1>31.5000000000000     ",
        text,
        count=1,
    )
    custom_run_def.write_text(text, encoding="utf-8")

    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961, run_def_path=custom_run_def)

    assert np.allclose(bundle.run_scalars.slowproc_height[13], 31.5)


def test_materialized_reference_case_run_def_prepares_case_specific_driver_context(tmp_path):
    materialized = write_materialized_run_def(
        materialize_case_run_def_values(base_used_run_def=USED_RUN_DEF, case_run_def=REFERENCE_CASE_RUN_DEF),
        tmp_path / "case_used.def",
    )

    context = prepare_paper_1961_driver_context(
        CONFIG,
        used_run_def_path=materialized,
        reference_run_dir=REFERENCE_CASE_RUN_DEF.parent,
    )

    assert context.domain_override == {"west": -180.0, "east": -178.0, "south": -20.0, "north": -18.0}
    assert context.reference_run_dir == REFERENCE_CASE_RUN_DEF.parent
    assert context.first_step_restart_state.run_dir == REFERENCE_CASE_RUN_DEF.parent
    assert context.first_step_stomate_boundary.stomate_files.run_dir == REFERENCE_CASE_RUN_DEF.parent
    assert context.first_step_bundle.domain.nbindex == 1
    assert np.allclose(context.first_step_bundle.domain.lalo, [[-19.0, -179.0]])
    assert np.allclose(context.first_step_bundle.run_scalars.slowproc_height[13], 30.0)
