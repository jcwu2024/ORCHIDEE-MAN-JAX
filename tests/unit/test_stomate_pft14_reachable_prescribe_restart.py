import numpy as np

from jax_orchidee.stomate.carbon_kernels import (
    ICARBON,
    NLEAFAGES,
    NPARTS,
    prescribe_step,
)
from jax_orchidee.coupled import stomate_restart_input_bundles


def _pft14_empty_positive_cover_inputs():
    """Build the landpoint branch that distinguishes restart from cold start."""

    npts = 1
    nvm = 14
    nelements = 1
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[:, 0] = 0.4
    veget_max[:, 13] = 0.6
    biomass = np.zeros((npts, nvm, NPARTS, nelements), dtype=np.float64)
    bm_sapl = np.zeros((nvm, NPARTS, nelements), dtype=np.float64)
    bm_sapl[13, :, ICARBON] = np.arange(1.0, NPARTS + 1.0)
    return {
        "veget_max": veget_max,
        "dt_days": 1.0,
        "pft_present": np.zeros((npts, nvm), dtype=bool),
        "everywhere": np.zeros((npts, nvm), dtype=np.float64),
        "when_growthinit": np.zeros((npts, nvm), dtype=np.float64),
        "biomass": biomass,
        "leaf_frac": np.zeros((npts, nvm, NLEAFAGES), dtype=np.float64),
        "ind": np.zeros((npts, nvm), dtype=np.float64),
        "cn_ind": np.zeros((npts, nvm), dtype=np.float64),
        "co2_to_bm": np.zeros((npts, nvm), dtype=np.float64),
        "natural": np.ones((nvm,), dtype=bool),
        "pasture": np.zeros((nvm,), dtype=bool),
        "is_tree": np.ones((nvm,), dtype=bool),
        "bm_sapl": bm_sapl,
        "maxdia": np.ones((nvm,), dtype=np.float64),
        "pheno_is_none": np.ones((nvm,), dtype=bool),
        "ok_dgvm": False,
        "lpj_gap_const_mort": True,
        "firstcall": True,
    }


def test_pft14_restart_default_does_not_inject_cold_start_biomass():
    """Restart entry must skip prescribe's ``stom_restname_in == NONE`` block.

    Fortran provenance: ``src_stomate/stomate_prescribe.f90::prescribe``
    lines 257-346.  The paper reference run uses
    ``STOMATE_RESTART_FILEIN=stomate_start.nc``; positive PFT14 cover and an
    empty local pool therefore must not activate the no-restart initializer.
    """

    result = prescribe_step(**_pft14_empty_positive_cover_inputs())

    np.testing.assert_array_equal(np.asarray(result.biomass[:, 13]), 0.0)
    np.testing.assert_array_equal(np.asarray(result.co2_to_bm[:, 13]), 0.0)
    np.testing.assert_array_equal(np.asarray(result.pft_present[:, 13]), False)


def test_pft14_explicit_no_restart_still_runs_cold_start_initializer():
    """The same landpoint initializes only when the no-restart gate is explicit."""

    inputs = _pft14_empty_positive_cover_inputs()
    result = prescribe_step(**inputs, stomate_restart_none=True)

    assert np.sum(np.asarray(result.biomass[:, 13, :, ICARBON])) > 0.0
    assert np.asarray(result.co2_to_bm)[0, 13] > 0.0
    assert bool(np.asarray(result.pft_present)[0, 13])


def test_stomate_bundle_contract_forwards_explicit_restart_mode():
    """Production bundle construction must not depend on prescribe's default."""

    import inspect

    source = inspect.getsource(stomate_restart_input_bundles)
    assert 'if bool(stomate_restart_none):' in source
    assert 'prescribe_inputs["stomate_restart_none"] = True' in source
