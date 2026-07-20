import numpy as np
import pytest

from jax_orchidee.stomate.carbon_kernels import IACTIVE, IBELOW, IPASSIVE, ISLOW
from jax_orchidee.stomate.soilcarbon_kernels import soilcarbon_pft14_owner


def _inputs(*, npts=2, nvm=5):
    carbon = np.zeros((npts, 3, nvm), dtype=np.float64)
    soil_input = np.zeros_like(carbon)
    for m in range(nvm):
        carbon[:, :, m] = np.asarray([100.0 + m, 200.0 + m, 300.0 + m])
        soil_input[:, :, m] = np.asarray([1.0, 2.0, 3.0])
    return {
        "dt": 0.5,
        "clay": np.asarray([0.2, 0.6])[:npts],
        "soilcarbon_input": soil_input,
        "control_temp": np.tile(np.asarray([0.9, 0.8]), (npts, 1)),
        "control_moist": np.tile(np.asarray([0.7, 0.5]), (npts, 1)),
        "carbon": carbon,
        "matrix_a": np.zeros((npts, nvm, 7, 7), dtype=np.float64),
        "natural": np.asarray([True, True, False, False, True])[:nvm],
        "is_peat": np.asarray([False, False, False, False, True])[:nvm],
        "is_c4": np.asarray([False, False, False, True, False])[:nvm],
    }


def _classic_reference(args):
    """Literal NumPy transcription of soilcarbon lines 294-436."""

    dt = args["dt"]
    clay = args["clay"]
    carbon = args["carbon"].copy() + args["soilcarbon_input"] * dt
    tau = np.asarray([0.149, 5.48, 241.0]) * 365.0
    frac = np.zeros((clay.size, 3, 3))
    frac[:, IACTIVE, IPASSIVE] = 0.004
    frac[:, IACTIVE, ISLOW] = 1.0 - (0.85 - 0.68 * clay) - 0.004
    frac[:, ISLOW, IACTIVE] = 0.42
    frac[:, ISLOW, IPASSIVE] = 0.03
    frac[:, IPASSIVE, IACTIVE] = 0.45
    frac[:, IPASSIVE, ISLOW] = 0.0
    frac_resp = 1.0 - frac[:, :, IACTIVE] - frac[:, :, ISLOW] - frac[:, :, IPASSIVE]
    controls = args["control_moist"][:, IBELOW] * args["control_temp"][:, IBELOW]
    resp = np.zeros((clay.size, carbon.shape[2]))
    factors = (1.2, 1.4, 0.75)
    for m in range(1, carbon.shape[2]):
        if args["is_peat"][m]:
            continue
        factor = 1.0 if args["natural"][m] else factors[1 if args["is_c4"][m] else 0]
        total = np.zeros((clay.size, 3))
        flux = np.zeros((clay.size, 3, 3))
        for source in range(3):
            total[:, source] = dt / tau[source] * carbon[:, source, m] * controls * factor
            if source == IACTIVE:
                total[:, source] *= 1.0 - factors[2] * clay
            carbon[:, source, m] -= total[:, source]
            for destination in range(3):
                flux[:, source, destination] = frac[:, source, destination] * total[:, source]
        resp[:, m] = np.sum(frac_resp * total, axis=1) / dt
        for destination in range(3):
            carbon[:, destination, m] += (
                flux[:, IACTIVE, destination]
                + flux[:, IPASSIVE, destination]
                + flux[:, ISLOW, destination]
            )
    return carbon, resp, tau, frac


def test_classic_three_pool_owner_matches_source_branches_and_save_state():
    """Fortran: stomate_soilcarbon.f90 soilcarbon lines 294-436."""

    args = _inputs()
    expected_carbon, expected_resp, tau, _ = _classic_reference(args)
    result = soilcarbon_pft14_owner(**args)

    np.testing.assert_allclose(result.carbon, expected_carbon, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(result.resp_hetero_soil, expected_resp, rtol=0.0, atol=2e-16)
    np.testing.assert_array_equal(result.carbon_tau, tau)
    assert result.firstcall_soilcarbon is False
    assert result.height_acro is None
    # The peat column receives line-348 input but skips lines 364-423.
    np.testing.assert_array_equal(result.carbon[:, :, 4], args["carbon"][:, :, 4] + 0.5 * args["soilcarbon_input"][:, :, 4])

    subsequent = soilcarbon_pft14_owner(
        **args,
        firstcall_soilcarbon=False,
        carbon_tau=np.asarray([11.0, 22.0, 33.0]),
    )
    np.testing.assert_array_equal(subsequent.carbon_tau, [11.0, 22.0, 33.0])


def test_save_state_and_fixed_ok_leak_dispatch_fail_explicitly():
    """Fortran: soilcarbon 312-339; stomate_main 3384-3522."""

    args = _inputs()
    with pytest.raises(ValueError, match="312-339"):
        soilcarbon_pft14_owner(**args, firstcall_soilcarbon=False)
    with pytest.raises(ValueError, match="3384-3522"):
        soilcarbon_pft14_owner(**args, ok_leak=True)


def test_analytical_spinup_matrix_preserves_other_entries_and_adds_identity():
    """Fortran: stomate_soilcarbon.f90 soilcarbon lines 442-572."""

    args = _inputs(npts=1, nvm=4)
    args["matrix_a"].fill(9.0)
    result = soilcarbon_pft14_owner(**args, spinup_analytic=True)
    matrix = np.asarray(result.matrix_a)
    tau = np.asarray([0.149, 5.48, 241.0]) * 365.0
    base = 0.5 * 0.5 * 0.8
    active_clay = 1.0 - 0.75 * 0.2
    frac_active_slow = 1.0 - (0.85 - 0.68 * 0.2) - 0.004

    # Bare-soil is not filled by the m=2,nvm loop, but identity is global.
    np.testing.assert_array_equal(np.diag(matrix[0, 0]), np.full(7, 10.0))
    assert matrix[0, 1, 4, 4] == pytest.approx(1.0 - base / tau[IACTIVE] * active_clay)
    assert matrix[0, 1, 5, 4] == pytest.approx(frac_active_slow * base / tau[IACTIVE] * active_clay)
    assert matrix[0, 1, 4, 5] == pytest.approx(0.42 * base / tau[ISLOW])
    # C3 and C4 crop factors multiply all eight source-assigned soil entries.
    assert matrix[0, 2, 4, 4] == pytest.approx(1.0 + (-base / tau[IACTIVE] * active_clay) * 1.2)
    assert matrix[0, 3, 4, 4] == pytest.approx(1.0 + (-base / tau[IACTIVE] * active_clay) * 1.4)
    # An unassigned off-diagonal pool remains untouched.
    assert matrix[0, 2, 0, 1] == 9.0


def test_legacy_peat_owner_covers_all_water_table_branches_and_writeback():
    """Fortran: stomate_soilcarbon.f90 soilcarbon lines 425-433 and 576-630."""

    npts, nvm = 3, 14
    args = _inputs(npts=npts, nvm=5)
    # Build the actual PFT14 column while retaining a non-peat PFT.
    args["carbon"] = np.pad(args["carbon"], ((0, 0), (0, 0), (0, nvm - 5)), constant_values=10.0)
    args["soilcarbon_input"] = np.pad(args["soilcarbon_input"], ((0, 0), (0, 0), (0, nvm - 5)), constant_values=0.0)
    args["soilcarbon_input"][:, IACTIVE, 13] = 2.0
    args["soilcarbon_input"][:, ISLOW, 13] = 3.0
    args["clay"] = np.asarray([0.2, 0.3, 0.4])
    args["control_temp"] = np.asarray([[0.9, 0.8], [0.9, 0.6], [0.9, 0.4]])
    args["control_moist"] = np.full((npts, 2), 0.5)
    args["matrix_a"] = np.zeros((npts, nvm, 7, 7))
    args["natural"] = np.ones(nvm, dtype=bool)
    args["is_peat"] = np.zeros(nvm, dtype=bool)
    args["is_peat"][13] = True
    args["is_c4"] = np.zeros(nvm, dtype=bool)
    height = np.asarray([0.2, 0.2, 0.2])
    # wtd <= 0, 0 < wtd < height, and wtd >= height.
    wtp = np.asarray([400.0, 200.0, 0.0])
    acro = np.full((npts, nvm), 100.0)
    cato = np.full((npts, nvm), 200.0)

    result = soilcarbon_pft14_owner(
        **args,
        ok_peat=True,
        height_acro=height,
        carbon_acro=acro,
        carbon_cato=cato,
        wtp_peat=wtp,
    )
    b = np.asarray([1.0, 0.5, 0.0])
    ka = args["control_temp"][:, IBELOW] * 0.067 * 0.5 / 365.0
    kp = args["control_temp"][:, IBELOW] * 1.91e-2 * 0.5 / 365.0
    kc = 3.35e-5 * 0.5 / 365.0
    oxic = b * ka * 100.0
    anoxic = (1.0 - b) * 0.35 * ka * 100.0
    to_cato = kp * 100.0
    cato_resp = kc * 200.0
    expected_acro = 100.0 + 2.5 - to_cato - oxic - anoxic
    expected_cato = 200.0 + to_cato - cato_resp

    np.testing.assert_allclose(result.resp_acro_oxic[:, 13], oxic)
    np.testing.assert_allclose(result.resp_acro_anoxic[:, 13], anoxic)
    np.testing.assert_allclose(result.carbon_acro[:, 13], expected_acro)
    np.testing.assert_allclose(result.carbon_cato[:, 13], expected_cato)
    np.testing.assert_allclose(result.height_acro, expected_acro / (3.5e4 * 0.5))
    np.testing.assert_allclose(result.height_cato, expected_cato / (9.1e4 * 0.52))
    np.testing.assert_array_equal(result.carbon[:, :, 13], 0.0)
    # Lines 425-433 zero legacy peat state for every non-peat PFT.
    np.testing.assert_array_equal(result.carbon_acro[:, 0], 100.0)
    np.testing.assert_array_equal(result.carbon_cato[:, 0], 200.0)
    np.testing.assert_array_equal(result.carbon_acro[:, 1:13], 0.0)
    np.testing.assert_array_equal(result.carbon_cato[:, 1:13], 0.0)


def test_ok_peat_requires_every_legacy_state_before_mutation():
    """Fortran: stomate_soilcarbon.f90 soilcarbon lines 576-630."""

    with pytest.raises(ValueError, match="576-630"):
        soilcarbon_pft14_owner(**_inputs(), ok_peat=True)
