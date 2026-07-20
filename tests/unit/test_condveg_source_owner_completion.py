import numpy as np
import pytest

from jax_orchidee.sechiba.condveg import (
    CONDVEG_SOILALB_PROVENANCE,
    condveg_main_output_packet,
    condveg_soilalb_from_interpolation,
)


def test_soilalb_uses_all_and_partial_soil_classes_in_source_order():
    fractions = np.array(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.00, 0.25, 0.00, 0.75],
        ],
        dtype=np.float64,
    )
    parameters = {
        "vis_dry": np.array([0.4, 0.3, 0.2, 0.1]),
        "nir_dry": np.array([0.8, 0.6, 0.4, 0.2]),
        "vis_wet": np.array([0.2, 0.15, 0.1, 0.05]),
        "nir_wet": np.array([0.4, 0.3, 0.2, 0.1]),
        "albsoil_vis": np.array([0.3, 0.225, 0.15, 0.075]),
        "albsoil_nir": np.array([0.6, 0.45, 0.3, 0.15]),
    }

    result = condveg_soilalb_from_interpolation(
        soilcolrefrac=fractions,
        asoilcol=np.ones(2),
        **parameters,
    )

    np.testing.assert_allclose(result.soilalb_dry[:, 0], fractions @ parameters["vis_dry"])
    np.testing.assert_allclose(result.soilalb_dry[:, 1], fractions @ parameters["nir_dry"])
    np.testing.assert_allclose(result.soilalb_wet[:, 0], fractions @ parameters["vis_wet"])
    np.testing.assert_allclose(result.soilalb_wet[:, 1], fractions @ parameters["nir_wet"])
    np.testing.assert_allclose(result.soilalb_moy[:, 0], fractions @ parameters["albsoil_vis"])
    np.testing.assert_allclose(result.soilalb_moy[:, 1], fractions @ parameters["albsoil_nir"])
    assert result.fallback_count == 0


def test_soilalb_fallback_matches_fortran_mean_for_no_class_or_low_availability():
    fractions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.3, 0.5],
            [0.2, 0.3, 0.5],
        ],
        dtype=np.float64,
    )
    vis_dry = np.array([0.3, 0.2, 0.1])
    nir_dry = np.array([0.6, 0.4, 0.2])
    vis_wet = np.array([0.15, 0.10, 0.05])
    nir_wet = np.array([0.30, 0.20, 0.10])
    albsoil_vis = np.array([0.225, 0.15, 0.075])
    albsoil_nir = np.array([0.45, 0.30, 0.15])

    result = condveg_soilalb_from_interpolation(
        soilcolrefrac=fractions,
        asoilcol=np.array([1.0, 0.0, 1.0]),
        vis_dry=vis_dry,
        nir_dry=nir_dry,
        vis_wet=vis_wet,
        nir_wet=nir_wet,
        albsoil_vis=albsoil_vis,
        albsoil_nir=albsoil_nir,
    )

    fallback_dry = np.array([(vis_dry.mean() + vis_wet.mean()) / 2.0, (nir_dry.mean() + nir_wet.mean()) / 2.0])
    fallback_moy = np.array([albsoil_vis.mean(), albsoil_nir.mean()])
    np.testing.assert_allclose(result.soilalb_dry[:2], np.broadcast_to(fallback_dry, (2, 2)))
    np.testing.assert_allclose(result.soilalb_wet[:2], np.broadcast_to(fallback_dry, (2, 2)))
    np.testing.assert_allclose(result.soilalb_moy[:2], np.broadcast_to(fallback_moy, (2, 2)))
    np.testing.assert_allclose(result.soilalb_dry[2], fractions[2] @ np.column_stack((vis_dry, nir_dry)))
    assert result.fallback_count == 2


def test_soilalb_zeros_subthreshold_weights_only_on_interpolated_branch():
    result = condveg_soilalb_from_interpolation(
        soilcolrefrac=np.array([[0.5, 1.0e-9, 0.5]], dtype=np.float64),
        asoilcol=np.array([1.0]),
        vis_dry=np.array([0.3, 1000.0, 0.1]),
        nir_dry=np.array([0.6, 1000.0, 0.2]),
        vis_wet=np.array([0.15, 1000.0, 0.05]),
        nir_wet=np.array([0.30, 1000.0, 0.10]),
        albsoil_vis=np.array([0.225, 1000.0, 0.075]),
        albsoil_nir=np.array([0.45, 1000.0, 0.15]),
    )

    np.testing.assert_allclose(result.soilalb_dry, [[0.2, 0.4]])
    np.testing.assert_allclose(result.soilalb_wet, [[0.1, 0.2]])
    np.testing.assert_allclose(result.soilalb_moy, [[0.15, 0.3]])


@pytest.mark.parametrize(
    ("fractions", "availability", "message"),
    [
        (np.ones(3), np.ones(1), "soilcolrefrac"),
        (np.ones((2, 3)), np.ones(3), "asoilcol"),
    ],
)
def test_soilalb_rejects_invalid_interpolation_contract(fractions, availability, message):
    with pytest.raises(ValueError, match=message):
        condveg_soilalb_from_interpolation(
            soilcolrefrac=fractions,
            asoilcol=availability,
            vis_dry=np.ones(3),
            nir_dry=np.ones(3),
            vis_wet=np.ones(3),
            nir_wet=np.ones(3),
            albsoil_vis=np.ones(3),
            albsoil_nir=np.ones(3),
        )


def test_soilalb_has_precise_source_provenance():
    assert any("condveg_soilalb lines 932-1131" in item for item in CONDVEG_SOILALB_PROVENANCE)
    assert any("constantes_var.f90 lines 736-761" in item for item in CONDVEG_SOILALB_PROVENANCE)


def _diagnostic_inputs():
    return {
        "alb_bare": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "alb_veget": np.array([[0.05, 0.25], [0.07, 0.27]]),
        "albedo": np.array([[0.12, 0.24], [0.32, 0.44]]),
        "albedo_snow": np.array([[0.8, 0.6], [0.5, 0.3]]),
        "snow": np.array([0.0, 2.0]),
    }


def test_condveg_main_non_alma_packet_preserves_source_field_order_and_hist2_gate():
    packet = condveg_main_output_packet(**_diagnostic_inputs(), almaoutput=False, hist2_id=4)

    assert [name for name, _ in packet.xios] == [
        "soilalb_vis",
        "soilalb_nir",
        "vegalb_vis",
        "vegalb_nir",
        "albedo_vis",
        "albedo_nir",
        "albedo_snow",
    ]
    assert [name for name, _ in packet.history_primary] == [
        "soilalb_vis",
        "soilalb_nir",
        "vegalb_vis",
        "vegalb_nir",
    ]
    assert [name for name, _ in packet.history_secondary] == [name for name, _ in packet.history_primary]
    np.testing.assert_allclose(packet.xios[-1][1], [0.0, 0.4])


def test_condveg_main_alma_packet_uses_band_means_and_can_disable_hist2():
    inputs = _diagnostic_inputs()
    packet = condveg_main_output_packet(**inputs, almaoutput=True, hist2_id=0)

    assert [name for name, _ in packet.history_primary] == ["Albedo", "SAlbedo"]
    np.testing.assert_allclose(packet.history_primary[0][1], inputs["albedo"].mean(axis=1))
    np.testing.assert_allclose(packet.history_primary[1][1], inputs["albedo_snow"].mean(axis=1))
    assert packet.history_secondary == ()
