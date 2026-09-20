import numpy as np

from scripts.hpc.probe_mixed_carbon_endpoints import compact_carbon_sum, endpoint_terms


def test_endpoint_terms_preserve_mask_and_close_update():
    result = endpoint_terms(
        np.array([10.0, 20.0]),
        np.array([0.1, 0.2]),
        np.array([1.0, 2.0]),
        np.array([True, False]),
    )

    np.testing.assert_allclose(result["outgoing_amount"], [1.0, 0.0])
    np.testing.assert_allclose(result["incoming_amount"], [1.0, 0.0])
    np.testing.assert_allclose(result["endpoint"], [10.0, 20.0])
    np.testing.assert_array_equal(result["defined"], [True, False])


def test_compact_carbon_sum_uses_owner_area_basis_and_masks():
    litter = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    defined = np.ones_like(litter, dtype=bool)
    fraction = np.array([[0.25, 0.75]])
    active = np.array([[True, True]])

    litter_total = compact_carbon_sum(
        litter,
        defined,
        fraction=fraction,
        active=active,
        owner="litter_above",
    )
    assert litter_total == {
        "value_gC_m2": 5.5,
        "valid_coordinate_count": 4,
        "missing_active_coordinate_count": 0,
        "complete": True,
    }

    canopy = np.array([[[10.0], [20.0]]])
    canopy_total = compact_carbon_sum(
        canopy,
        np.array([[[True], [False]]]),
        fraction=fraction,
        active=active,
        owner="interception_storage",
    )
    assert canopy_total["value_gC_m2"] == 10.0
    assert canopy_total["valid_coordinate_count"] == 1
    assert canopy_total["missing_active_coordinate_count"] == 1
    assert canopy_total["complete"] is False
