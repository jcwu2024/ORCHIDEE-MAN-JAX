import numpy as np
import pytest

from jax_orchidee.driver.static import (
    SOILCLASS_DEFAULT_FAO,
    SOILCLASS_DEFAULT_USDA,
    ZOBLER_TEXTFRAC_TABLE,
    slowproc_soilt_from_explicit_overlap,
)


def _packet(classes, *, bulk=None, ph=None, poor=None, areas=None):
    classes = np.asarray(classes, dtype=np.float64).reshape(-1, 1)
    size = classes.shape[0]
    indices = np.asarray([[[index + 1, 1] for index in range(size)]], dtype=np.int32)
    return (
        classes,
        np.asarray(bulk if bulk is not None else np.ones(size), dtype=np.float64).reshape(-1, 1),
        np.asarray(ph if ph is not None else np.full(size, 6.0), dtype=np.float64).reshape(-1, 1),
        np.asarray(poor if poor is not None else np.zeros(size), dtype=np.float64).reshape(-1, 1),
        indices,
        np.asarray([areas if areas is not None else np.ones(size)], dtype=np.float64),
    )


def test_zobler_maps_classes_and_excludes_glacier_before_normalizing():
    """Fortran slowproc_soilt lines 4627-4692: Zobler mask/map/normalize."""

    args = _packet([1, 2, 5, 6, 7], bulk=[1, 2, 3, 99, 4], ph=[5, 6, 7, 99, 8], poor=[0.1, 0.2, 0.3, 9, 0.4])
    result = slowproc_soilt_from_explicit_overlap(*args, soil_classif="zobler")

    np.testing.assert_allclose(result["soilclass"], [[0.25, 0.5, 0.25]])
    expected_texture = np.mean(ZOBLER_TEXTFRAC_TABLE[[0, 1, 4, 6]], axis=0)
    np.testing.assert_allclose(result["silt_frac"], [expected_texture[0]])
    np.testing.assert_allclose(result["sand_frac"], [expected_texture[1]])
    np.testing.assert_allclose(result["clay_frac"], [expected_texture[2]])
    np.testing.assert_allclose(result["bulk_dens"], [2.5])
    np.testing.assert_allclose(result["soil_ph"], [6.5])
    np.testing.assert_allclose(result["poor_soils"], [0.25])
    np.testing.assert_array_equal(result["njsc"], [2])
    np.testing.assert_array_equal(result["used_default"], [False])


def test_zobler_repairs_negative_density_and_ph_but_usda_does_not():
    """Fortran slowproc_soilt lines 4618-4621 versus USDA lines 4837-4889."""

    args = _packet([1], bulk=[-2.0], ph=[-3.0])
    zobler = slowproc_soilt_from_explicit_overlap(*args, soil_classif="zobler")
    usda = slowproc_soilt_from_explicit_overlap(*args, soil_classif="usda")

    np.testing.assert_allclose(zobler["bulk_dens"], [1.65])
    np.testing.assert_allclose(zobler["soil_ph"], [7.0])
    np.testing.assert_allclose(usda["bulk_dens"], [-2.0])
    np.testing.assert_allclose(usda["soil_ph"], [-3.0])


@pytest.mark.parametrize("classification,expected", [("zobler", SOILCLASS_DEFAULT_FAO), ("usda", SOILCLASS_DEFAULT_USDA)])
def test_no_valid_texture_uses_source_defaults(classification, expected):
    """Fortran slowproc_soilt lines 4600-4608, 4676-4684, and 4825-4834."""

    args = _packet([0, 6] if classification == "zobler" else [0], areas=[2.0, 3.0] if classification == "zobler" else [2.0])
    result = slowproc_soilt_from_explicit_overlap(*args, soil_classif=classification)

    np.testing.assert_allclose(result["soilclass"], expected[None, :])
    np.testing.assert_allclose(result["clay_frac"], [0.2])
    np.testing.assert_allclose(result["sand_frac"], [0.4])
    np.testing.assert_allclose(result["silt_frac"], [0.4])
    np.testing.assert_allclose(result["bulk_dens"], [1.65])
    np.testing.assert_allclose(result["soil_ph"], [7.0])
    np.testing.assert_allclose(result["poor_soils"], [0.0])
    np.testing.assert_array_equal(result["used_default"], [True])


def test_no_overlap_uses_defaults_before_reading_source_indices():
    """Fortran slowproc_soilt lines 4588-4608 and 4813-4834: fopt == 0."""

    args = list(_packet([12], areas=[0.0]))
    args[4][0, 0] = [999, 999]
    result = slowproc_soilt_from_explicit_overlap(*args, soil_classif="usda")

    np.testing.assert_allclose(result["soilclass"], SOILCLASS_DEFAULT_USDA[None, :])
    np.testing.assert_array_equal(result["used_default"], [True])


def test_usda_accumulates_all_outputs_and_poor_soil_switch():
    """Fortran slowproc_soilt lines 4438-4443 and 4849-4889."""

    args = _packet([1, 12], bulk=[1.2, 1.8], ph=[5.0, 7.0], poor=[0.25, 0.75], areas=[1.0, 3.0])
    with_poor = slowproc_soilt_from_explicit_overlap(*args, soil_classif="usda")
    no_poor_args = list(args)
    no_poor_args[3] = None
    without_poor = slowproc_soilt_from_explicit_overlap(*no_poor_args, soil_classif="usda", do_poor_soils=False)

    expected_class = np.zeros(12)
    expected_class[[0, 11]] = [0.25, 0.75]
    np.testing.assert_allclose(with_poor["soilclass"], expected_class[None, :])
    np.testing.assert_allclose(with_poor["bulk_dens"], [1.65])
    np.testing.assert_allclose(with_poor["soil_ph"], [6.5])
    np.testing.assert_allclose(with_poor["poor_soils"], [0.625])
    np.testing.assert_allclose(without_poor["poor_soils"], [0.0])
    np.testing.assert_array_equal(with_poor["njsc"], [12])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"soil_classif": "none"}, "poor_soils undefined"),
        ({"soil_classif": "fao"}, "fatal consistency check"),
        ({"soil_classif": "usda", "impsoilt": True}, "not entered"),
        ({"soil_classif": "unknown"}, "unsupported soil classification"),
    ],
)
def test_source_fatal_or_external_paths_are_explicit(kwargs, message):
    """Fortran slowproc_soilt lines 4560-4570, 4698-4706, and 4896-4900."""

    with pytest.raises(ValueError, match=message):
        slowproc_soilt_from_explicit_overlap(*_packet([1]), **kwargs)


def test_invalid_class_and_malformed_overlap_fail_instead_of_falling_back():
    with pytest.raises(ValueError, match="bad usda soiltext class"):
        slowproc_soilt_from_explicit_overlap(*_packet([13]), soil_classif="usda")

    args = list(_packet([1, 2], areas=[0.0, 1.0]))
    with pytest.raises(ValueError, match="packed first"):
        slowproc_soilt_from_explicit_overlap(*args, soil_classif="usda")

    args = list(_packet([1]))
    args[4][0, 0] = [2, 1]
    with pytest.raises(ValueError, match="out-of-range"):
        slowproc_soilt_from_explicit_overlap(*args, soil_classif="usda")
