from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.sechiba import io_contracts as io


NO_KEY = "NO_KEYWORD"


@pytest.mark.parametrize(
    ("function", "var", "expected", "put"),
    [
        (io.i0setvar, np.int32(-1), np.int32(-1), np.int32(3)),
        (io.i10setvar, np.full(2, -1, dtype=np.int32), np.int32(-1), np.int32(3)),
        (io.i11setvar, np.full(2, -1, dtype=np.int32), np.int32(-1), np.array([3, 4], dtype=np.int32)),
        (io.i20setvar, np.full((2, 3), -1, dtype=np.int32), np.int32(-1), np.int32(3)),
        (io.i21setvar, np.full((2, 3), -1, dtype=np.int32), np.int32(-1), np.array([3, 4, 5], dtype=np.int32)),
        (io.i22setvar, np.full((2, 3), -1, dtype=np.int32), np.int32(-1), np.arange(6, dtype=np.int32).reshape(2, 3)),
        (io.r0setvar, np.float64(-1), np.float64(-1), np.float64(3)),
        (io.r10setvar, np.full(2, -1.0), np.float64(-1), np.float64(3)),
        (io.r11setvar, np.full(2, -1.0), np.float64(-1), np.array([3.0, 4.0])),
        (io.r20setvar, np.full((2, 3), -1.0), np.float64(-1), np.float64(3)),
        (io.r21setvar, np.full((2, 3), -1.0), np.float64(-1), np.array([3.0, 4.0, 5.0])),
        (io.r22setvar, np.full((2, 3), -1.0), np.float64(-1), np.arange(6.0).reshape(2, 3)),
        (io.r30setvar, np.full((2, 2, 2), -1.0), np.float64(-1), np.float64(3)),
    ],
)
def test_all_serial_rank_type_variants_assign_only_wholly_exceptional(function, var, expected, put):
    result = function(var, expected, NO_KEY, put)
    if np.asarray(var).ndim == 2 and np.asarray(put).ndim == 1:
        np.testing.assert_array_equal(result, np.repeat(np.asarray(put)[None, :], 2, axis=0))
    elif np.asarray(put).ndim == 0 and np.asarray(var).ndim > 0:
        np.testing.assert_array_equal(result, np.full(np.asarray(var).shape, put))
    else:
        np.testing.assert_array_equal(result, put)

    retained = np.array(var, copy=True)
    if retained.ndim:
        retained.flat[0] = 99
    else:
        retained = np.asarray(99, dtype=retained.dtype)
    np.testing.assert_array_equal(function(retained, expected, NO_KEY, put), retained)


def test_generic_dispatch_and_key_lookup_are_source_routed():
    reader = io.MappingConfigurationReader({"COUNT": np.int32(7), "PROFILE": np.array([8.0, 9.0])})
    assert io.setvar(np.int32(-1), np.int32(-1), "COUNT", np.int32(3), config=reader) == 7
    np.testing.assert_array_equal(
        io.setvar(np.full(2, -1.0), -1.0, "PROFILE", np.array([3.0, 4.0]), config=reader),
        [8.0, 9.0],
    )
    with pytest.raises(io.ConfigurationBoundaryError, match="explicit configuration reader"):
        io.r0setvar(-1.0, -1.0, "MISSING_BOUNDARY", 3.0)


def test_no_keyword_detection_matches_fortran_case_sensitive_substring_index():
    assert io.i0setvar(np.int32(-1), np.int32(-1), "prefix_NO_KEYWORD_suffix", np.int32(4)) == 4
    with pytest.raises(io.ConfigurationBoundaryError):
        io.i0setvar(np.int32(-1), np.int32(-1), "no_keyword", np.int32(4))


def test_rank_type_and_shape_mismatches_are_rejected_without_broadcasting():
    with pytest.raises(io.SetvarContractError, match="rank 1"):
        io.i11setvar(np.full(2, -1, dtype=np.int32), np.int32(-1), NO_KEY, np.ones((1, 2), dtype=np.int32))
    with pytest.raises(io.SetvarContractError, match="Fortran integer"):
        io.i10setvar(np.full(2, -1.0), np.int32(-1), NO_KEY, np.int32(2))
    with pytest.raises(io.SetvarContractError, match="must equal var shape"):
        io.r22setvar(np.full((2, 3), -1.0), -1.0, NO_KEY, np.ones((3, 2)))
    with pytest.raises(io.SetvarContractError, match="matches neither"):
        io.r21setvar(np.full((2, 3), -1.0), -1.0, NO_KEY, np.ones(4))


def test_spread_preserves_fortran_axis_precedence_and_copy_count():
    square = io.i21setvar(
        np.full((2, 2), -1, dtype=np.int32), np.int32(-1), NO_KEY, np.array([4, 5], dtype=np.int32)
    )
    np.testing.assert_array_equal(square, [[4, 4], [5, 5]])
    with pytest.raises(io.SetvarContractError, match="SPREAD result"):
        io.i21setvar(
            np.full((2, 3), -1, dtype=np.int32), np.int32(-1), NO_KEY, np.array([4, 5], dtype=np.int32)
        )


def test_parallel_variants_match_serial_when_optional_grid_argument_is_absent():
    pairs = [(name, getattr(io, name), getattr(io, name + "_p")) for name in (
        "i0setvar", "i10setvar", "i11setvar", "i20setvar", "i21setvar", "i22setvar",
        "r0setvar", "r10setvar", "r11setvar", "r20setvar", "r21setvar", "r30setvar",
    )]
    arguments = {
        "i0setvar": (np.int32(-1), np.int32(-1), np.int32(2)),
        "i10setvar": (np.full(2, -1, dtype=np.int32), np.int32(-1), np.int32(2)),
        "i11setvar": (np.full(2, -1, dtype=np.int32), np.int32(-1), np.array([2, 3], dtype=np.int32)),
        "i20setvar": (np.full((2, 2), -1, dtype=np.int32), np.int32(-1), np.int32(2)),
        "i21setvar": (np.full((2, 2), -1, dtype=np.int32), np.int32(-1), np.array([2, 3], dtype=np.int32)),
        "i22setvar": (np.full((2, 2), -1, dtype=np.int32), np.int32(-1), np.array([[2, 3], [4, 5]], dtype=np.int32)),
        "r0setvar": (-1.0, -1.0, 2.0), "r10setvar": (np.full(2, -1.0), -1.0, 2.0),
        "r11setvar": (np.full(2, -1.0), -1.0, np.array([2.0, 3.0])),
        "r20setvar": (np.full((2, 2), -1.0), -1.0, 2.0),
        "r21setvar": (np.full((2, 2), -1.0), -1.0, np.array([2.0, 3.0])),
        "r30setvar": (np.full((2, 2, 2), -1.0), -1.0, 2.0),
    }
    for name, serial, parallel in pairs:
        var, expected, put = arguments[name]
        np.testing.assert_array_equal(parallel(var, expected, NO_KEY, put), serial(var, expected, NO_KEY, put))


def test_present_false_is_grid_still_routes_gather_root_read_scatter():
    reader = io.MappingConfigurationReader({"GRID_IDS": np.array([11, 12], dtype=np.int32)})
    backend = io.SerialGridParallelIO(reader, np.array([1, 2], dtype=np.int32))
    result = io.i11setvar_p(
        np.full(2, -1, dtype=np.int32), np.int32(-1), "GRID_IDS",
        np.array([3, 4], dtype=np.int32), is_grid=False, grid_io=backend,
    )
    np.testing.assert_array_equal(result, [11, 12])
    with pytest.raises(io.ParallelIOBoundaryError, match="PRESENT\(is_grid\)"):
        io.i11setvar_p(
            np.full(2, -1, dtype=np.int32), np.int32(-1), "GRID_IDS",
            np.array([3, 4], dtype=np.int32), is_grid=False,
        )


def test_parallel_r22_reads_only_scalar_and_keeps_default_when_key_missing():
    default = np.arange(6.0).reshape(2, 3)
    target = np.full((2, 3), -1.0)
    np.testing.assert_array_equal(
        io.r22setvar_p(target, -1.0, "FIELD", default, config=io.MappingConfigurationReader({"FIELD": 9.0})),
        np.full((2, 3), 9.0),
    )
    np.testing.assert_array_equal(
        io.r22setvar_p(target, -1.0, "ABSENT", default, config=io.MappingConfigurationReader({})), default
    )


def test_all_26_procedures_have_direct_fortran_provenance():
    names = [prefix + suffix for suffix in ("", "_p") for prefix in (
        "i0setvar", "i10setvar", "i11setvar", "i20setvar", "i21setvar", "i22setvar",
        "r0setvar", "r10setvar", "r11setvar", "r20setvar", "r21setvar", "r22setvar", "r30setvar",
    )]
    assert len(names) == 26
    assert all("sechiba_io" in (getattr(io, name).__doc__ or "") and "lines" in getattr(io, name).__doc__ for name in names)
