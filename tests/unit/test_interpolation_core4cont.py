from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from jax_orchidee.driver.interpolation import interpweight_calc_resolution_in
from jax_orchidee.driver.interpolation_core4cont import (
    UNDEFINED_MASK,
    FortranUndefinedInputError,
    interpweight_2dcont,
    interpweight_2dcont_routed,
    interpweight_4d,
    interpweight_4d_routed,
    select_interpweight_input,
)
from jax_orchidee.driver.interpolation_core12 import (
    AggregatePacket,
    InterpolationSource,
    InterpolationTarget,
)


def _geometry(areas=((6.0, 4.0, 0.0),)):
    area = np.asarray(areas, dtype=np.float64)
    index = np.zeros(area.shape + (2,), dtype=np.int32)
    index[:, 0, :] = (1, 1)
    if area.shape[1] > 1:
        index[:, 1, :] = (2, 1)
    return area, index


def _target(resolution: float) -> InterpolationTarget:
    return InterpolationTarget(
        lalo=np.array([[45.0, 2.0]]),
        resolution=np.full((1, 2), resolution, dtype=np.float64),
        neighbours=np.arange(8, dtype=np.int32).reshape(1, 8) + 1,
        contfrac=np.array([0.5]),
    )


def _aggregate_packet(request, *, ok: bool) -> AggregatePacket:
    area = np.zeros((1, request.nbvmax), dtype=np.float64)
    index = np.zeros((1, request.nbvmax, 2), dtype=np.int32)
    if ok:
        area[0, 0] = 2.0
        index[0, 0, :] = (1, 1)
    return AggregatePacket(index, area, ok)


def test_time_selection_is_positional_contiguous_and_calendar_neutral() -> None:
    values = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
    dates = np.array([10.0, 20.0, 40.0, 80.0, 160.0])

    first = select_interpweight_input(
        values, initime=0, time_values=dates, calendar="360_day"
    )
    middle = select_interpweight_input(
        values, initime=3, time_values=dates, calendar="gregorian"
    )
    last = select_interpweight_input(values, initime=99, time_values=dates)
    full = select_interpweight_input(values, initime=-1, time_values=dates)

    np.testing.assert_array_equal(first.values, values[:, :, :, 0])
    np.testing.assert_array_equal(middle.values, values[:, :, :, 2])
    np.testing.assert_array_equal(last.values, values[:, :, :, -1])
    np.testing.assert_array_equal(full.values, values)
    assert first.selected_time_index == 1
    assert middle.selected_time_index == 3
    assert last.selected_time_index == 5
    assert first.time_values.tolist() == [10.0]
    assert first.calendar == "360_day"
    assert full.selected_time_index is None
    assert full.effective_rank == 4


def test_rank_specific_unallocated_downstream_arguments_are_explicit() -> None:
    area, index = _geometry()
    common = dict(
        lon=np.array([0.0, 1.0]),
        lat=np.array([0.0]),
        sub_area=area,
        sub_index=index,
        resolution=np.array([[10.0, 10.0]]),
        contfrac=np.array([1.0]),
    )
    with pytest.raises(FortranUndefinedInputError, match="unallocated invar4D"):
        interpweight_4d(
            np.ones((2, 1, 2, 2)),
            variabletypes=np.array([1.0, 2.0]),
            varmin=np.ones(2),
            varmax=np.ones(2) * 2.0,
            initime=1,
            **common,
        )
    with pytest.raises(FortranUndefinedInputError, match="unallocated invar2D"):
        interpweight_2dcont(np.ones((2, 1, 2)), **common)


def test_4d_shared_mask_noneg_fraction_fallback_and_dynamic_shapes() -> None:
    values = np.array(
        [
            [[[0.2, -0.4, 0.6], [0.8, 1.4, 0.4]]],
            [[[0.5, 0.2, 0.1], [0.5, 0.8, 0.9]]],
            [[[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]]],
        ],
        dtype=np.float64,
    )
    area = np.array([[3.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    index = np.array(
        [
            [[1, 1], [2, 1], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
        ],
        dtype=np.int32,
    )
    result = interpweight_4d(
        values,
        variabletypes=np.array([-1.0, 99.0]),
        varmin=np.array([-5.0, -5.0]),
        varmax=np.array([5.0, 5.0]),
        lon=np.array([0.0, 1.0, 2.0]),
        lat=np.array([45.0]),
        sub_area=area,
        sub_index=index,
        resolution=np.array([[10.0, 5.0], [2.0, 4.0]]),
        contfrac=np.array([0.5, 1.0]),
        noneg=True,
        masktype="mbelow",
        maskvalues=np.array([0.1, 0.0, 2.0]),
    )

    # One value below 0.1 marks source cell 1 for every type and time.  Cell 2
    # is unwritten by the 4-D shared mask and is therefore not treated as land.
    np.testing.assert_array_equal(result.mask[:, 0], [1, UNDEFINED_MASK, UNDEFINED_MASK])
    np.testing.assert_array_equal(result.overlap_area[0], [3.0, 0.0, 0.0])
    assert result.source_values[0, 0, 0, 1] == 0.0
    np.testing.assert_allclose(result.output[0], result.source_values[0, 0])
    np.testing.assert_array_equal(result.variable_types, [1.0, 2.0])
    np.testing.assert_array_equal(result.varmin, [1.0, 1.0])
    np.testing.assert_array_equal(result.varmax, [2.0, 2.0])
    assert result.availability[0] == pytest.approx(3.0 / 50.0 / 0.5)
    assert result.availability[1] == -1.0
    np.testing.assert_array_equal(result.output[1, 0, :], np.ones(3))


def test_4d_msumrange_uses_one_mask_for_all_times_and_normalizes_in_place() -> None:
    values = np.array([[[[0.4, 1.2], [0.4, 0.8]]]], dtype=np.float64)
    area = np.array([[2.0]])
    index = np.array([[[1, 1]]], dtype=np.int32)
    result = interpweight_4d(
        values,
        variabletypes=np.array([1.0, 2.0]),
        varmin=np.ones(2),
        varmax=np.ones(2) * 2.0,
        lon=np.array([0.0]),
        lat=np.array([0.0]),
        sub_area=area,
        sub_index=index,
        resolution=np.array([[2.0, 2.0]]),
        contfrac=np.array([1.0]),
        masktype="msumrange",
        maskvalues=np.array([1.0, 0.5, 3.0]),
    )
    assert result.mask[0, 0] == 1
    np.testing.assert_allclose(result.source_values[0, 0, :, 0], [0.4, 0.4])
    np.testing.assert_allclose(result.source_values[0, 0, :, 1], [0.6, 0.4])


def test_2dcont_mask_compaction_interpolation_slopecalc_and_undef_writeback() -> None:
    values = np.array([[2.0], [6.0], [10.0]])
    area = np.array([[3.0, 2.0, 1.0], [0.0, 0.0, 0.0]])
    index = np.array(
        [
            [[1, 1], [2, 1], [3, 1]],
            [[0, 0], [0, 0], [0, 0]],
        ],
        dtype=np.int32,
    )
    result = interpweight_2dcont(
        values,
        lon=np.array([0.0, 1.0, 2.0]),
        lat=np.array([0.0]),
        sub_area=area,
        sub_index=index,
        resolution=np.array([[2.0, 2.0], [3.0, 4.0]]),
        contfrac=np.array([0.5, 1.0]),
        masktype="var",
        maskvar=np.array([[1.0], [0.0], [1.0]]),
        typefrac="slopecalc",
        defaultvalue=-7.0,
        default_no_value=8.0,
    )
    np.testing.assert_array_equal(result.mask[:, 0], [1, 0, 1])
    np.testing.assert_array_equal(result.overlap_area[0], [3.0, 1.0, 0.0])
    assert result.output[0] == pytest.approx(1.0 - ((2.0 / 8.0) * 3.0 + 1.0) / 4.0)
    assert result.availability[0] == pytest.approx(4.0 / 4.0 / 0.5)
    assert result.output[1] == -7.0
    assert result.availability[1] == -1.0


def test_routed_4d_explicit_maxres_has_200_floor_and_doubles_retry() -> None:
    seen = []

    def aggregate(request):
        seen.append(request)
        return _aggregate_packet(request, ok=len(seen) == 2)

    result = interpweight_4d_routed(
        InterpolationSource(
            values=np.full((2, 2, 2, 2), 0.5),
            longitude=np.array([0.0, 0.001]),
            latitude=np.array([45.0, 45.001]),
            variable_name="lai",
        ),
        _target(10.0),
        np.array([1.0, 2.0]),
        aggregate,
        varmin=np.ones(2),
        varmax=np.ones(2) * 2.0,
        masktype="nomask",
        max_resolution_lon=5.0,
        max_resolution_lat=5.0,
    )

    assert result.nbvmax_attempts == (200, 400)
    assert [request.nbvmax for request in seen] == [200, 400]
    assert seen[0].source_rank == 4
    assert seen[0].callsign == "lai map"
    np.testing.assert_array_equal(seen[0].lalo, _target(10.0).lalo)
    np.testing.assert_array_equal(seen[0].neighbours, _target(10.0).neighbours)
    np.testing.assert_array_equal(seen[0].mask, np.ones((2, 2), dtype=np.int32))
    np.testing.assert_allclose(result.output[0], 0.5)
    assert result.availability[0] == pytest.approx(2.0 / 100.0 / 0.5)


def test_routed_4d_calculates_source_resolution_when_either_maxres_is_missing() -> None:
    seen = []

    def aggregate(request):
        seen.append(request)
        return _aggregate_packet(request, ok=True)

    result = interpweight_4d_routed(
        InterpolationSource(
            np.full((2, 2, 2, 1), 0.5),
            np.array([0.0, 0.001]),
            np.array([45.0, 45.001]),
            variable_name="lai",
        ),
        _target(1_500.0),
        np.array([1.0, 2.0]),
        aggregate,
        varmin=np.ones(2),
        varmax=np.ones(2) * 2.0,
        masktype="nomask",
        max_resolution_lon=5.0,
        max_resolution_lat=-1.0,
    )

    lon = np.broadcast_to(np.array([0.0, 0.001])[:, None], (2, 2))
    lat = np.broadcast_to(np.array([45.0, 45.001])[None, :], (2, 2))
    source_resolution = interpweight_calc_resolution_in(lon, lat)
    expected_nix = int(1_500.0 / np.max(source_resolution[:, :, 0])) + 2
    expected_njx = int(1_500.0 / np.max(source_resolution[:, :, 1])) + 2
    assert seen[0].nbvmax == max(expected_nix * expected_njx, 200)
    assert result.nbvmax_attempts == (seen[0].nbvmax,)
    assert seen[0].max_resolution_lon == 5.0
    assert seen[0].max_resolution_lat == -1.0


def test_routed_2dcont_source_resolution_has_no_200_floor_and_retries() -> None:
    seen = []

    def aggregate(request):
        seen.append(request)
        return _aggregate_packet(request, ok=len(seen) == 2)

    result = interpweight_2dcont_routed(
        InterpolationSource(
            np.array([[2.0, 3.0], [4.0, 5.0]]),
            np.array([0.0, 0.001]),
            np.array([45.0, 45.001]),
            variable_name="slope",
        ),
        _target(100.0),
        aggregate,
        masktype="nomask",
    )

    lon = np.broadcast_to(np.array([0.0, 0.001])[:, None], (2, 2))
    lat = np.broadcast_to(np.array([45.0, 45.001])[None, :], (2, 2))
    source_resolution = interpweight_calc_resolution_in(lon, lat)
    expected_nix = int(100.0 / np.max(source_resolution[:, :, 0])) + 2
    expected_njx = int(100.0 / np.max(source_resolution[:, :, 1])) + 2
    assert seen[0].nbvmax == expected_nix * expected_njx
    assert seen[0].nbvmax < 200
    assert seen[1].nbvmax == seen[0].nbvmax * 2
    assert result.nbvmax_attempts == (seen[0].nbvmax, seen[1].nbvmax)
    assert seen[0].source_rank == 2
    assert seen[0].max_resolution_lon == -1.0
    assert seen[0].max_resolution_lat == -1.0
    assert result.output[0] == pytest.approx(2.0)


GFORTRAN = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")
if not GFORTRAN.is_file():
    discovered = shutil.which("gfortran")
    GFORTRAN = Path(discovered) if discovered else GFORTRAN


@pytest.fixture(scope="module")
def interpolation_oracle(tmp_path_factory):
    if not GFORTRAN.is_file():
        pytest.skip("gfortran is unavailable")
    directory = tmp_path_factory.mktemp("interpweight_core_oracle")
    source = directory / "oracle.f90"
    executable = directory / "oracle.exe"
    # Computational statements are extracted from lines 1554-1575,
    # 3722-3755, and 3856-3888.  IO/MPI/geometry calls stay outside the oracle.
    source.write_text(
        textwrap.dedent(
            """
            program oracle
              implicit none
              real(8) :: v4(2,1,2,2), area(3), out4(2,2), avail, v2(2,1), out2
              integer :: idx(3,2), ib,it,itt,jj,ip,jp,idi,idi_last,initime,tml,selected
              v4=0d0
              v4(1,1,1,:)=(/0.2d0,0.4d0/); v4(1,1,2,:)=(/0.8d0,0.6d0/)
              v4(2,1,1,:)=(/0.5d0,0.25d0/); v4(2,1,2,:)=(/0.5d0,0.75d0/)
              area=(/6d0,4d0,0d0/); idx=0; idx(1,:)=(/1,1/); idx(2,:)=(/2,1/)
              out4=0d0; avail=0d0; idi=count(area > 0d0)
              do jj=1,idi
                ip=idx(jj,1); jp=idx(jj,2)
                do it=1,2; do itt=1,2
                  if (v4(ip,jp,it,itt)>0d0 .and. v4(ip,jp,it,itt)<20d0) &
                    out4(it,itt)=out4(it,itt)+v4(ip,jp,it,itt)*area(jj)
                end do; end do
                avail=avail+area(jj)
              end do
              out4=out4/avail
              write(*,'(5(ES25.16,1X))') out4(1,1),out4(2,1),out4(1,2),out4(2,2),avail

              v2(:,1)=(/2d0,6d0/); area=(/3d0,1d0,0d0/); out2=0d0; avail=0d0
              idi_last=3
              do idi=1,3
                if (area(idi)<=0d0) then; idi_last=idi-1; exit; end if
                out2=out2+v2(idx(idi,1),idx(idi,2))*area(idi); avail=avail+area(idi)
              end do
              if (idi_last>=1) out2=out2/avail
              write(*,'(2(ES25.16,1X))') out2,avail

              tml=5
              do initime=0,8,4
                if (initime>0) then; selected=min(initime,tml); else; selected=1; end if
                write(*,'(I0,1X)',advance='no') selected
              end do
              write(*,*)
            end program oracle
            """
        ),
        encoding="ascii",
    )
    env = os.environ.copy()
    env["PATH"] = f"{GFORTRAN.parent}{os.pathsep}{env.get('PATH', '')}"
    compiled = None
    for _ in range(3):
        compiled = subprocess.run(
            [str(GFORTRAN), str(source), "-o", str(executable)],
            capture_output=True,
            text=True,
            env=env,
        )
        if compiled.returncode == 0:
            break
        time.sleep(0.1)
    assert compiled is not None
    compiled.check_returncode()
    run = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, env=env
    )
    return run.stdout.splitlines()


def test_numerics_and_time_selection_match_source_extracted_gfortran(
    interpolation_oracle,
) -> None:
    four_d_oracle = np.fromstring(interpolation_oracle[0], sep=" ")
    two_d_oracle = np.fromstring(interpolation_oracle[1], sep=" ")
    time_oracle = np.fromstring(interpolation_oracle[2], sep=" ", dtype=np.int64)

    area, index = _geometry()
    values4 = np.array(
        [
            [[[0.2, 0.4], [0.8, 0.6]]],
            [[[0.5, 0.25], [0.5, 0.75]]],
        ]
    )
    result4 = interpweight_4d(
        values4,
        variabletypes=np.array([1.0, 2.0]),
        varmin=np.ones(2),
        varmax=np.ones(2) * 2.0,
        lon=np.array([0.0, 1.0]),
        lat=np.array([0.0]),
        sub_area=area,
        sub_index=index,
        resolution=np.array([[1.0, 1.0]]),
        contfrac=np.array([1.0]),
    )
    np.testing.assert_allclose(
        [
            result4.output[0, 0, 0],
            result4.output[0, 1, 0],
            result4.output[0, 0, 1],
            result4.output[0, 1, 1],
            result4.availability[0],
        ],
        four_d_oracle,
        rtol=2e-15,
        atol=2e-15,
    )

    result2 = interpweight_2dcont(
        np.array([[2.0], [6.0]]),
        lon=np.array([0.0, 1.0]),
        lat=np.array([0.0]),
        sub_area=np.array([[3.0, 1.0, 0.0]]),
        sub_index=index,
        resolution=np.array([[1.0, 1.0]]),
        contfrac=np.array([1.0]),
    )
    np.testing.assert_allclose(
        [result2.output[0], result2.availability[0]], two_d_oracle, rtol=0.0, atol=0.0
    )
    time_values = np.zeros((1, 1, 1, 5))
    selected = [
        select_interpweight_input(time_values, initime=value).selected_time_index
        for value in (0, 4, 8)
    ]
    np.testing.assert_array_equal(selected, time_oracle)
