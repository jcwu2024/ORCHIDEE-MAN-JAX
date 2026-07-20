from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from jax_orchidee.driver.geometry_polygons import (
    FortranUndefinedOutputError,
    polygones_area,
    polygones_cleanup,
    polygones_convexhull,
    polygones_crossing,
    polygones_extend,
    polygones_intersection,
    polygones_lineintersect,
    polygones_pointinside,
)


# Source-extracted oracle: executable statements are copied from
# polygones.f90 lines 20-59, 63-100 and 343-377. Only kinds and IO are adapted.
POLYGON_ORACLE = r"""
module oracle_polygones
  implicit none
contains
  subroutine pointinside(nvert_in, poly, point_x, point_y, inside)
    integer, intent(in) :: nvert_in
    real(8), dimension(:,:), intent(in) :: poly
    real(8), intent(in) :: point_x, point_y
    logical, intent(out) :: inside
    integer :: i, j, nvert
    real(8) :: xonline
    inside = .false.
    nvert=size(poly,dim=1)
    if (nvert < nvert_in) stop 11
    if (size(poly,dim=2) .ne. 2) stop 12
    j = nvert_in
    do i=1,nvert_in
      if ((poly(i,2) > point_y) .neqv. (poly(j,2) > point_y)) then
        xonline=(poly(j,1)-poly(i,1))*(point_y-poly(i,2))/(poly(j,2)-poly(i,2))+poly(i,1)
        if (point_x < xonline) inside=(.not. inside)
      endif
      j=i
    enddo
  end subroutine
  subroutine lineintersect(la, lb, intersection, point_x, point_y)
    real(8), dimension(2,2), intent(in) :: la, lb
    logical, intent(out) :: intersection
    real(8), intent(out) :: point_x, point_y
    real(8) :: den, ua, ub, x1,x2,x3,x4,y1,y2,y3,y4
    intersection=.false.
    x1=la(1,1); y1=la(1,2); x2=la(2,1); y2=la(2,2)
    x3=lb(1,1); y3=lb(1,2); x4=lb(2,1); y4=lb(2,2)
    den=(y4-y3)*(x2-x1)-(x4-x3)*(y2-y1)
    if (abs(den) > epsilon(den)) then
      ua=((x4-x3)*(y1-y3)-(y4-y3)*(x1-x3))/den
      ub=((x2-x1)*(y1-y3)-(y2-y1)*(x1-x3))/den
      if (ua >= 0 .and. ua <= 1 .and. ub >= 0 .and. ub <= 1) then
        intersection=.true.
        point_x=x1+ua*(x2-x1); point_y=y1+ua*(y2-y1)
      endif
    endif
  end subroutine
  subroutine polygon_area(nvert_in, poly_in, dx, dy, area)
    integer, intent(in) :: nvert_in
    real(8), dimension(:,:), intent(in) :: poly_in
    real(8), intent(in) :: dx,dy
    real(8), intent(out) :: area
    integer :: nvert,i,j
    nvert=size(poly_in,dim=1)
    if (nvert < nvert_in) stop 13
    if (size(poly_in,dim=2) .ne. 2) stop 14
    area=0.0_8; j=nvert_in
    do i=1,nvert_in
      area=area+dy*dx/2.0_8*(poly_in(j,2)+poly_in(i,2))*(poly_in(j,1)-poly_in(i,1))
      j=i
    enddo
    area=abs(area)
  end subroutine
end module
program oracle
  use oracle_polygones
  implicit none
  integer :: mode,n,i
  real(8), allocatable :: p(:,:)
  real(8) :: px,py,dx,dy,a
  real(8) :: la(2,2),lb(2,2)
  logical :: flag
  read(*,*) mode
  if (mode == 1) then
    read(*,*) n,px,py; allocate(p(n,2))
    do i=1,n; read(*,*) p(i,1),p(i,2); enddo
    call pointinside(n,p,px,py,flag); write(*,*) flag
  else if (mode == 2) then
    do i=1,2; read(*,*) la(i,1),la(i,2); enddo
    do i=1,2; read(*,*) lb(i,1),lb(i,2); enddo
    call lineintersect(la,lb,flag,px,py); write(*,*) flag,px,py
  else
    read(*,*) n,dx,dy; allocate(p(n,2))
    do i=1,n; read(*,*) p(i,1),p(i,2); enddo
    call polygon_area(n,p,dx,dy,a); write(*,'(ES25.16)') a
  endif
end program
"""


@pytest.fixture(scope="module")
def polygon_oracle(tmp_path_factory):
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran is required for the source-extracted polygon oracle")
    directory = tmp_path_factory.mktemp("polygon_oracle")
    source = directory / "polygon_oracle.f90"
    executable = directory / "polygon_oracle.exe"
    source.write_text(POLYGON_ORACLE, encoding="ascii")
    oracle_env = os.environ.copy()
    oracle_env["PATH"] = f"{Path(compiler).parent}{os.pathsep}{oracle_env['PATH']}"
    compile_result = None
    for _ in range(3):
        compile_result = subprocess.run(
            [compiler, source.name, "-o", executable.name],
            cwd=directory,
            capture_output=True,
            env=oracle_env,
            text=True,
        )
        if compile_result.returncode == 0:
            break
        time.sleep(0.1)
    assert compile_result is not None
    compile_result.check_returncode()
    return executable, oracle_env


def _oracle(oracle, lines):
    executable, oracle_env = oracle
    completed = subprocess.run(
        [str(executable)],
        input="\n".join(lines) + "\n",
        check=True,
        capture_output=True,
        env=oracle_env,
        text=True,
    )
    return completed.stdout.split()


def test_pointinside_matches_gfortran_and_preserves_asymmetric_boundary(polygon_oracle):
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    rows = ["0 0", "2 0", "2 2", "0 2"]
    for point in ((1.0, 1.0), (3.0, 1.0), (0.0, 1.0), (2.0, 1.0)):
        oracle = _oracle(polygon_oracle, ["1", f"4 {point[0]} {point[1]}", *rows])[0].upper().startswith("T")
        assert polygones_pointinside(4, square, *point) is oracle
    assert polygones_pointinside(4, square, 0.0, 1.0)
    assert not polygones_pointinside(4, square, 2.0, 1.0)


def test_line_intersection_and_area_match_gfortran(polygon_oracle):
    la = np.array([[0.0, 0.0], [2.0, 2.0]])
    lb = np.array([[0.0, 2.0], [2.0, 0.0]])
    expected = _oracle(polygon_oracle, ["2", "0 0", "2 2", "0 2", "2 0"])
    actual = polygones_lineintersect(la, lb)
    assert actual.intersection is expected[0].upper().startswith("T")
    np.testing.assert_allclose([actual.point_x, actual.point_y], np.asarray(expected[1:], dtype=float))
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    expected_area = float(_oracle(polygon_oracle, ["3", "4 3 4", "0 0", "2 0", "2 2", "0 2"])[0])
    assert polygones_area(4, square, 3.0, 4.0) == pytest.approx(expected_area)
    parallel = polygones_lineintersect([[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]])
    assert not parallel.intersection and np.isnan(parallel.point_x) and np.isnan(parallel.point_y)


def test_extend_keeps_last_to_first_order_and_closed_polygon_duplicate():
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    result = polygones_extend(4, square, 2)
    np.testing.assert_array_equal(
        result.vertices,
        [[0, 2], [0, 1], [0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2]],
    )
    closed = np.vstack((square, square[0]))
    closed_result = polygones_extend(5, closed, 1)
    np.testing.assert_array_equal(closed_result.vertices, [[0, 0], [0, 0], [2, 0], [2, 2], [0, 2]])


def test_intersection_cleanup_and_crossing_preserve_source_order_and_duplicates():
    outer = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    inner = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
    intersection = polygones_intersection(4, outer, 4, inner)
    np.testing.assert_array_equal(intersection.vertices, inner)
    dirty = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    cleaned = polygones_cleanup(5, dirty)
    np.testing.assert_array_equal(cleaned.vertices, [[0, 0], [1, 0], [2, 0], [1, 1]])
    a = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    b = np.array([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]])
    crossings = polygones_crossing(4, a, 4, b)
    # Two perpendicular edge pairs report the shared vertex. The other two
    # pairs are collinear and lineintersect deliberately rejects den == 0.
    assert crossings.nvert == 2
    np.testing.assert_array_equal(crossings.vertices, np.repeat([[2.0, 2.0]], 2, axis=0))


def test_convexhull_keeps_fortran_order_and_exposes_undefined_vertical_return():
    points = np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 0.5]])
    hull = polygones_convexhull(6, points)
    np.testing.assert_array_equal(hull.vertices, [[0, 0], [0, 2], [2, 2], [2, 0]])
    with pytest.raises(FortranUndefinedOutputError, match="before assigning nvert_out"):
        polygones_convexhull(4, [[1.0, 0.0], [1.0, 2.0], [1.0, 1.0], [1.0, 1.0]])


def test_shape_and_capacity_fatal_arms_are_explicit():
    with pytest.raises(ValueError, match="shape"):
        polygones_pointinside(2, np.ones((2, 3)), 0.0, 0.0)
    with pytest.raises(ValueError, match="too small"):
        polygones_extend(3, np.ones((3, 2)), 2, output_capacity=5)
    with pytest.raises(ValueError, match="poly_b is smaller than nvert_a"):
        polygones_intersection(6, np.ones((6, 2)), 2, np.ones((2, 2)))
