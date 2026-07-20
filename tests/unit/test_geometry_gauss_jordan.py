from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from jax_orchidee.driver.geometry_gauss_jordan import (
    GaussJordanError,
    error_l1_passive,
    gauss_jordan_method,
)


GFORTRAN = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")

# Source-extracted lines 61-157 and 177-241. The fixture constants replace
# constantes_var dependencies; loop and arithmetic order are unchanged.
GAUSS_ORACLE = r"""
program gauss_oracle
  implicit none
  real(8) :: a(3,3),b(3),cur(3,2,7),prev(3,2,7),veg(3,2)
  logical :: flag(3)
  a(1,:)=(/0d0,2d0,1d0/); a(2,:)=(/1d0,1d0,0d0/); a(3,:)=(/2d0,0d0,1d0/)
  b=(/7d0,3d0,5d0/)
  call solve(3,a,b)
  write(*,'(3(ES25.17E3,1X))') b
  write(*,'(9(ES25.17E3,1X))') transpose(a)
  cur=0d0; prev=0d0; veg=reshape((/0.25d0,1d0,0.5d0,0.75d0,0d0,0.5d0/),(/3,2/))
  prev(:,1,7)=(/100d0,1d-10,4d0/); prev(:,2,7)=(/20d0,1d-10,6d0/)
  cur(:,1,7)=(/101d0,2d-10,4.01d0/); cur(:,2,7)=(/19d0,3d-10,6.01d0/)
  call err(cur,prev,veg,0.2d0,flag)
  write(*,'(3(I1,1X))') merge(1,0,flag)
contains
  subroutine solve(n,matrix_a,vector_b)
    integer,intent(in)::n
    real(8),intent(inout)::matrix_a(n,n),vector_b(n)
    integer::i,col,row,j,k,ii,jj,index_pivot(n),index_col(n),index_row(n)
    real(8)::pivot_max,inv_pivot,temp
    index_pivot=0; col=0; row=0
    do i=1,n
      pivot_max=0d0
      do j=1,n
        if(index_pivot(j)/=1)then
          do k=1,n
            if(index_pivot(k)==0)then
              if(abs(matrix_a(j,k))>=pivot_max)then
                pivot_max=abs(matrix_a(j,k)); row=j; col=k
              endif
            endif
          enddo
        endif
      enddo
      if(col==0) error stop 'Method failed'
      index_pivot(col)=index_pivot(col)+1
      if(row/=col)then
        do j=1,n
          temp=matrix_a(row,j);matrix_a(row,j)=matrix_a(col,j);matrix_a(col,j)=temp
        enddo
        temp=vector_b(row);vector_b(row)=vector_b(col);vector_b(col)=temp
      endif
      index_row(i)=row;index_col(i)=col
      if(matrix_a(col,col)==0d0) error stop 'not inversible'
      inv_pivot=1d0/matrix_a(col,col)
      do j=1,n;matrix_a(col,j)=matrix_a(col,j)*inv_pivot;enddo
      vector_b(col)=vector_b(col)*inv_pivot
      do ii=1,n
        if(ii/=col)then
          temp=matrix_a(ii,col);matrix_a(ii,col)=0d0
          do jj=1,n;matrix_a(ii,jj)=matrix_a(ii,jj)-matrix_a(col,jj)*temp;enddo
          vector_b(ii)=vector_b(ii)-vector_b(col)*temp
        endif
      enddo
    enddo
    do j=n,1,-1
      if(index_row(j)/=index_col(j))then
        do i=1,n
          temp=matrix_a(i,index_row(j));matrix_a(i,index_row(j))=matrix_a(i,index_col(j));matrix_a(i,index_col(j))=temp
        enddo
      endif
    enddo
  end subroutine solve
  subroutine err(current,previous,veget,criterion,flag)
    real(8),intent(in)::current(3,2,7),previous(3,2,7),veget(3,2),criterion
    logical,intent(out)::flag(3)
    real(8)::ps(3),cs(3),eg(3),diff(3)
    integer::j
    flag=.false.;eg=0d0;ps=0d0
    do j=1,2;ps=ps+previous(:,j,7)*veget(:,j);enddo
    cs=0d0
    do j=1,2;cs=cs+current(:,j,7)*veget(:,j);enddo
    diff=cs-ps
    where(ps>1d-8);eg=100d0*abs(diff)/ps;elsewhere;eg=abs(diff);endwhere
    where(eg<=criterion);flag=.true.;endwhere
  end subroutine err
end program gauss_oracle
"""


@pytest.fixture(scope="module")
def gauss_oracle(tmp_path_factory):
    compiler = GFORTRAN if GFORTRAN.exists() else shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran is unavailable")
    work = tmp_path_factory.mktemp("gauss_jordan_micro_oracle")
    source = work / "gauss_oracle.f90"
    source.write_text(GAUSS_ORACLE, encoding="ascii")
    executable = work / "gauss_oracle.exe"
    env = os.environ.copy()
    if GFORTRAN.exists():
        env["PATH"] = rf"{GFORTRAN.parent};C:\msys64\usr\bin;{env['PATH']}"
    subprocess.run([str(compiler), "-O0", str(source), "-o", str(executable)], check=True, env=env)
    return executable, env


def _oracle_output(oracle):
    executable, env = oracle
    completed = subprocess.run([str(executable)], check=True, capture_output=True, text=True, env=env)
    lines = completed.stdout.strip().splitlines()
    return np.fromstring(lines[0], sep=" "), np.fromstring(lines[1], sep=" ").reshape(3, 3), np.fromstring(lines[2], sep=" ", dtype=int)


def test_full_pivot_solution_and_inverse_match_source_extracted_gfortran(gauss_oracle):
    expected_b, expected_a, _ = _oracle_output(gauss_oracle)
    a = np.asarray([[0.0, 2.0, 1.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    b = np.asarray([7.0, 3.0, 5.0])
    result = gauss_jordan_method(a, b)
    np.testing.assert_allclose(result.vector_b, expected_b, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(result.matrix_a, expected_a, rtol=0.0, atol=2e-15)
    np.testing.assert_array_equal(a, [[0, 2, 1], [1, 1, 0], [2, 0, 1]])


def test_inplace_contract_and_exact_zero_singular_guard():
    a = np.asarray([[2.0, 1.0], [1.0, 3.0]])
    b = np.asarray([1.0, 2.0])
    result = gauss_jordan_method(a, b, inplace=True)
    assert result.matrix_a is a and result.vector_b is b
    with pytest.raises(GaussJordanError, match="inversible"):
        gauss_jordan_method(np.zeros((2, 2)), np.ones(2))


def test_passive_error_both_where_arms_match_source_extracted_gfortran(gauss_oracle):
    _, _, expected = _oracle_output(gauss_oracle)
    current = np.zeros((3, 2, 7))
    previous = np.zeros_like(current)
    vegetation = np.asarray([[0.25, 0.75], [1.0, 0.0], [0.5, 0.5]])
    previous[:, 0, 6] = [100.0, 1e-10, 4.0]
    previous[:, 1, 6] = [20.0, 1e-10, 6.0]
    current[:, 0, 6] = [101.0, 2e-10, 4.01]
    current[:, 1, 6] = [19.0, 3e-10, 6.01]
    actual = error_l1_passive(current, previous, vegetation, 0.2)
    np.testing.assert_array_equal(actual.astype(int), expected)


def test_passive_pool_shape_and_fortran_index_guards():
    values = np.zeros((1, 1, 6))
    with pytest.raises(IndexError, match="ipassive_pool"):
        error_l1_passive(values, values, np.ones((1, 1)), 1.0)
    with pytest.raises(ValueError, match="veget_max"):
        error_l1_passive(np.zeros((2, 1, 7)), np.zeros((2, 1, 7)), np.ones((1, 1)), 1.0)
