! Minimal extraction of vertical_soil.f90::vertical_soil_init lines 268-447.
program vertical_soil_geometry_oracle
  use iso_fortran_env, only: int64
  implicit none
  integer, parameter :: dp=kind(1.0d0), nblayermax=500
  integer :: i, ntmp, nslm, ngrnd, loc(1)
  real(dp) :: zmaxt, zmaxh, top, cst, geom, ratio_below, ratio, hh
  real(dp) :: ztmp(nblayermax+1), zint(nblayermax+1), dtmp(nblayermax+1)
  real(dp), allocatable :: znh(:), dnh(:), dlh(:), zlh(:), znt(:), dlt(:), zlt(:)
  character(len=64) :: arg

  call get_command_argument(1,arg); read(arg,*) zmaxt
  call get_command_argument(2,arg); read(arg,*) zmaxh
  call get_command_argument(3,arg); read(arg,*) top
  call get_command_argument(4,arg); read(arg,*) cst
  call get_command_argument(5,arg); read(arg,*) geom
  call get_command_argument(6,arg); read(arg,*) ratio_below
  if ((cst /= zmaxh) .and. (cst > zmaxh/2.0_dp)) cst=zmaxh

  ztmp=0.0_dp; dtmp=0.0_dp
  do i=1,nblayermax
    if (ztmp(i) < cst) then
      ztmp(i+1)=top*2.0_dp*(real(2_int64**i,dp)-1.0_dp)
      dtmp(i+1)=ztmp(i+1)-ztmp(i)
    else
      ztmp(i+1)=ztmp(i)+dtmp(i); dtmp(i+1)=dtmp(i)
    end if
  end do
  loc=minloc(abs(ztmp-zmaxh)); nslm=loc(1)
  allocate(znh(nslm),dnh(nslm),dlh(nslm),zlh(nslm))
  znh=ztmp(1:nslm); znh(nslm)=zmaxh
  dnh=dtmp(1:nslm); dnh(nslm)=ztmp(nslm)-ztmp(nslm-1)
  do i=1,nslm-1
    dlh(i)=(dnh(i)+dnh(i+1))/2.0_dp
  end do
  dlh(nslm)=dnh(nslm)/2.0_dp

  ntmp=nslm; ztmp=0.0_dp; ztmp(1)=top/2.0_dp
  do i=2,ntmp
    ztmp(i)=znh(i)
  end do
  hh=dnh(ntmp)/2.0_dp
  ztmp(ntmp)=ztmp(ntmp)-hh/2.0_dp
  ztmp(ntmp+1)=ztmp(ntmp)+hh*1.5_dp
  ztmp(ntmp+2)=ztmp(ntmp+1)+hh*2.0_dp
  ntmp=ntmp+2
  do i=ntmp,nblayermax
    if (ztmp(i) < geom) then
      ratio=1.0_dp
    else
      ratio=ratio_below
    end if
    ztmp(i+1)=ztmp(i)+ratio*(ztmp(i)-ztmp(i-1))
  end do
  zint=0.0_dp; zint(1)=top
  do i=2,nblayermax-1
    zint(i)=(ztmp(i)+ztmp(i+1))/2.0_dp
  end do
  zint(nslm-1)=(znh(nslm-1)+znh(nslm))/2.0_dp
  zint(nslm)=znh(nslm)
  zint(nblayermax)=ztmp(nblayermax)+(ztmp(nblayermax)-ztmp(nblayermax-1))/2.0_dp
  loc=minloc(abs(zint-zmaxt)); ngrnd=loc(1)
  allocate(znt(ngrnd),dlt(ngrnd),zlt(ngrnd))
  znt=ztmp(1:ngrnd); zlt=zint(1:ngrnd); dlt(1)=zint(1)
  do i=2,ngrnd
    dlt(i)=zint(i)-zint(i-1)
  end do
  zlh=zlt(1:nslm); zlt(ngrnd)=zmaxt; dlt(ngrnd)=zmaxt-zint(ngrnd-1)

  write(*,'(I0,1X,I0)') nslm,ngrnd
  write(*,'(*(ES25.16E3,1X))') znh
  write(*,'(*(ES25.16E3,1X))') dnh
  write(*,'(*(ES25.16E3,1X))') dlh
  write(*,'(*(ES25.16E3,1X))') zlh
  write(*,'(*(ES25.16E3,1X))') znt
  write(*,'(*(ES25.16E3,1X))') dlt
  write(*,'(*(ES25.16E3,1X))') zlt
end program vertical_soil_geometry_oracle
