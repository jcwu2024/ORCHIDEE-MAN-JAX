module constantes
  implicit none
  integer, parameter :: r_std=8
  real(kind=r_std), parameter :: pi=3.1415926535897932384626433832795_r_std
  real(kind=r_std), parameter :: zero=0._r_std, un=1._r_std
  real(kind=r_std), parameter :: one_day=86400._r_std, val_exp=1.e30_r_std
  real(kind=r_std), parameter :: cte_molr=287.05_r_std, cte_grav=9.80665_r_std
  real(kind=r_std), parameter :: cp_air=1004.675_r_std
  integer :: numout=6
end module constantes

module defprec
end module defprec

module calendar
  implicit none
  real :: one_year=365.
end module calendar

module ioipsl_para
  implicit none
  interface bcast
    module procedure bcast_i0, bcast_i1, bcast_r0, bcast_r1, bcast_r2, bcast_l0
  end interface
  interface gather
    module procedure gather_i2, gather_r1, gather_r2
  end interface
contains
  subroutine bcast_i0(x); integer,intent(inout)::x; end
  subroutine bcast_i1(x); integer,intent(inout)::x(:); end
  subroutine bcast_r0(x); real,intent(inout)::x; end
  subroutine bcast_r1(x); real,intent(inout)::x(:); end
  subroutine bcast_r2(x); real,intent(inout)::x(:,:); end
  subroutine bcast_l0(x); logical,intent(inout)::x; end
  subroutine gather_i2(x,y); integer,intent(in)::x(:,:); integer,intent(out)::y(:,:); y=x; end
  subroutine gather_r1(x,y); real,intent(in)::x(:); real,intent(out)::y(:); y=x; end
  subroutine gather_r2(x,y); real,intent(in)::x(:,:); real,intent(out)::y(:,:); y=x; end
  subroutine scatter(x,y); integer,intent(in)::x(:); integer,intent(out)::y(:); y=x(1:size(y)); end
  subroutine ipslerr_p(level,where,m1,m2,m3)
    integer,intent(in)::level; character(len=*),intent(in)::where,m1,m2,m3
    if (level >= 3) error stop trim(where)//': '//trim(m1)//' '//trim(m2)//' '//trim(m3)
  end
  real function itau2date(itau,date0,dt); integer,intent(in)::itau; real,intent(in)::date0,dt; itau2date=date0+itau*dt/86400.; end
  subroutine ju2ymds(j,yy,mm,dd,ss); real,intent(in)::j; integer,intent(out)::yy,mm,dd; real,intent(out)::ss; yy=2001;mm=1;dd=1+int(j);ss=0.; end
  subroutine ymds2ju(yy,mm,dd,ss,j); integer,intent(in)::yy,mm,dd; real,intent(in)::ss; real,intent(out)::j; j=real(dd-1); end
end module ioipsl_para

module weather
  implicit none
contains
  subroutine weathgen_qsat_2d(iim,jjm,t,p,q)
    integer,intent(in)::iim,jjm; real,intent(in)::t(iim,jjm),p(iim,jjm); real,intent(out)::q(iim,jjm)
    real :: tl(iim,jjm),ti(iim,jjm),e(iim,jjm)
    tl=min(100.,max(t-273.15,0.)); ti=max(-60.,min(t-273.15,0.))
    e=100.*(merge(6.1078,6.109178,t>273.15)+tl*(.44365185+tl*(.014289458+tl*(.00026506485+tl*(.0000030312404+tl*(.000000020340809+tl*.000000000061368209)))))+ti*(.5034699+ti*(.018860134+ti*(.00041762237+ti*(.0000058247203+ti*(.000000048388032+ti*.00000000018388269))))))
    q=.622*e/max(p-(1.-.622)*e,.622*e)
  end
end module weather

module timer
end module timer

module grid
  implicit none
  integer, allocatable :: neighbours(:,:), neighbours_g(:,:), index_g(:)
  real, allocatable :: resolution(:,:), resolution_g(:,:), area(:), area_g(:), lon_g(:), lat_g(:)
contains
  subroutine grid_init(n,k,a,b); integer,intent(in)::n,k; character(len=*),intent(in)::a,b
    if(.not.allocated(neighbours)) allocate(neighbours(n,8),neighbours_g(n,8),resolution(n,2),resolution_g(n,2),area(n),area_g(n),index_g(n),lon_g(n),lat_g(n))
    neighbours=-1; resolution=1.; index_g=[(k,k=1,n)]; lon_g=0.;lat_g=0.
  end
  subroutine grid_stuff(n,ig,jg,lo,la,idx); integer,intent(in)::n,ig,jg,idx(:); real,intent(out)::lo(:),la(:); lo=0.;la=0.; end
end module grid

module mod_orchidee_para
  implicit none
  integer :: nbp_glo=1,nbp_loc=1,iim_g=1,jjm_g=1,ii_begin=1,ii_end=1,jj_begin=1,offset=0
end module mod_orchidee_para
