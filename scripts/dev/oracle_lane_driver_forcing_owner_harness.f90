program driver_forcing_owner_oracle
  use readdim2, only: forcing_read_interpol, data_full, i_index, j_index, daily_interpol, oracle_watchout, &
       oracle_contfrac, oracle_wind_n, oracle_high_solar
  implicit none
  integer, parameter :: iim=2,jjm=2,ttm=24
  integer :: kindex(4),nbindex,unit,step,nsteps,split,spread,itau_base
  real :: dt_force
  logical :: conserve
  character(len=32) :: scenario
  integer :: neighbours(iim,jjm,8)
  real :: lon(iim,jjm),lat(iim,jjm),zlev(iim,jjm),zlevuv(iim,jjm)
  real :: swdown(iim,jjm),coszang(iim,jjm),rainf(iim,jjm),snowf(iim,jjm),tair(iim,jjm)
  real :: u(iim,jjm),v(iim,jjm),qair(iim,jjm),pb(iim,jjm),lwdown(iim,jjm),contfrac(iim,jjm)
  real :: resolution(iim,jjm,2),swnet(iim,jjm),eair(iim,jjm),peta(iim,jjm),peqa(iim,jjm)
  real :: petb(iim,jjm),peqb(iim,jjm),cdrag(iim,jjm),ccanopy(iim,jjm)
  lon=reshape([-30.,0.,30.,60.],shape(lon)); lat=reshape([-20.,10.,30.,60.],shape(lat))
  allocate(data_full(iim,jjm),i_index(iim),j_index(jjm)); i_index=[1,2];j_index=[1,2]
  call get_command_argument(2,scenario)
  daily_interpol=trim(scenario)=='daily'.or.trim(scenario)=='daily_watchout'.or. &
       trim(scenario)=='daily_wrap'.or.trim(scenario)=='high_daily'.or. &
       trim(scenario)=='daily_split1'
  oracle_watchout=trim(scenario)=='watchout'.or.trim(scenario)=='daily_watchout'
  oracle_contfrac=trim(scenario)/='no_contfrac'
  oracle_wind_n=trim(scenario)/='scalar_wind'
  oracle_high_solar=trim(scenario)=='high_daily'.or.trim(scenario)=='high_standard'
  conserve=trim(scenario)=='netrad'
  itau_base=1
  if(daily_interpol) then
    split=24; spread=6; dt_force=86400.; nsteps=48
    if(trim(scenario)=='daily_split1') then; split=1; spread=1; nsteps=2; endif
  else if(trim(scenario)=='hourly') then
    split=1; spread=1; dt_force=3600.; nsteps=2
  else
    split=4; spread=2; dt_force=21600.; nsteps=8
  endif
  if(trim(scenario)=='daily_wrap') then; itau_base=1; nsteps=589; endif
  if(trim(scenario)=='wrap') then; itau_base=24; nsteps=5; endif
  kindex=0; nbindex=0
  call forcing_read_interpol('memory',0,0,split,spread,conserve,0.,dt_force,iim,jjm,lon,lat,zlev,zlevuv,ttm,swdown,coszang,rainf,snowf,tair,u,v,qair,pb,lwdown,contfrac,neighbours,resolution,swnet,eair,peta,peqa,petb,peqb,cdrag,ccanopy,kindex,nbindex,1)
  open(newunit=unit,file=trim(command_argument()),status='replace',action='write')
  call emit(unit,'contfrac',0,contfrac)
  call emit(unit,'resolutionx',0,resolution(:,:,1)); call emit(unit,'resolutiony',0,resolution(:,:,2))
  call emit_int(unit,'neighbours',0,neighbours)
  do step=1,nsteps
    call forcing_read_interpol('memory',itau_base+(step-1)/split,mod(step-1,split)+1,split,spread,conserve,0.,dt_force,iim,jjm,lon,lat,zlev,zlevuv,ttm,swdown,coszang,rainf,snowf,tair,u,v,qair,pb,lwdown,contfrac,neighbours,resolution,swnet,eair,peta,peqa,petb,peqb,cdrag,ccanopy,kindex,nbindex,1)
    call emit(unit,'tair',step,tair); call emit(unit,'qair',step,qair); call emit(unit,'swdown',step,swdown)
    call emit(unit,'rainf',step,rainf); call emit(unit,'snowf',step,snowf)
    call emit(unit,'pb',step,pb); call emit(unit,'u',step,u); call emit(unit,'v',step,v); call emit(unit,'lwdown',step,lwdown)
    call emit(unit,'zlev',step,zlev); call emit(unit,'zlevuv',step,zlevuv)
    call emit(unit,'swnet',step,swnet); call emit(unit,'eair',step,eair); call emit(unit,'petacoef',step,peta)
    call emit(unit,'peqacoef',step,peqa); call emit(unit,'petbcoef',step,petb); call emit(unit,'peqbcoef',step,peqb)
    call emit(unit,'cdrag',step,cdrag); call emit(unit,'ccanopy',step,ccanopy)
  end do
  close(unit)
contains
  function command_argument() result(value); character(len=1024)::value; call get_command_argument(1,value); end
  subroutine emit(unit,name,step,value)
    integer,intent(in)::unit,step; character(len=*),intent(in)::name; real,intent(in)::value(:,:); integer::i,j
    do j=1,size(value,2);do i=1,size(value,1);write(unit,'(A,",",I0,",",ES26.17E3)')trim(name),step,value(i,j);enddo;enddo
  end
  subroutine emit_int(unit,name,step,value)
    integer,intent(in)::unit,step,value(:,:,:); character(len=*),intent(in)::name; integer::i,j,k
    do k=1,size(value,3);do j=1,size(value,2);do i=1,size(value,1)
      write(unit,'(A,"_",I0,",",I0,",",I0)')trim(name),k,step,value(i,j,k)
    enddo;enddo;enddo
  end
end program
