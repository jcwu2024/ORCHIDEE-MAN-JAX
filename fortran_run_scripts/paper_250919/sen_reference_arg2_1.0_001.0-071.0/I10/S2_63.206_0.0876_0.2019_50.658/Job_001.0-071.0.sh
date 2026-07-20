#!/bin/ksh
#qsub -l nodes=1:ppn=1 -q medium site

set -vx

MODELDIR=/public/share/qhcess/qhcess2/User/zzhao/modipsl_250919_dev/
#MODELDIR=/public/share/qhcess/qhcess2/User/zzhao/modipsl_240321_dev/
BINDIR=${MODELDIR}bin/
XIOSSERVER=${MODELDIR}modeles/XIOS/bin/
XIOSDIR=/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_250919/peat-leak_xml_zz/
RUNDEF=/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_250919//sen/arg2_1.0/Job/001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658/run.def_63.206_0.0876_0.2019_50.658
OUTLOC=/public/share/qhcess/qhcess2/User/zzhao/OUT/orc_calibrate_250919_sen/arg2_1.0/001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658/

#IF restart
#STARTDIR_init=/home/orchidee03/vnaipal/LEAK_test2/out1179/
#STARTDIR_init=/home/orchidee01/zhezhao/OUT/lateral_vn1.1/

FORDIR=/public/home/qhcess2/Share/Forcing/CRUNCEPv8/zip_twodeg/
FORCEWTFILE=/public/share/qhcess/qhcess2/User/zzhao/forcing_VN/
CO2FILE=/public/share/qhcess/qhcess2/User/zzhao/INPUTDIR_ZZ/global_co2_ann_1700_2017.txt

INPUTDIR_LW=/public/home/qhcess2/User/wli/MICT_BIOE/Input/
INPUTDIR_ZZ=/public/share/qhcess/qhcess2/User/zzhao/INPUTDIR_ZZ/

#- function to change the run.def
remplace()
{
sed "/$1/D" run.def > r2
cat <<END >>r2 
$1=$2 
END
mv r2 run.def
}

if [ ! -d ${OUTLOC} ] ; then
    mkdir -p ${OUTLOC}
fi

  cd ${OUTLOC}
  cp ${RUNDEF} run.def
  echo dir  ${OUTLOC} ok

  cp ${BINDIR}orchidee_ol orchidee.e
  #cp ${BINDIR}forcesoil forcesoil.e
  cp ${XIOSSERVER}xios_server.exe xios_server.exe
  cp ${XIOSDIR}*.xml .

  remplace XIOS_ORCHIDEE_OK y

west_bound=-180.0
east_bound=-178.0
south_bound=-20.0
north_bound=-18.0
 
  remplace LIMIT_WEST $west_bound
  remplace LIMIT_EAST $east_bound
  remplace LIMIT_NORTH $north_bound
  remplace LIMIT_SOUTH $south_bound


#set mangrove PFT
remplace IMPOSE_VEG y
remplace LAND_COVER_CHANGE n
remplace SECHIBA_VEGMAX__00001 0.0
remplace SECHIBA_VEGMAX__00002 0.0
remplace SECHIBA_VEGMAX__00003 0.0
remplace SECHIBA_VEGMAX__00004 0.0
remplace SECHIBA_VEGMAX__00005 0.0
remplace SECHIBA_VEGMAX__00006 0.0
remplace SECHIBA_VEGMAX__00007 0.0
remplace SECHIBA_VEGMAX__00008 0.0
remplace SECHIBA_VEGMAX__00009 0.0
remplace SECHIBA_VEGMAX__00010 0.0
remplace SECHIBA_VEGMAX__00011 0.0
remplace SECHIBA_VEGMAX__00012 0.0
remplace SECHIBA_VEGMAX__00013 0.0
remplace SECHIBA_VEGMAX__00014 1.0


# add water table file-VN
  remplace wtd_filename ${FORCEWTFILE}WTM_forcing_pos.txt
  remplace wtd_filename2 ${FORCEWTFILE}WTM_forcing_diff.txt
# tide module-VN
#  remplace tides y


  remplace ATM_CO2 286.42
  remplace TIME_LENGTH 1Y

  remplace POPDENS_FILE ${INPUTDIR_LW}Population/popd_1901.nc 
  remplace RATIO_FILE   ${INPUTDIR_LW}ratio_GFED4s_MICTRef_Opt6.nc 

  remplace SOILALB_FILE      ${INPUTDIR_LW}soils_param.nc
  remplace ALB_BG_FILE       ${INPUTDIR_LW}alb_bg_modisopt_2D.nc
  remplace TOPOGRAPHY_FILE   ${INPUTDIR_LW}cartepente2d_15min.nc
  remplace ROUTING_FILE      ${INPUTDIR_LW}routing.nc
  remplace IRRIGATION_FILE   ${INPUTDIR_LW}floodplains.nc
  remplace REFTEMP_FILE      ${INPUTDIR_LW}reftemp.nc

  remplace SOIL_REFSOC_FILE    ${INPUTDIR_LW}refSOC_NCSCD_linear_05deg_v5.nc
  remplace SOIL_REFSOC_1d_FILE ${INPUTDIR_LW}refSOC_NCSCD_05deg_v3_0-0.3m.nc

  remplace RESTART_FILEOUT driver_restart.nc
  remplace SECHIBA_rest_out sechiba_restart.nc
  remplace STOMATE_RESTART_FILEOUT stomate_restart.nc
  remplace OUTPUT_FILE sechiba_history.nc 
  remplace STOMATE_OUTPUT_FILE stomate_history.nc 
  remplace STOMATE_IPCC_OUTPUT_FILE stomate_ipcc_history.nc

  config_SOCinsul=y
  config_use_refSOC=y
  config_use_refSOC_hydrol=n
  config_dgvm=n

  activateVD=$config_dgvm # avtivate DGVM or not
  remplace STOMATE_OK_DGVM $activateVD
  if [ "$activateVD" = "y" ];then
    remplace LPJ_GAP_CONST_MORT n
    remplace AGRICULTURE n
    remplace HARVEST_AGRI n
    remplace VEGET_UPDATE 0Y
  else
    remplace LPJ_GAP_CONST_MORT y
    remplace AGRICULTURE y
    remplace HARVEST_AGRI y
    remplace VEGET_UPDATE 0Y  #remplace VEGET_UPDATE 1Y
  fi


##qcj++ peatland
  remplace dyn_nroot_larix y
  remplace cryoturbate n
  remplace use_new_cryoturbation n
  remplace FIRE_DISABLE y ### if =y, fire module will not be called
  config_peathydro=y
  peathydro=$config_peathydro
  remplace PEAT_HYDRO $peathydro
  if [ "$peathydro" = "y" ]; then
     remplace NSTM 4
     remplace NVM 14
     remplace PREF_SOIL_VEG__00014 4
     remplace PFT_TO_MTC__00014 15
     remplace PEAT_NODR y
     remplace OK_RU2PEAT y
     remplace OK_WT_AB y
     remplace max_wt_ab 100.
     #remplace VCMAX25__14 46.7
     #remplace SLA__00014 0.0153
     #remplace ARJV__00014 2.805
     remplace PFT_TO_MTC__00014 2
     remplace IS_PEAT__00014 y
  fi

##vn++ tides
  config_tides=y
  tides=$config_tides
  remplace tides $tides
  if [ "$tides" = "y" ]; then
     remplace NSTM 6
     remplace NVM 14
     #remplace PREF_SOIL_VEG__00014 6
     remplace PREF_SOIL_VEG__00014 4
     remplace PFT_TO_MTC__00014 15
     remplace PEAT_NODR y
     remplace OK_RU2PEAT y
     remplace OK_WT_AB y
     remplace max_wt_ab 100.
     #remplace VCMAX25__14 32.
     #remplace SLA__00014 0.0153
     #remplace SLA_MAX__00014 0.0153
     #remplace SLA_MIN__00014 0.0153
     #remplace ARJV__00014 3.125
     #rempalce MAINT_RESP_SLOPE_C__00014 0.12
     remplace PFT_TO_MTC__00014 2
     remplace IS_PEAT__00014 y
  fi

##zhaoz: mangrove
  #remplace CONTROL_INUDATE_MIN 1
  #remplace CONTROL_SALINITY_MIN 1

  remplace VEGETATION_FILE   ${INPUTDIR_ZZ}PFT1860_mangr_025deg.nc


#############Peat carbon module
####popular two-layered module
  remplace OK_PEAT n   ###NOTE:If OK_PEAT=y, OK_PC need to be n  
###multi-layerd module
  remplace OK_PC n
  remplace PERMA_PEAT y
#
##dynamic peat
  remplace TOPMODEL_NEW n

  remplace TOPMODEL_NEW_FILE ${INPUTDIR_ZZ}catch2_wtd_vkq_socoff_10.nc
  remplace PEAT_OCCUR n
  numyears=360  #30 years
  remplace MONTHS_NUM $numyears
#  remplace SAT_DURATION 270 #$numyears-30
###Summer Water balance threshold for peatland expanding
   remplace SAT_GSL 1.0

  remplace dynpeat_PWT n
  remplace PWT_LIM 60 ##mm/yr
###Carbon threshold for peatland expanding
  remplace dynpeat_PC n
  remplace PC_LIM  50  ##kg/m2
#
  remplace DYN_PEAT n
###peat turnover time, t0
  remplace TAU_PEAT 3.1536E7
###the e-folding depth of intrinsic turnover rates
  remplace Z_TAU 1.5
###frac1 thershold of C amount
  remplace FRAC1 0.70
#  ###frac2 flux of C
  remplace FRAC2 0.10
###number of layers to calculate mean WT for newtopmodel
  remplace NUMLAYERS 10
##########LEAK
  remplace OK_LEAK y
#  remplace DANS_RESTART n # use restart of MICT
  remplace TF_DOC y  #Calculate thoughfall DOC
  remplace RIVER_ROUTING y
  remplace DO_FLOODPLAINS n
  remplace POOR_SOILS y
  remplace PRIMING y
  remplace DO_FLOODINFILT n
  remplace NEW_FLOOD_SCHEME n
  remplace DO_STREAM_SWELL n
  remplace cryoturbate_doc_POC n
  remplace use_new_cryoturbation n

  remplace SOILCLASS_FILE ${INPUTDIR_ZZ}test.nc

  remplace TOPM_CALCUL n

  remplace USE_SOILC_TEMPDIFF $config_SOCinsul
  remplace use_refSOC $config_use_refSOC
  remplace use_refSOC_hydrol $config_use_refSOC_hydrol

  remplace ROUGH_DYN y
  remplace frozen_respiration_func 1

  remplace LEAFAGECRIT__00006 160
  remplace LEAFAGECRIT__00008 220
  remplace LEAFAGECRIT__00009 120
  remplace LEAFAGECRIT__00010 80
  remplace SENESCENCE_TEMP_C__00006 16
  remplace SENESCENCE_TEMP_C__00008 14
  remplace SENESCENCE_TEMP_C__00009 10
  remplace SENESCENCE_TEMP_C__00010 5
  remplace LEAFFALL__00006 30
  remplace LEAFFALL__00008 5
  remplace HUM_MIN_TIME__00010 36
  remplace NOSENESCENCE_HUM__00010 0.6
  remplace VCMAX25__00002 50

  remplace VCMAX25__00003 50
  remplace VCMAX25__00013 50

  remplace VCMAX25__00010 55
  remplace VCMAX25__00011 25
  remplace VCMAX25__00012 50
  remplace SLA__00010 4.2E-2
  remplace SLA__00011 4.1E-2
  remplace SLA_MAX__00010 4.2E-2
  remplace SLA_MAX__00011 4.1E-2
  remplace SLA_MIN__00010 4.2E-2
  remplace SLA_MIN__00011 4.1E-2


  remplace STOMATE_FORCING_NAME NONE #stomate_forcing.nc
  remplace STOMATE_CFORCING_NAME NONE #stomate_Cforcing.nc
  remplace STOMATE_CFORCING_PF_NM NONE #stomate_Cforcing_permafrost.nc

  rm driver_restart.nc sechiba_restart.nc stomate_restart.nc

#IF restart
#  cp ${STARTDIR_init}sechiba_restart.nc sechiba_start.nc
#  cp ${STARTDIR_init}stomate_restart.nc stomate_start.nc
#  cp ${STARTDIR_init}driver_restart.nc driver_start.nc
#  remplace RESTART_FILEIN driver_start.nc
#  remplace SECHIBA_restart_in sechiba_start.nc
#  remplace STOMATE_RESTART_FILEIN stomate_start.nc

#IF not restart, that is to say if start to run from 0   
  remplace RESTART_FILEIN NONE 
  remplace SECHIBA_restart_in NONE 
  remplace STOMATE_RESTART_FILEIN NONE 

let num_CPU_ORC=${BATCH_NUM_PROC_TOT}-1
cat << END > ${OUTLOC}run_file
-np ${num_CPU_ORC} orchidee.e
-np 1 xios_server.exe
END
  
  let i=0
  let k=100
 
  YEAR0=1961
  #YEAR0=1966
  YEAR=$YEAR0

#################################### FULL ORCHIDEE
IITER=50
#IITER=31   #Run 31 years, at the end of the full orchidee process, YEAR=1961, thus 1961 will be used in forcesoil processes
####FULL ORCHIDEE, yearly output
cp file_def_orchidee_pods_year_240405.xml file_def_orchidee.xml

  remplace FORCESOIL_STEP_PER_YEAR 365
  
  while [ $i -lt $IITER ] ; do
   echo i  $i IIETER $IITER k $k
      mv driver_restart.nc driver_start.nc
      mv sechiba_restart.nc sechiba_start.nc
      mv stomate_restart.nc stomate_start.nc

      if [ $i -eq 1 ] ; then
  	  remplace RESTART_FILEIN driver_start.nc
  	  remplace SECHIBA_restart_in sechiba_start.nc
	    remplace STOMATE_RESTART_FILEIN stomate_start.nc
      fi

      #remplace FORCING_FILE ${FORDIR}cruncep_onedeg_${YEAR}.nc
      #remplace FORCING_FILE ${FORDIR}cruncep_halfdeg_${YEAR}.nc
      remplace FORCING_FILE ${FORDIR}cruncep_twodeg_${YEAR}.nc

      #if [ $YEAR -eq 2015 ] ; then
      #remplace FORCING_FILE ${FOR_RP}
      #cp file_def_orchidee_4site_HH_240315.xml file_def_orchidee.xml
      #fi

      CO2=$(grep ${YEAR} ${CO2FILE} |awk '{print $2}')
      remplace ATM_CO2 $CO2    #use YEAR_real corresponding transient  CO2

      remplace JUDGE_PC n

      cp run.def run.def.${YEAR}

      #-------------run orchidee--------------
      time ./orchidee.e > out_orchidee_${YEAR}.txt

      OUT_TEMPO=${OUTLOC}out${YEAR}/
      if [ ! -d ${OUT_TEMPO} ] ; then
	    mkdir -p ${OUT_TEMPO}
      fi

      if [ ! -e stomate_history.nc ] ; then
        exit
      fi

      #mv sechiba_history2.nc sechiba_history_${YEAR}.nc
      mv stomate_history.nc stomate_history_${YEAR}.nc

      #mv out_orchidee_${YEAR}.txt ${OUT_TEMPO}
      rm -rf out_orchidee_${YEAR}.txt
      mv run.def.${YEAR} ${OUT_TEMPO}
      mv used_run.def ${OUT_TEMPO}
      #mv out_orchidee_000? ${OUT_TEMPO}
      rm -rf out_orchidee_000?
      
    
      #zhaoz: mv temp file
      #mv ../tide_height.txt ${OUT_TEMPO}
      #mv ../frac_root_anoxia.txt ${OUT_TEMPO}

      cp driver_restart.nc ${OUT_TEMPO}
      cp sechiba_restart.nc ${OUT_TEMPO}
      cp stomate_restart.nc ${OUT_TEMPO}


      let k=k+1  
      let i=i+1
      let YEAR=YEAR+1
  done

rm -rf orchidee.e forcesoil.e xios_server.exe
mv out2010 z1
rm -rf out*
#rm -rf file*
rm -rf z1/out_orchidee_0000
