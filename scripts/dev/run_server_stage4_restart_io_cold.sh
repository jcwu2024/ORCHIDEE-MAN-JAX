#!/usr/bin/env bash
set -eo pipefail

ROOT=/public/share/qhcess/qhcess2/User/jcwu/stage4_restart_oracle
MODELDIR="$ROOT/modeles/ORCHIDEE"
FIXTURE="$ROOT/fixtures/stage4_restart_io"
OUTLOC="$ROOT/outputs/stage4_restart_io"

mkdir -p "$OUTLOC"
rm -f "$FIXTURE/cold_sechiba.nc" "$FIXTURE/cold_stomate.nc" "$OUTLOC/cold_sechiba_restart.nc" "$OUTLOC/cold_stomate_restart.nc" "$OUTLOC/cold_run.log"
cp "$ROOT/fixtures/stage4_stomate_data/run.def" "$FIXTURE/run.def"
sed -i '/STOMATE_RESTART_FILEIN/d;/STOMATE_RESTART_FILEOUT/d;/SECHIBA_restart_in/d;/SECHIBA_rest_out/d;/XIOS_ORCHIDEE_OK/d' "$FIXTURE/run.def"
cat >> "$FIXTURE/run.def" <<EOF
XIOS_ORCHIDEE_OK=n
SECHIBA_restart_in=NONE
SECHIBA_rest_out=cold_sechiba.nc
STOMATE_RESTART_FILEIN=NONE
STOMATE_RESTART_FILEOUT=cold_stomate.nc
EOF

source /etc/profile
module purge
module load compiler/intel/intel-compiler-2020.1.217
module load mathlib/netcdf/4.4.1/intel
export LD_LIBRARY_PATH=/public/software/mpi/intelmpi/2017.4.239/intel64/lib:/public/software/mathlib/netcdf/4.4.1/intel/lib:/public/software/compiler/intel/intel-2020/compilers_and_libraries_2020.1.217/compiler/lib/intel64:/public/software/compiler/intel/intel-2020/compilers_and_libraries_2020.1.217/mkl/lib/intel64:/public/software/compiler/intel/intel-2020/compilers_and_libraries_2020.1.217/tbb/lib/intel64

cd "$FIXTURE"
"$ROOT/build/stage4_stomate_io_driver.exe" > "$OUTLOC/cold_run.log" 2>&1
test -f "$FIXTURE/cold_stomate.nc"
mv "$FIXTURE/cold_sechiba.nc" "$OUTLOC/cold_sechiba_restart.nc"
mv "$FIXTURE/cold_stomate.nc" "$OUTLOC/cold_stomate_restart.nc"
printf 'MODELDIR=%s\nOUTLOC=%s\n' "$MODELDIR" "$OUTLOC"
