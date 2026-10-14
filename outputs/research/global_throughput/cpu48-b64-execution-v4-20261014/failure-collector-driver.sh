set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
exec /rmprog/slurm/v22.05.7/bin/salloc --partition=cntest --nodes=1 --ntasks=1 --cpus-per-task=1 --time=00:10:00 /rmprog/slurm/v22.05.7/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=cores bash "$P/runtime/transfer/global-cpu48-b64-v4-60dccbddde10/failure-collector-on-node.sh"
