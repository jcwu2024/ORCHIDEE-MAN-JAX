set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
exec /rmprog/slurm/v22.05.7/bin/salloc --partition=cntest --nodes=1 --ntasks=1 --cpus-per-task=1 --time=00:05:00 /rmprog/slurm/v22.05.7/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=cores bash "$P/runtime/transfer/global-cpu48-b64-v5-0100c8b09be5/free-check-on-node.sh"
