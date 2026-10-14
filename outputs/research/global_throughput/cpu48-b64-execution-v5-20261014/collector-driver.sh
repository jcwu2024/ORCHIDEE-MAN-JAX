set -euo pipefail
E=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v5-20261014
exec /rmprog/slurm/v22.05.7/bin/salloc --partition=cntest --nodes=1 --ntasks=1 --cpus-per-task=1 --time=00:05:00 /rmprog/slurm/v22.05.7/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=cores bash "$E/collector-on-node.sh"
