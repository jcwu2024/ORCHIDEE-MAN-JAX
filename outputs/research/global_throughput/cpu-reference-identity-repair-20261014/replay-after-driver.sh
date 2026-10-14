set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE=$P/runtime/diagnostics/global-cpu-reference-identity-20261014
OUT=$P/runtime/outputs/research/global_throughput/cpu-reference-identity-repair-20261014
cp "$OUT/after-overlay/benchmark_global_server_cpu.py" "$OUT/after-overlay/benchmark_global_batch.py" "$OUT/after-overlay/diagnose_global_cpu_references.py" "$CODE/scripts/dev/"
exec /rmprog/slurm/v22.05.7/bin/salloc --partition=cntest --nodes=1 --ntasks=1 --cpus-per-task=1 --time=00:05:00 /rmprog/slurm/v22.05.7/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=cores bash "$OUT/replay-on-node.sh" after
