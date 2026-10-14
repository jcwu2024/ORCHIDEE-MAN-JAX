set -u
S=/rmprog/slurm/v22.05.7/bin
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
R=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/15853683
date -Is
"$S/squeue" -j 15853683 -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'
"$S/scontrol" show job -dd 15853683
"$S/sacct" -j 15853683 --parsable2 --units=K --format=JobIDRaw,JobName,Partition,State,ExitCode,Elapsed,ElapsedRaw,AllocCPUS,AllocTRES,NodeList,TotalCPU,UserCPU,SystemCPU,MaxRSS
find "$R" -maxdepth 3 -type f -printf '%p %s bytes\n'
tail -n 120 "$P/runtime/logs/global_cpu48_b64_v4_15853683.txt"
for F in "$R/orchestration.json" "$R/coordinator.json" "$R/parent-receipt.json"; do
 if test -f "$F"; then cat "$F"; fi
done
