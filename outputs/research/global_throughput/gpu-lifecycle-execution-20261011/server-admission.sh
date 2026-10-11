set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RT=$PROJECT/runtime
hostname
date -u +%FT%TZ
git --version
git --git-dir="$RT/deployment/repository.git" show-ref --heads
/rmprog/slurm/v22.05.7/bin/sinfo -p gnall -N -o '%N|%T|%C|%O|%G'
/rmprog/slurm/v22.05.7/bin/scontrol show node ibc13b03n02
test -d "$RT/logs" && test -w "$RT/logs"
echo SHARED_LOG_PARENT_WRITABLE
test ! -e "$RT/worktrees/global-throughput-gpu-lifecycle-v1"
echo LIFECYCLE_CHECKOUT_ABSENT
sha256sum "$RT/deployment/registry.json"
for path in \
  "$RT/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl" \
  "$RT/outputs/research/global_throughput/gpu-round4/model" \
  "$RT/outputs/research/global_throughput/server-benchmark/global-throughput-v2-2b14506/gpu-b64/candidate-first" \
  "$RT/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1/capacity" \
  "$RT/outputs/research/global_throughput/round4-inputs/selected64-assets-v2"; do
  test -e "$path"
  printf 'existing_asset=%s\n' "$path"
done
