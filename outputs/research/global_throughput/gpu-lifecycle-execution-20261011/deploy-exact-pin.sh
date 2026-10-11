set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RT=$PROJECT/runtime
CODE=$RT/worktrees/global-throughput-gpu-lifecycle-v1
STORE=$RT/deployment/repository.git
TRANSFER=$RT/transfer/gpu-lifecycle-20261011-3b4ae35
BUNDLE=$TRANSFER/global-throughput-3b4ae35.bundle
PIN=3b4ae35e695f85480ecb72461e4ba93f4534bbd8
BASE_REGISTRY=be35fd8949f8b1df0581424641b70398e1ff0a382baa5b9e9ee1dcfd966ecb97
test "$(realpath -m "$CODE")" = "$CODE"
test "$(realpath -m "$TRANSFER")" = "$TRANSFER"
test ! -e "$CODE"
test "$(sha256sum "$RT/deployment/registry.json" | awk '{print $1}')" = "$BASE_REGISTRY"
test "$(sha256sum "$BUNDLE" | awk '{print $1}')" = c41277078f7c05efac832a7f25136e5cc64b037f8594774c64e2af86fbfa4ed5
git --git-dir="$STORE" bundle verify "$BUNDLE"
git bundle list-heads "$BUNDLE"
test "$(git bundle list-heads "$BUNDLE")" = "$PIN refs/heads/research/global-throughput"
test "$(git --git-dir="$STORE" rev-parse refs/heads/research/global-throughput)" = c537f59f33568b6e5960254fe55fb307f82ca503
git --git-dir="$STORE" fetch "$BUNDLE" refs/heads/research/global-throughput:refs/heads/research/global-throughput
test "$(git --git-dir="$STORE" rev-parse refs/heads/research/global-throughput)" = "$PIN"
test "$(git --git-dir="$STORE" rev-parse refs/heads/main)" = 7cae3e02adad23674767a590fdf4d27f239ce065
test "$(git --git-dir="$STORE" rev-parse refs/heads/research/daily-coarse-graining)" = 7461227ad38bf8a4fb399c47cc588ce0e1b771c6
git clone --no-checkout --no-hardlinks --single-branch --branch research/global-throughput "$STORE" "$CODE"
cd "$CODE"
git checkout --detach "$PIN"
test "$(git rev-parse HEAD)" = "$PIN"
test -z "$(git status --porcelain --untracked-files=all)"
test ! -s .git/objects/info/alternates
bash -n scripts/hpc/slurm_global_gpu_lifecycle.sh
git rev-parse HEAD
git status --porcelain --untracked-files=all
printf 'exact_code_dir=%s\nno_alternates=true\n' "$CODE"
"$PROJECT/.venvs/orcjax_cpu/bin/python" -B "$TRANSFER/register-deployment.py"
