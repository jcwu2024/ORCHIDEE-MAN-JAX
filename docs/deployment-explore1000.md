# Explore1000 Deployment

Explore1000 is the current remote compute platform for this project. The old
qhcess/Cancon server is used only for historical Fortran provenance or
explicitly approved trace work.

## Safety and Topology

- Connect through the user's `cln01` SSH alias.
- Write only under `/WORK/liwei_work/jcwu/`.
- Never run computation on `cln01`.
- Use `test01` through `test04` only for small correctness and compatibility
  smoke tests. They are shared resources and must not be used for accepted
  performance benchmarks.
- Use `gln01` for small GPU tests after checking `nvidia-smi`.
- Use `cnmix` or `cnall` for paid CPU jobs and `gnall` for paid GPU jobs.
- Before every `sbatch`, obtain explicit approval for resources, finite wall
  time, working directory, command, and worst-case cost.

Current rates used for estimates are CNY 0.07 per allocated CPU core-hour and
CNY 2.20 per allocated GPU-hour. Treat CPU and GPU charges as additive unless
the center confirms otherwise.

Accepted CPU performance measurements must run inside a Slurm allocation with
fixed CPU resources. Measure cold compilation and repeated hot execution in
the same process and allocation, and record the node, allocated CPUs, thread
environment, JAX version, command, and workload. A test-node wall time is only
diagnostic because other users can preempt effective CPU time.

## Checkout Transfer

GitHub access from the cluster is unreliable. Transfer a Git bundle from the
workstation instead of using third-party mirrors.

On the workstation:

```powershell
git bundle create ORCHIDEE-MAN-JAX-daily.bundle research/daily-coarse-graining
git bundle verify ORCHIDEE-MAN-JAX-daily.bundle
scp ORCHIDEE-MAN-JAX-daily.bundle cln01:/WORK/liwei_work/jcwu/transfer/
```

For the initial cluster checkout:

```bash
cd /WORK/liwei_work/jcwu
git clone --branch research/daily-coarse-graining \
  /WORK/liwei_work/jcwu/transfer/ORCHIDEE-MAN-JAX-daily.bundle \
  ORCHIDEE-MAN-JAX
cd ORCHIDEE-MAN-JAX
git rev-parse --short HEAD
```

For later updates:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
git fetch /WORK/liwei_work/jcwu/transfer/ORCHIDEE-MAN-JAX-daily.bundle \
  research/daily-coarse-graining
git merge --ff-only FETCH_HEAD
git rev-parse --short HEAD
```

Stop if the cluster checkout has local changes or diverged history. Never use
an implicit merge to hide server-side edits.

## Environments

Canonical project-scoped environments:

- CPU: `/WORK/liwei_work/jcwu/.venvs/orcjax_cpu`
- GPU: `/WORK/liwei_work/jcwu/.venvs/orcjax_gpu`

The CPU environment is uv-managed and pinned to CentOS 7-compatible wheels.
Build or reconcile it on `cln01`:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
bash scripts/hpc/bootstrap_orcjax_cpu.sh
```

Compute nodes have no package-index access and must consume the shared
environment read-only. The general project `uv.lock` currently targets newer
JAX and may resolve wheels incompatible with the cluster's glibc; do not use
it to replace the accepted CPU compatibility profile without a new gate.

The canonical `orcjax_gpu` environment has not yet been frozen. Historical
`jc_gpu`, `orcj_gpu`, and `orcj_gpu_compat` environments are validation assets,
not project-wide defaults. Keep them read-only until `orcjax_gpu` reproduces
the accepted V100 compatibility gate with the final training dependencies.

## Workload Policy

- Teacher shard generation: persistent CPU workers by default.
- Neural-network training: GPU after the pilot dataset and architecture are
  frozen.
- Neural inference: retain both CPU and GPU support; choose from measured
  latency, batched throughput, and cost.
- Full 669-point Teacher acceptance: one landpoint per process or scheduler
  array task with isolated checkpoints and outputs.

The authoritative job-script and cost workflow is maintained by the local
`explore1000-job-script` skill.
