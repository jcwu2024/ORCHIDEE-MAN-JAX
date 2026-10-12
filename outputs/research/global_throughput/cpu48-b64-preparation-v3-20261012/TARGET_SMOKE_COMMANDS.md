# test01 smoke command record

All commands below were run from the Windows global-throughput checkout with
the existing SSH identity. Empty stdout from `test` commands plus exit code 0
is recorded as an empty string. No `sbatch` or `srun` command was invoked.

## Read-only target inspection

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 test01 hostname
stdout: ibtest01
exit: 0

ssh -o BatchMode=yes -o ConnectTimeout=12 test01 'test -d /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput'
stdout: <empty>
exit: 0

ssh -o BatchMode=yes -o ConnectTimeout=12 test01 'test ! -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run'
stdout: <empty>
exit: 0

ssh -o BatchMode=yes -o ConnectTimeout=12 test01 '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python --version'
stdout: Python 3.11.15
exit: 0

ssh -o BatchMode=yes -o ConnectTimeout=12 test01 'test ! -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3'
stdout: <empty>
exit: 0
```

## Staging and hashes

Only these four files were copied from the detached local v3 verification
checkout to the unique staging subtree:

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 test01 'mkdir -p /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/dev /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/hpc'

scp -q "E:\ORCHIDEE-MAN-JAX\runtime\worktrees\global-throughput-cpu48-b64-v3-local-verification\scripts\dev\benchmark_global_server_cpu.py" "test01:/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/dev/benchmark_global_server_cpu.py"
scp -q "E:\ORCHIDEE-MAN-JAX\runtime\worktrees\global-throughput-cpu48-b64-v3-local-verification\scripts\dev\global_server_resources.py" "test01:/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/dev/global_server_resources.py"
scp -q "E:\ORCHIDEE-MAN-JAX\runtime\worktrees\global-throughput-cpu48-b64-v3-local-verification\scripts\hpc\global_cpu48_b64_coordinator.py" "test01:/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/hpc/global_cpu48_b64_coordinator.py"
scp -q "E:\ORCHIDEE-MAN-JAX\runtime\worktrees\global-throughput-cpu48-b64-v3-local-verification\scripts\hpc\slurm_global_server_cpu_48core.sh" "test01:/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/hpc/slurm_global_server_cpu_48core.sh"
```

Hash command and captured stdout:

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 test01 'sha256sum /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/dev/benchmark_global_server_cpu.py /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/dev/global_server_resources.py /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/hpc/global_cpu48_b64_coordinator.py /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/hpc/slurm_global_server_cpu_48core.sh'
a773dbe64fc21a93bca4e4fda952c37f97445fc0eaa9eda3b35c93c482aa91dc  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/dev/benchmark_global_server_cpu.py
c17f37f4d0acd888a9525154489e1815d3d24ff60c6fac030c73e359284dc271  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/dev/global_server_resources.py
95bc4b4d98609b134492489501763489a2c783c79713a0a83334a4d9d79af816  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/hpc/global_cpu48_b64_coordinator.py
577d5d08cc4e7862390661f02cbc38a9b9f247410c8db162ceb5e1b121e96349  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/hpc/slurm_global_server_cpu_48core.sh
exit: 0
```

## Dry-run commands

Launcher command:

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 test01 'env CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3 EXPECTED_CODE_COMMIT=03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8 SLURM_CPUS_PER_TASK=48 bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/hpc/slurm_global_server_cpu_48core.sh --dry-run'
exit: 0
stdout: target-launcher.stdout.txt
```

Coordinator command:

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 test01 '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/hpc/global_cpu48_b64_coordinator.py --output-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/coordinator-output --project-root /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX --code-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3 --python /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python --entry /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run/scripts/dev/benchmark_global_server_cpu.py --expected-code-commit 03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8 --prepared-input /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl --gpu-reference-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-round4/model --config /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3/configs/orchidee_man_250919.yaml --dry-run'
exit: 0
stdout: target-coordinator.stdout.json
```

## Cleanup

Before removal, `find .../test01-dry-run -mindepth 1 -maxdepth 4 -type f -print`
listed exactly the four files above. The captured cleanup command was:

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 test01 'rm -r -- /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run'
exit: 0

ssh -o BatchMode=yes -o ConnectTimeout=12 test01 'test ! -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v3-20261012/test01-dry-run'
stdout: <empty>
exit: 0
```
