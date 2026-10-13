# Scalar Audit Execution Notes

The successful saved-array audit ran as Slurm job `15852254` on `cntest`,
node `ibtest02`, with one CPU for 23 seconds. Its receipt is
`scalar-audit-receipt-v3.json` (SHA-256
`cb1331edb7094e74ec1814c6813a0eee218c58ea4e89f1c0c1f630e981479431`).
The exact staged script snapshot is retained as
`audit_cpu_scalar_admission-v3.py`, SHA-256
`90a1056f7ace3d7ecb7a3e322dbbd8f69e52302ed3ebe19b0caa111568985ff6`,
18,683 bytes. The receipt records the absolute Python command, the pinned
checkout in `PYTHONPATH`, and observed CPU affinity. The reusable working-tree
script later gained a direct-invocation `ROOT` bootstrap; that source is now
SHA-256 `0b44af19e7143572204ca173ffce7c1d3d4471d73e270b906ab845601bfebefe`.
It is not the script snapshot used for this audit.

The authoritative successful remote command was:

```bash
/rmprog/slurm/v22.05.7/bin/srun -p cntest -N 1 -n 1 -c 1 -t 00:05:00 \
  --cpu-bind=cores \
  --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3 \
  --export=ALL,PYTHONPATH=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1 \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/staging/cpu48-b64-v4-20261014/scalar-audit/audit_cpu_scalar_admission-v3.py \
  --scalar-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar \
  --gpu-reference-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-round4/model \
  --output /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/staging/cpu48-b64-v4-20261014/scalar-audit/scalar-audit-receipt-v3.json
```

Captured stdout was `{"status":"saved_arrays_audited","GPU_gate_count":64,"failed_gate_count":3,"reset_arrays_recomputed":false,"receipt_sha256":"cb1331edb7094e74ec1814c6813a0eee218c58ea4e89f1c0c1f630e981479431"}`; remote exit status was `0`.

The receipt's `audit_execution.scheduler_launch_command` field is a shorthand
with a relative Python script argument that was not the command actually run.
Use the absolute-path command above and `audit_execution.command` as the
authoritative launch evidence; the receipt bytes remain unchanged. The audit
is an internal scalar/GPU comparison, not scientific acceptance, and frozen-
reset arrays were not recomputed.

The initial launch attempt used the same one-CPU `cntest` request but omitted
the pinned checkout from `PYTHONPATH`:

```bash
/rmprog/slurm/v22.05.7/bin/srun -p cntest -N 1 -n 1 -c 1 -t 00:05:00 \
  --cpu-bind=cores \
  --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3 \
  --export=ALL,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1 \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/staging/cpu48-b64-v4-20261014/scalar-audit/audit_cpu_scalar_admission-v3.py \
  --scalar-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar \
  --gpu-reference-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-round4/model \
  --output /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/staging/cpu48-b64-v4-20261014/scalar-audit/scalar-audit-receipt-v3.json
```

That attempt exited `1` with `ModuleNotFoundError: No module named 'scripts'`
before loading any saved arrays, and wrote no receipt. The successful command
added
`PYTHONPATH=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3`.
This setup failure is separate from the successful audit result.
