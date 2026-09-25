# Bounded saved-executable comparison on an idle paid node

Status: prepared and target-shell dry-run passed; not approved or submitted.

The resumed free probe104560 started13:24CST through cln01. At13:47 it
remained in bootstrap, with20seconds cumulative Python CPU across23minutes,
waiting in Lustre calls. The same Python library file was readable from cln01
in0.11seconds. This does not establish the storage fault's cause, but supports
trying a different compute node rather than recompiling the model.

## Exact proposed allocation

- Partition: gnall,1node,2V100,8allocated CPUs,64GiB,30minutes maximum.
- Preferred node: ibc13b03n04. Scheduler snapshot13:44CST: IDLE,0allocated
  CPUs/GPUs,CPULoad0.01. Recheck before submission; do not request exclusivity.
- Working directory:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1`
  (unchanged c599236).
- Exact command:
  `/rmprog/slurm/v22.05.7/bin/sbatch --nodelist=ibc13b03n04 /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/job_shared_serialized_dual.sh`
- Maximum charge using current project rates and additive CPU/GPU billing:
  CPU8*0.5*0.07=0.28yuan; GPU2*0.5*2.20=2.20yuan; total2.48yuan.

`job_shared_serialized_dual.sh` maps the actual two Slurm CUDA selectors to
the persistent parent/worker; target-shell dry-runs cover0,1 and2,5 allocations.
It reads64bytes each from the existing runner, Python stdlib and saved profile,
with15second bounds per file. A failed read exits75 before model setup. This
is an early rejection for clear IO stalls, not proof all later reads are fast.
The outer Slurm bound is30minutes and inner model probe28minutes, with30second
termination grace. The known-free-only launcher has a separate paid staging
filename, leaving the active free launch script untouched.

The job loads existing trusted daily/T7/T30 binaries, compares two changing
updates and matched32/64 complete proposals, including IPC and AdamW. It does
not recompile model gradients, change architecture/loss, regenerate Teacher
data or retain a formal trained model. Exact comparison protocol remains in
`shared_split_serialized_dual_preparation_20260925.md`. Reports use the Slurm
job ID. Success is measured parity plus actual complete-update throughput;
no speedup is promised from the node change alone.
