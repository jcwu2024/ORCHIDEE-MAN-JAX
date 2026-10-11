# Slurm 22.05.7 CPU Job Fixture

The installed command reported `slurm 22.05.7` in
`outputs/T7-affine-paid-preparation-20261011/slurm22-compatibility-v1/actual-job-and-docs.v1.stdout.txt`.
This public fixture contains a field-trimmed record from the one actual visible
running job returned by `scontrol show job --details -o 15779975`, captured at
`2026-10-11T15:23:08.710398+00:00`. The raw capture is preserved at
`outputs/T7-affine-paid-preparation-20261011/slurm22-compatibility-v1/actual-visible-job.v1.stdout.txt`.
Values shown in the fixture are preserved from that record. User, account,
command, working-directory, log-path, and other unrelated fields are omitted.
An earlier `gnall`-only job-ID selection returned no visible running job; it
is not used as evidence.

The observed job was `RUNNING` on 2026-10-11, in partition `cnall`, on node
`ibc11b07n08`, with 48 allocated CPUs and no GPU (`GRES=` and no GPU TRES).
This is real CPU-only scheduler evidence for parser compatibility. The
`CPU_IDs` value is Slurm's allocation bitmap representation; tests bound and
count those IDs, but do not compare them one-by-one with process CPU affinity.
The record does not establish a paid batch environment's GPU visibility or
GPU allocation. GPU-shaped values used for admission tests are synthetic.

Tagged Slurm 22.05.7-1 source references used for interpreting the record:

- [`job_info.c` TRES allocation/request fallback](https://github.com/SchedMD/slurm/blob/slurm-22-05-7-1/src/api/job_info.c#L696)
- [`job_info.c` allocated node detail](https://github.com/SchedMD/slurm/blob/slurm-22-05-7-1/src/api/job_info.c#L808)
- [`job_info.c` CPU and GRES detail](https://github.com/SchedMD/slurm/blob/slurm-22-05-7-1/src/api/job_info.c#L851)
- [`slurm_protocol_defs.c`](https://github.com/SchedMD/slurm/blob/slurm-22-05-7-1/src/common/slurm_protocol_defs.c#L2807)
- [`env.c`](https://github.com/SchedMD/slurm/blob/slurm-22-05-7-1/src/common/env.c#L1162)
- [`sbatch` manual](https://github.com/SchedMD/slurm/blob/slurm-22-05-7-1/doc/man/man1/sbatch.1)
