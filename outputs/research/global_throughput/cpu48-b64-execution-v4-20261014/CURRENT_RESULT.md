# CPU48 B64 v4 current result

The sole approved paid attempt, job **15853683**, ended **FAILED, exit 1:0** on
`ibc11b09n36` at 2026-10-14 09:56:55 Asia/Shanghai, after 50 seconds.
No complete B64 measurement or CPU/GPU throughput ratio is available.
Scientific acceptance remains **false**. This attempt is closed: no source
repair, second paid submission, extension, cancellation, or scalar regeneration
was performed.

## Immutable deployment and preflight boundary

The fresh detached checkout is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v4`
at `60dccbddde10be4295aab7e7fe55fa7864d3c3f3`; the 104-file physical tree
`75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba` and science SHA
`11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`
are unchanged. Bundle/pin/original proposal/admission hashes were verified;
the bundle prerequisite and all 15 actual harness blobs passed. The checkout
has no Git alternates and was still clean after the paid run and collector.

Free production-path preflight **15853600**, `cntest`, `ibtest01`, 1 CPU,
passed actual `candidate_identity` and `validate_cpu_reference_admission`
identity/seal checks with the real preflight job ID. The unchanged proposal
and launcher/coordinator/runner dry-run passed using production paths.
No JAX/model was imported/executed in that preflight. This evidence did not
exercise the real benchmark runner's post-import `load_reference` comparison.

The paid child subsequently logged `exec_boundary_ready` and
`resource_admission_passed` with the exact thread policy, 48 distinct physical
cores and parent `cpu=48` allocation before importing JAX. The real runner
then recorded JAX/jaxlib 0.4.38, one CPU device and x64 enabled. Its first
substantive fault was:

```text
ValueError: FULL64 CPU scalar admission candidate provenance differs from current execution identity
benchmark_global_server_cpu.py:411 -> global_server_reference.py:283
```

`batch/report.json` preserves the complete traceback. Its `phases`, `timing`
and `gates` are empty; no lower/compile/first-call/warm measurements or saved
candidate validation arrays/checkpoints were produced. The child ended after
43.821602583 s; coordinator scope was 43.900519609 s, both shorter than the
50-second scheduler parent allocation.

The saved `server_resources.candidate_provenance` and admission
`candidate_identity` summaries differ only in `job_id` (`15853683` versus
`null`). The same kind of job-ID difference passed the preflight and collector
admission validators. The failing `load_reference` full expected/current
identity pair was not independently serialized; these summaries do not
diagnose the cause of that failure. No broader diagnosis or repair was run.

## Independent failure collection and cost

Pinned `scripts/dev/collect_global_cpu48_b64.py`, SHA
`efcb32efa754bbdbb75936f358dc6cb5a286e6742948096bda3e3ba22d27a9e3`,
ran once using canonical CPU Python in free job **15853855**, `cntest`,
`ibtest01`, 1 CPU, affinity `[0]`. It exited successfully and produced
`incomplete_run_failure_preserved`, with claim scope
`failure accounting only; no throughput claim`. It revalidated the scalar
admission and bound the new and old parent rows separately; it did not run
the physical model or certify candidate array gates.

| Parent scope | Job | Seconds | Allocated core-hours | CNY at 0.05 | CNY at 0.07 |
| --- | --- | ---: | ---: | ---: | ---: |
| New paid allocation, charged once | 15853683 | 50 | 0.666666667 | 0.033333333 | 0.046666667 |
| Prior scalar source, separate and charged once | 15820467 | 2197 | 29.293333333 | 1.464666667 | 2.050533333 |

These are unverified billing scenarios, not invoices. The collector's generic
new-parent scope string mentions a scalar step; this batch-only run contains
no new scalar child. Whole-parent cost is `48 × 50 / 3600` core-hours, not the
shorter coordinator interval. Prior scalar reuse adds no new allocation
charge. Scheduler MaxRSS is blank/zero; `/usr/bin/time -v` reports 1,044,316 KiB
and runner Linux `ru_maxrss` 1,069,379,584 B for the failed process.
These failure-resource observations are not warm throughput telemetry.

## Preserved scientific boundaries and evidence

The unchanged workload contract is 64 true unique points, Days 2–8 and 9–15,
896 point-days, all 760 native leaves/masks/dtypes and the original tolerances
and gates. Candidate numerical/permutation/replacement/reset/nonfinite gates
were not reached. The reused scalar's prior three finite cross-GPU failing
points/four comparisons and sealed-proof-only reset limitation remain.

Existing GPU B64 comparator job 15579869 retains core median
5.632416466251016 s (159.07914575720022 point-days/s) and wrapper median
6.222332641482353 s (143.99744462818447 point-days/s). With no new CPU samples,
no matching-boundary ratio or warm cost comparison can be computed. GPU whole
parent cost spans other work and must not be divided by 896. B1024 remains a
different-workload reference. See `GPU_COMPARATOR_EVIDENCE.md`.

Evidence is locally under this directory and remotely at `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014`:

- `preflight/15853600/production-preflight.json` and original preflight logs/argv.
- `paid-run/orchestration.json`, `paid-run/batch.log`, `paid-run/batch/report.json`
  mirror remote `15853683/`; originals retain stderr and complete failure details.
- `identity-and-child-receipts.json` is a derived compact extraction with source hash.
- `terminal/submission.*.raw`, `terminal/slurm-parent.log`, parent/step sacct rows,
  exact parent accounting, clean-checkout and collector execution receipts.
- `collector-result.json`, SHA `d7faeb702d568c897aefdc0f3ca2b560684f326edee2e90a270fc681cd7db0ca`.
- `operations/` preserves exact SSH/SCP argv, scripts, raw stdout/stderr and receipts.
- `runtime-evidence-manifest.json` maps local compact evidence hashes to remote originals.

The only original `sbatch` call returned `Submitted batch job 15853683`;
the one-attempt marker and raw receipt remain. All execution source/pins,
old data/checkouts and completed scalar reference outputs remain unchanged.
