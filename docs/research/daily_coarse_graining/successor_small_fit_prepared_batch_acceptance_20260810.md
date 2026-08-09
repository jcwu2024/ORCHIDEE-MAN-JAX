# Successor Small-Fit Prepared-Batch Acceptance

Date: 2026-08-10.

Status: CPU prepared-batch boundary and same-process GPU hot reuse accepted;
the former 20-hour paid envelope is rejected, and cross-process persistent
JAX cache reuse is not accepted.

## Bound Evidence

- Accepted implementation commit: `1c63494649207eecc365669fd3d9fcd7b85f389c`.
- Runner contract SHA256:
  `8317fc74b6ba921b0ed60e774743884ddc81b233c75ce1823f3246211d1640de`.
- Two-update schedule SHA256:
  `2c6ff41808b97b833850ecee667ddeb2e0561ad9d64dba7ce18835a7ce49fadc`.
- Prepared manifest:
  `runtime/outputs/training/successor-small-fit-prepared-2u-1c63494/prepared_manifest.json`.
- Prepared manifest SHA256:
  `2076ab3220f58be61d26bc1de0e0c10d2b16eb99c85318f16b02fda2bb9753c6`.
- Fixed-seed cold GPU report:
  `runtime/outputs/training/successor-small-fit-hashseed0-cold-2u-1c63494/runner_report.json`.
- Cold GPU report SHA256:
  `481581d1648699e140fa12c1b6a8e342f154ecb2216473f01a6916bd95c31f60`.

No paid Slurm job was submitted. The CPU probes ran on `test01`; GPU probes
ran on free `gln01`. Teacher data were not regenerated, historical v5 was not
read, and no checkpoint was written.

## Prepared Boundary

`prepare_successor_small_fit_batches.py` materializes each deterministic
schedule entry and its retained-tail dynamic arrays on a CPU host. Every
artifact is bound to the source HEAD, runner contract, schedule entry, trace
signature and SHA256. The GPU runner requires this manifest and has no fallback
to per-update host assembly. It builds one live static retained owner, compares
its trace signature with the artifacts, and then supplies artifact `day_inputs`
as dynamic JIT inputs.

The two CPU artifacts are byte-identical before and after fixing
`PYTHONHASHSEED=0`. They total 66,868,764 bytes. The accepted run prepared the
entries in 71.73 and 63.55 seconds after runtime loading, with peak RSS
5,017,320 KiB. Their shared trace signature is
`13ed0a2...cc51d8b`.

## GPU Result

The fixed-seed cold run passed both matched arms and both schedule entries.
All gradients were finite. Budget residual, negative stock, nonfinite defined
prediction, mask mismatch and discrete-state mismatch counts were zero.
Replay, attention and control added no executable on the switched second
entry.

| Stage | Cold/first entry | Switched-point hot entry |
| --- | ---: | ---: |
| Prepared artifact load | 0.179 s | 0.235 s |
| Live retained owner build | 29.97 s | 0 s |
| Teacher replay | 31.99 s | 1.311 s |
| Attention update | 575.43 s | 1.138 s |
| Graph-control update | 575.89 s | 0.879 s |

Total artifact load was 0.414 seconds. This removes the earlier 86.57 seconds
of repeated host batch/retained preparation from every hot update. The new
hot marginal is about 3.33 seconds, of which about 0.24 seconds is artifact
loading.

The complete cold process still took 1,908.38 seconds. It included 408.08
seconds of coefficient preflight, about 19.19 minutes for the two first
training compilations, and about 210.40 seconds for final-evaluation
compilation/execution.

## Rejected Cache Assumption

The first prepared run showed different `jit_train_step` persistent-cache keys
across fresh processes. Fixing `PYTHONHASHSEED=0` did not resolve the issue: a
fresh process using the completed fixed-seed cache again entered a slow
`jit_train_step` compilation and was terminated by its remote 10-minute hard
limit. The GPU returned to zero allocation and no report was admitted.

Therefore a free-`gln01` precompile cannot currently be claimed to remove
paid-process compilation. Do not project a paid envelope by assuming a
cross-process train-step cache hit.

## Active Gate

The old 20-hour, CNY 49.60 envelope and its Slurm script are withdrawn. No
paid envelope is active. Before producing all 500 CPU artifacts or requesting
`gnall`, remove avoidable same-process compilation from coefficient preflight
and final evaluation, then run a new bounded cold/hot timing probe. Any new
paid request must include the measured cold compilation once, the measured hot
500-update cost, artifact loading, a finite wall-time limit and a fresh
worst-case charge approval.

The 7/30-day screen, 365-day rollout, long-held points, Gate D2 and product
acceptance remain closed.
