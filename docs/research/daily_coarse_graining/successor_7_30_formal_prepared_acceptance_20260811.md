# Successor 7/30-day Formal Prepared-Panel Acceptance

Status: **accepted immutable CPU-prepared input; paid formal GPU screen awaits
explicit authorization**.

Date: 2026-08-11.

## Decision

The deterministic `formal_bounded_screen` panel has been materialized from the
accepted v6 release on the free `test01` CPU node at source commit
`753db81bd39edf55f17c7cbec069859f2f2b1cda`. The product contains all 96
predeclared 30-day windows: eight landpoint/year artifacts per temporal,
spatial, and joint slice, with four nonoverlapping starts in each artifact.

This accepts the exact host input to the paid screen. It does not run either
checkpoint, calculate validation metrics, classify the models, or authorize
the Slurm submission.

## Product Identity

Server directory:

```text
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/
successor-7-30-formal-prepared-753db81
```

- prepared-manifest SHA256:
  `6f4c4a300320113cd67b7c7601a04b394274de505f5d30c35c2d4509023d0074`;
- window-plan file SHA256:
  `71a1a1ac7747536b4f0ba9f750b163a59aac86bf7f2f5c3d9002b23f9f397965`;
- internal plan SHA256:
  `2476309696b3ba24f62b435f7c08e47192c58279242e754c12b0cd4654f722b6`;
- artifact count: `24`;
- represented window count: `96`;
- total artifact bytes: `212,275,046`;
- preparation elapsed time: `1175.747` seconds.

The two small JSON files are archived locally under
`outputs/diagnostics/successor-7-30-formal-prepared-753db81/` with matching
hashes.

## Independent Reload

A fresh `test01` Python process loaded the manifest with the formal reader,
rehash-verified and unpickled every artifact, and reproduced:

- manifest status `passed` and mode `formal_bounded_screen`;
- `8/8/8` temporal/spatial/joint artifacts;
- `96` total windows;
- one common retained-tail trace signature;
- exact declared total size;
- `teacher_data_regenerated=false`, `historical_v5_used=false`, and
  `sealed_test_used=false`.

The first foreground preparation attempt was terminated by its SSH session
after six artifacts and before manifest publication. Those incomplete files
were removed. The successful retry ran detached, rebuilt the complete product
from the same frozen plan, exited zero, and atomically published the manifest.
This was an orchestration issue, not a data or numerical failure.

## Next Gate

Submit the already frozen formal evaluator only after explicit approval of
the `gnall` resources, 45-minute wall-time ceiling, exact worktree and command,
and worst-case charge. The evaluator must bind the prepared-manifest and
window-plan hashes above, the accepted fit report and dual-checkpoint hashes,
and source commit `753db81`. No alternate window, model, loss, or threshold is
admitted.

The 365-day rollout, long-held points, Gate D2, sealed test, and product
acceptance remain closed until the formal screen passes.
