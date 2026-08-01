# OK_LEAK Auxiliary Capture and Persisted Replay

Date: 2026-08-01

## Decision

The bounded train-only `OK_LEAK` driver asset passes both its capture-interface
gate and its persisted exact-scan replay gate.

This admits the asset for the remaining conservative micro-gates. It does not
authorize paid neural training by itself and does not claim full outer
seven-day Teacher replay equivalence.

## Frozen Identity

```text
capture plan:
manifests/coarse_graining/ok_leak_auxiliary_capture_96day_v1.json

capture plan SHA256:
8ebe5345ec53a449f41acd46f9a54e692ad71e7ed3ba768e7c1af83863b922c7

capture root:
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/ok-leak-auxiliary-capture-96day-v1-teacher-exact-7ef3db3

evaluation commit:
332ab26d23d714cb02f388bc9e02d0593e4733c6

replay root:
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/ok-leak-persisted-replay-96day-332ab26
```

The plan contains 96 unique train-only days: 16 selected days for each of six
training landpoints. The sealed test split was not used.

## Capture Result

- status: `complete`
- completed target-day captures: `96/96`
- capture-interface failures: `0`
- complete hash-bound reader: passed for all 96 records
- unrelated full next-state diagnostics: `91/96` passed

Target-day capture acceptance is deliberately independent of later days in
the enclosing seven-day diagnostic block. This preserves the frozen `1e-12`
process tolerance without rejecting a valid target capture because of an
unrelated later outer-state difference.

## Persisted Replay Result

Slurm job `14446150` replayed every stored 13-driver sequence through the
current exact 48-step `_paper_compiled_ok_leak_fold`.

- Slurm state: `COMPLETED`
- elapsed time: `00:29:34`
- exit code: `0:0`
- node: `ibc11b10n27`
- peak batch RSS: `9054672K` (about 8.64 GiB)
- completed records: `96/96`
- failed records: `0`
- records within the frozen `1e-12` endpoint tolerance: `96/96`
- global maximum absolute error: `4.5401182813264995e-14`
- maximum-error record: `069.0-119.0`, 1961 Day 7
- fully bit-exact records: `6`

The number of bit-exact endpoints out of 14 is distributed as follows:

```text
4: 3 records
5: 3 records
8: 1 record
10: 1 record
11: 3 records
12: 2 records
13: 77 records
14: 6 records
```

The non-bit-exact values remain far below the declared float64 tolerance. No
tolerance was widened and no endpoint was skipped.

## Interpretation

The result proves that the persisted auxiliary arrays contain the complete
dynamic input needed to reproduce the source-backed `OK_LEAK` carbon endpoint
transition over the accepted cross-point, cross-year, and cross-season sample.
It also validates the restartable, atomic, hash-bound capture and replay
infrastructure.

It does not prove that reconstructing a single day outside the original outer
seven-day compiled block reproduces every unrelated Teacher field. Historical
Teacher shards were generated in seven-day compiled blocks, so the accepted
verification boundary is the isolated persisted-driver exact scan.

## Next Gate

Before a paid matched neural A/B, complete:

1. source-backed combined carbon-inventory conservation;
2. physically feasible driver perturbations and nonnegative stock endpoints;
3. real forward- and reverse-mode finite gradients through the 48-step scan;
4. bounded tiny-set fitting while `carbon_32l`, `DOC`, and `deepC_peat` remain
   exact-scan-owned and do not regress relative to the parent.

