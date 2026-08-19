# Mechanistic Output v2 Attainability Result

Date: 2026-08-18.

Status: accepted. The zero-training oracle proves that the mechanistic-output
v2 learned coordinates can attain the existing v6 Teacher boundary. The
remaining one-point training floor is therefore a neural mapping or
optimization problem, not an output-coordinate representation floor.

## Execution Boundary

The audit uses all 364 valid days from the frozen train/train point-year
`001.0-071.0`, 1961. It executes no neural body, has zero network parameters,
performs zero optimizer updates, and retains the accepted output transforms,
conservative update, post-update mechanistic reconstruction, masks, complete
loss, and hard state/budget checks.

It does not regenerate Teacher data, read historical v5 rows, reconstruct 48
half-hour forcing records, execute a 48-step state scan, read sealed test
data, or retain a checkpoint.

The accepted full-year report is:

`runtime/outputs/training/mechanistic-output-v2-attainability-source-conditioned-gpu0/report.json`

SHA256:

`292e8a037f1e627ed9a96465f0998988d2cb13611d83bdf945c57606674a0897`

The bounded four-day thermal source-alignment report is:

`runtime/outputs/training/mechanistic-output-v2-thermal-source-alignment-4day-gpu0/report.json`

SHA256:

`ffe388fb1df8dd322c24b73942cb921d85d0d0608c3c64aae3b38ff5ce55abce`

Both ran from the isolated diagnostic worktree based at commit
`806cffd1293f65ecf19931de5c7d98261987fad1` on free `gln01` GPU 0.

## Result

The full-year audit passes every operative gate:

- mean complete loss: `7.3612629e-11`;
- maximum learned-field normalized MSE: `2.0611537e-9`, below `1e-8`;
- maximum derived exact-field normalized MSE: `9.8732658e-22`, below `1e-8`;
- maximum DOC fixed-point residual: `0`;
- canopy-status mismatches: `0`;
- dynamic defined-status mismatches: `0`;
- prediction/state nonfinite counts: `0`;
- water/carbon budget failures: `0`;
- negative water/carbon inventory counts: `0`.

All seven expert losses are near numerical zero. The largest expert normalized
MSE is `5.1528841e-10` for canopy/GPP; litter/POC/DOC is
`1.2516482e-23` and surface energy is `1.9500859e-33`.

## Thermal Source Alignment

The first audit version incorrectly required every Teacher-derived exact
field to satisfy a fixed physical comparison of
`1e-10 + 1e-12 * abs(target)`. The accepted thermal panel proves exact
reconstruction when supplied the true v6 primitives, so the runtime sources
were compared directly before changing that gate.

All 22 dynamic, forcing, static, and interval inputs passed source physical
tolerance. Six sources were not bitwise identical because they pass through
finite inverse transforms or conservative arithmetic:

- `mc` and `mcl`: maximum absolute error `2.6732783e-13`;
- `snowrho`: `7.1054274e-15`;
- `snow`, `snow_nobio`, and `snowdz`: at or below `1e-30` scale.

`ptn`, `shum_ngrnd_permalong`, `refSOC`, vegetation inputs, surface
temperatures, snow temperature, pressure, and all static inputs were bitwise
identical on the panel. No source exceeded physical tolerance. The thermal
kernel can amplify the `mc/mcl` roundoff into approximately `1e-7` soil heat
capacity error and `1e-6` `pcapa_en` error while leaving normalized output
MSE near `1e-22`.

Derived fields are exact conditional on their materialized sources. Their
Teacher-coordinate accuracy is therefore gated in the same normalized output
space as other boundary fields; the fixed physical comparison remains a
reported sensitivity diagnostic, not an independent rejection gate. This
does not hide a model error: the source-alignment gate remains physical and
the normalized derived-field result is fourteen orders of magnitude below
its `1e-8` limit.

## Decision And Next Gate

Output-coordinate representation repair is closed. Retain hydrology/canopy
v2, the mechanistic exact reconstruction, and the current Attention backbone
for the next diagnostic.

Before another complete or paid training run, freeze and execute two fresh,
zero-retention one-expert memorization diagnostics on the same 364 train/train
days:

1. litter/POC/DOC only;
2. surface energy only.

Each diagnostic must isolate the selected expert's objective while preserving
the complete forward boundary and state/budget hard checks. It must report
field-local learning curves and gradient/update coverage. The purpose is to
distinguish failure of the shared training objective from failure of the
existing neural mapping. It is not an architecture comparison and does not
promote a checkpoint.

Do not open 64-point-year training, 7/30-day rollout, 365-day rollout, Gate
D2, sealed test, or product acceptance from this result alone. Any network
architecture change remains subject to prior discussion with the user.
