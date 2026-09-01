# Coupled-GPP Objective Calibration Result

Date: 2026-09-01

Status: accepted zero-update calibration; optimizer proposal remains untested

Explore1000 job `14899511` completed at commit `5582348` in `01:38:16`.
The report SHA256 is
`4f209b773d020964e485dfdfb1bfbc247be6106dc39540d4fde5be40800b5138`.
Job `14899347` was an engineering failure before objective evaluation and
produced no report; its JIT dictionary-order defect is repaired and covered by
a regression test.

All four source gradients are finite and nonzero over all 15,960,195 trainable
parameters and all 190 leaves for all eight train-only artifacts. All 16
reachable state fields are active. Retained parameters and the structured-GPP
checkpoint are unchanged. Prediction/state finite-value, unexpected-mask,
discrete-state, budget, and negative-stock hard failures are zero. The 849
declared-dynamic `rveget` mask differences are the established report-only
diagnostic.

The frozen coefficients are:

- direct GPP: `1.0`;
- reachable one-day state: `0.6524275555272667`;
- 7-day component-resolved science: `0.4072260558556481`;
- 30-day component-resolved science: `0.28249951035415655`.

The p90 combined-gradient clip candidate is `12.808193184117403`. Combined
gradients are finite and nonzero, but this is not evidence that an optimizer
step is safe. The within-artifact combined direction is opposed to 7-day
science in three of eight artifacts and to 30-day science in four of eight.
It is opposed to the reachable-state source in one artifact and never opposed
to direct GPP.

The next gate therefore uses all eight artifacts as one macro effective batch.
It averages gradients within each source across artifacts, applies the frozen
four-source scalarization, and evaluates one fresh-AdamW proposal against all
four macro losses. The proposal is discarded and no checkpoint is written.
