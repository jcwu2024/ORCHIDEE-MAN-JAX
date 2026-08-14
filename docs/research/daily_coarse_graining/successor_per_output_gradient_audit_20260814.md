# Successor Per-Output Gradient Audit

Date: 2026-08-14

## Question

The 100-update equal-sample A/B improved Attention mixed sampling by 4.35%
relative to Attention single sampling, while validation attribution showed an
opposing maintenance-respiration versus NPP/biomass pattern. This audit asks
one narrower train-only question:

> Does four-point mixed sampling improve the alignment between the Attention
> maintenance-respiration gradient and the NPP, AGB, BGB, and total-live-
> biomass gradients at the accepted update-1000 state?

The predeclared stable-conflict rule is median cosine at most `-0.1` with a
negative cosine in at least `3/4` groups. Mixed mitigation requires the median
cosine to improve by at least `0.1`. At least two confirmed and two mitigated
pairs are required to support the proposed mechanism.

## Scope

The accepted run:

- reads only existing v6 train/train prepared artifacts;
- loads the accepted update-1000 recovery without changing it;
- compares four single-point groups with the same four equal-sample mixed
  groups used by the batch-diversity probe;
- uses the frozen train/train scales, long-horizon coefficients, and 1/7/30-
  day definitions;
- audits Attention only and exactly the five outputs needed by the stated
  mechanism: NPP, maintenance respiration, AGB, BGB, and total live biomass;
- performs zero optimizer updates and writes no checkpoint or recovery;
- does not evaluate validation values or access the sealed test split.

An initial broader implementation attempted to retain eight complete
8,722,689-parameter gradient trees on the V100 simultaneously. It failed
before producing a report when XLA requested one `25.35 GiB` allocation. The
accepted implementation streams one gradient tree at a time to host memory
and computes the exact Gram matrix there. It also removes GPP, aggregate-loss,
one-day-anchor, and control-arm gradients because they are not needed for the
registered Attention conflict-mitigation question. This narrower result must
not be cited as a shared-arm loss-calibration audit.

## Accepted Evidence

- Git commit: `4f35e6adfaa4bba9bb7a7a2bd00675d690132441`
- Server report:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/successor-per-output-gradient-audit-4f35e6a/report.json`
- Report SHA256:
  `a476b332624a83515555e5532229ca51b265c6ff7eab1c349156c1a3d1410b49`
- Elapsed time: `779.454 s` on free `gln01`
- Peak JAX device bytes in use: `1,110,032,896`
- Training updates: `0`
- Hard-gate and nonfinite-gradient counts: all zero

The implementation and launcher pass Ruff, Python compilation, Bash syntax,
diff checks, and `65` focused unit tests.

## Conflict Result

| Maintenance versus | Single median | Single negative groups | Mixed median | Mixed minus single | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| NPP | -0.6288 | 3/4 | -0.9544 | -0.3256 | conflict confirmed; worsened |
| AGB | -0.8106 | 3/4 | -0.8709 | -0.0603 | conflict confirmed; worsened |
| BGB | +0.5485 | 1/4 | +0.3674 | -0.1811 | no stable single conflict |
| total live biomass | -0.7846 | 4/4 | -0.8739 | -0.0893 | conflict confirmed; worsened |

Three pairs satisfy the stable single-batch conflict rule. Zero pairs satisfy
the mixed-mitigation rule. The formal classification is
`attention_mixed_conflict_mitigation_not_supported`.

Mixed sampling's observed Attention validation improvement remains real, but
this audit rejects the explanation that it worked by reconciling these output
gradients. Sampling coverage, gradient variance, or a different interaction
may still explain the result.

## Gradient-Scale Result

The frozen loss also leaves stock gradients far smaller than productivity and
respiration gradients:

| Output | Single median norm | Mixed median norm |
| --- | ---: | ---: |
| NPP | 1.188e-1 | 1.543e-1 |
| maintenance respiration | 8.113e-3 | 3.487e-2 |
| AGB | 3.923e-6 | 2.631e-6 |
| BGB | 2.113e-7 | 2.270e-7 |
| total live biomass | 3.985e-6 | 2.442e-6 |

This is a four-to-six-order parameter-gradient imbalance after the existing
normalization and coefficients. It explains why terminal-stock terms can be
present in the scalar objective yet exert little influence on an update. A
large static multiplier would increase their magnitude but cannot change the
negative gradient cosines.

## Decision

Do not continue mixed sampling on the claim that it mitigates output conflict,
and do not start another unchanged training continuation. The evidence does
support a loss-objective design review because it confirms both stable
maintenance conflict and severe biomass-gradient suppression.

The preferred next proposal should evaluate a train-only normalized biomass
increment/tendency objective, retain the one-day named-process anchor, and
treat long-horizon maintenance as an explicit guard or constrained secondary
objective. This must not reopen the rejected `process_increment_v2` unchanged:
that experiment supervised all 3,854 anonymous state increments in an older
direct-state model. A materially new proposal must operate only on named
AGB/BGB quantities derived after the conservative daily operator and explain
why that changed ownership boundary addresses the old failure.

The proposal must define a matched current-loss control on the same Attention
architecture, retain a graph-control arm if making an architecture claim, and
freeze zero-update gradient and bounded short-fit gates before training.
Static validation-tuned weights and expensive per-update multi-gradient
surgery are not authorized by this result. The current design proposal is
[`successor_loss_objective_design_review_v1.md`](successor_loss_objective_design_review_v1.md).
