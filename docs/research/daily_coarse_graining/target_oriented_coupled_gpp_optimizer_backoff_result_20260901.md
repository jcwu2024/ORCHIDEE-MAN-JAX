# Coupled-GPP Optimizer Backoff Result

Date: 2026-09-01

Status: accepted discarded half-rate proposal; training contract remains pending

Explore1000 job `14902878` completed at commit `510a059` in `00:23:34` with
exit code zero. The report SHA256 is
`6abb472de28e7eee1683fac2bcd52d265f9d0ce1bdafd3c0cd43883978f751c5`.
The run evaluated one independent fresh-AdamW proposal at learning rate
`2e-6`, then discarded the parameters and optimizer without writing a
checkpoint or recovery.

Every frozen gate passes. All four averaged gradients, the combined gradient,
the optimizer, and all 15,960,195 trainable parameter values are finite. Both
trainable families and all 190 leaves change. Direct GPP, reachable state,
7-day science, and 30-day science gradients have positive cosine with the
realized negative parameter step: `0.222735`, `0.219283`, `0.0338874`, and
`0.0184158`.

The macro losses change as follows:

- direct GPP: `-4.08665%`;
- reachable one-day state: `-16.3007%`;
- 7-day component-resolved science: `-0.003126%`;
- 30-day component-resolved science: `-0.039625%`;
- weighted total: `-4.37440%`.

The raw gradient norm is `6.41398`, the clip scale is exactly `1.0`, and the
parameter-step norm is `0.00781028`. All physical hard counts are zero, all
93,504,164 fixed parameters retain the same JAX arrays, and the structured-GPP
checkpoint is unchanged.

This accepts optimizer realization at one macro update. It is not evidence of
multi-update training stability or science promotion. The next gate is to
freeze target-blind training and train-only monitor schedules, a bounded
training budget, recovery/checkpoint ownership, and stop rules before any
training runner or paid training submission is authorized.
