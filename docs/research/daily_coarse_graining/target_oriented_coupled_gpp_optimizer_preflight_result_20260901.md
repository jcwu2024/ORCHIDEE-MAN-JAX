# Coupled-GPP Optimizer Preflight Result

Date: 2026-09-01

Status: rejected finite optimizer step; common first-order direction remains

Explore1000 job `14902356` ran at commit `b54c502` for `00:26:32`. The JSON
report SHA256 is
`dd43a960fc077284fc5d5925ed55df4141c20e55d04a9489a820d0db58d146ab`.
The runner completed all eight source-gradient records, created one fresh
AdamW proposal, evaluated all eight post-proposal records, wrote no checkpoint
or recovery, and returned exit code one because the frozen scientific gate
failed. This was not an engineering or scheduler failure.

All four averaged gradients, the combined gradient, optimizer state, and all
15,960,195 proposed parameter values are finite. Both trainable families and
all 190 leaves change. Every source gradient has positive cosine with the
realized negative parameter step: direct GPP `0.222712`, reachable state
`0.219260`, 7-day science `0.0338848`, and 30-day science `0.0184151`. All
physical hard gates pass, all 93,504,164 fixed parameters retain the same JAX
arrays, and the structured checkpoint is unchanged.

The finite `4e-6` step is not admitted. Reachable-state loss falls `30.0427%`,
but direct GPP rises `11.9170%`, 7-day science rises `0.7175%`, and 30-day
science rises `0.0550%`. The weighted total falls `1.1798%`; this does not
override the frozen requirement that every source loss decrease. The raw
gradient norm is `6.41398`, below the `12.8082` clip threshold, so clipping is
not the cause.

The evidence classifies the failure as finite-step curvature after a valid
common first-order direction. A local quadratic estimate puts the strictest
zero crossing near learning rate `2.46e-6`. The next bounded diagnostic is one
independent fresh-AdamW proposal at exactly half the rejected rate, `2e-6`,
using the same gradients, artifacts, coefficients, and gates. It is not a
learning-rate search, writes no checkpoint, and authorizes no training.
