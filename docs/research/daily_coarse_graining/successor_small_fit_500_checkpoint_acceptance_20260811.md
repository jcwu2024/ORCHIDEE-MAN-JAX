# Successor 500-Update Fit and Checkpoint Acceptance

Date: 2026-08-11.

Status: accepted. Both matched arms completed the frozen 500-update fit and
the atomic checkpoint set passed the formal reader. The next active operation
is the already frozen 7/30-day rollout package.

## Bound Evidence

- Training and recovery identity: `9023de2a3fbb3ddb49f4476d9bdac4b003502d1b`.
- Completed-recovery finalizer: `d834d5df7056b17e8499926a0b9808f5634c0874`.
- Final fit report SHA256:
  `ea4b66cd2d09d20f5ac4754de461382d560e4feedca5d52e211bebe77a4a259a`.
- Checkpoint manifest SHA256:
  `f8eed186a90c2f5ecad39c6f9114b343fb29ab468266fed39dfe70e353275e40`.
- Attention checkpoint SHA256:
  `b299124877dd58fdc00507dfb6aeae210e5ebd8780dc6de112f01d28a47287f1`.
- Graph-control checkpoint SHA256:
  `158a097060745980b5b2422e1573a55ebc4c1ed83b7c624ee170a3941de89951`.

The accepted server product is under
`runtime/outputs/training/successor-small-fit-500u-9023de2-finalized-d834d5d`.
Each arm checkpoint is 209,404,674 bytes and contains 8,722,689 parameters,
17,445,379 optimizer elements, and exactly 500 completed updates.

## Recovery and Finalization

Paid job `14520379` resumed the accepted update-155 recovery and completed
updates 156-500 in 31 minutes with four CPUs and one V100. Every update was
applied, all recorded gradients were finite, and all five hard-gate counts
were zero. The job exited one only because the runner required every resumed
JIT cache to contain exactly one executable. The resumed host-to-device array
transition legitimately created a second specialization at update 157; no
new executable appeared in updates 158-500.

Commit `d834d5d` replaced that equality check with a bounded process-local
specialization policy. It also permits a cross-commit recovery only for
diagnosis or for finalization after both arms already reached 500 updates.
The free-`gln01` finalizer executed zero training updates, recomputed the
zero-rate final metrics, retained the `9023de2` training identity, and
atomically published both arms. A separate `test01` reader rehashed both
checkpoint files and verified their identities, parameter and optimizer tree
schemas, update counts, and finite numeric contents.

Teacher v6 data were not regenerated, historical v5 data were not used, the
sealed test split was not read, and neither 48 half-hour forcing records nor a
48-step state scan were reconstructed.

## Fit Result

Both arms passed the frozen minimum one-day and weighted long-horizon loss
reduction gates:

| Arm | One-day reduction | Weighted long-horizon reduction |
| --- | ---: | ---: |
| Process attention | 0.9929047 | 0.9858693 |
| Graph MLP control | 0.9244996 | 0.9976531 |

The fit gate does not select an arm. In particular, the attention arm has the
lower final one-day objective, while the control has the lower final weighted
long-horizon objective. Biomass, productivity, signed bias, and candidate to
control comparisons remain the responsibility of the frozen 7/30-day rollout
screen.

## Next Gate

Do not rerun the 500-update fit. Prepare the deterministic three-window panel
on a free CPU test node, then run the free `gln01` 7/30-day smoke using this
exact report and checkpoint manifest. The formal 96-window screen, 365-day
rollout, long-held points, Gate D2, and product acceptance remain closed.
