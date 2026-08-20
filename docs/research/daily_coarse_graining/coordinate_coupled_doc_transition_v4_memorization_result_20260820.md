# Coordinate-Coupled DOC Transition v4 Memorization Result

Date: 2026-08-20.

Status: completed normally and rejected by the frozen field gate. No
checkpoint is retained. The next operation is a bounded target/loss-geometry
audit, not another training run.

## Execution

Approved paid job `14627588` ran
`coordinate_coupled_doc_transition_attention_v4` at commit
`b903dac735c990f82a6f108d4e6ddd831d638bc9`. It used one V100, four CPUs, the
frozen 364 valid train/train days for point `001.0-071.0` in 1961, logical
batch size 8, exact size-2 gradient microbatches, and the unchanged unclipped
AdamW schedule.

The job completed 100 epochs and 4,600 updates in `05:30:23` with Slurm state
`COMPLETED` and exit code `0:0`. It used only the accepted v6 release, did not
regenerate Teacher data, did not use v5 data, reconstructed no 48-record
forcing, scanned no 48 state steps, and wrote no checkpoint. JAX reported
peak bytes in use `23,646,699,264` (`22.02 GiB`).

The report is at
`runtime/outputs/training/coordinate-coupled-doc-transition-litter-100-epoch-b903dac/report.json`
on Explore1000, with SHA256
`90cd1b3ef87e54d5d773a3873b45a0dca0ee4f65f31beff55c3aebe59aa00a9c`.

## Frozen Gate Result

Selected litter/POC/DOC loss fell from `0.18274861` to `0.001580235`, a
`99.1353%` reduction. This passes the aggregate `99%` threshold for the first
time. All observed gradients were finite, selected coordinate blocks and
field heads had nonzero gradients and updates at the first and final probes,
defined-value counts were unchanged, and every state, budget, finite-value,
and negative-inventory hard count was zero.

The candidate nevertheless passes only `24/28` initially dominant fields.
The frozen requirement is at least `95%` reduction or final loss at most
`1e-4` for every dominant field. Four fields fail:

| Field | Initial loss | Final loss | Reduction |
| --- | ---: | ---: | ---: |
| DOC outgoing fraction | `0.00119122` | `0.01265478` | `-962.3373%` |
| DOC free-to-adsorbed | `0.04039929` | `0.00599388` | `85.1634%` |
| DOC incoming amount | `0.05098627` | `0.00481887` | `90.5487%` |
| DOC advective interface | `0.04962206` | `0.00366499` | `92.6142%` |

The formal status is therefore `completed_failed_gate`, and no downstream
gate opens.

## Mechanism Interpretation

The recorded v3 mechanism control reduced aggregate loss `98.5804%` and
passed `25/28` dominant fields. V4 improves aggregate reduction to `99.1353%`
and changes DOC incoming reduction from `-0.3085%` to `90.5487%`. It also
improves free-to-adsorbed reduction from `83.2158%` to `85.1634%`. This is
strong evidence that coordinate-aligned upstream DOC tokens provide useful
information and should not be discarded.

The joint signed transition decoder is not accepted. It changes DOC outgoing
from a v3 pass at `99.5441%` reduction to a loss `10.62` times its initial
value. Advective reduction also slips from `93.5753%` to `92.6142%`. A
physically valid one-scalar endpoint identity is therefore not automatically
a well-conditioned training representation under the two separately
normalized endpoint losses. V4 exchanged the old incoming failure for a
larger outgoing failure and reduced the dominant-field pass count.

## Active Gate

Before designing v5 or spending more GPU time, audit the frozen real-v6 DOC
endpoint target and loss geometry without training:

1. invert the joint materializer for every admitted target and quantify raw
   target magnitude, sign balance, boundary distance, and conditioning;
2. separate incoming-active, outgoing-active, and zero-transition coordinates
   and measure their contribution under the current field-balanced loss;
3. test exact target reconstruction and local gradients for the current
   signed scalar and bounded alternatives such as explicit sign/status plus
   magnitude or coordinate-coupled branch-specific endpoint heads;
4. freeze a revised mechanism and matched absolute gates before any new
   network implementation or training.

Do not repeat v4 training, open multi-point training or rollout, read sealed
test data, advance to Gate D2, or retain a checkpoint from this run.
