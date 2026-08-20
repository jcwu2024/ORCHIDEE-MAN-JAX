# Coordinate-Coupled DOC Transition v4 GPU Smoke

Date: 2026-08-20.

Status: accepted. The bounded 100-epoch isolated memorization diagnostic is
prepared but requires explicit paid-job approval before submission.

## Execution

The no-checkpoint smoke ran on free node `gln01`, GPU 0, at commit
`5df121a560f30cf18c1326fcccad1fc6be0e4918`. It used the frozen 364 valid
train/train days for point `001.0-071.0`, year 1961, and the accepted logical
batch size 8 with exact size-2 gradient microbatches. The executed flags were:

```text
MECHANISTIC_OUTPUT=1
HIERARCHICAL_COORDINATE_ATTENTION=0
COORDINATE_COUPLED_DOC_TRANSITION=1
ISOLATED_EXPERT=litter_poc_doc
TRAINING_SMOKE=1
```

The run used the v6 release without regeneration, used no v5 data, did not
reconstruct 48 half-hour forcing records, did not scan 48 state steps, and
wrote no checkpoint.

The formal report is at
`runtime/outputs/training/coordinate-coupled-doc-transition-litter-one-epoch-smoke-5df121a/report.json`
on Explore1000, with SHA256
`ab8781f9362f3c3e0bf8bd72e854cab6b824441c62f2b4d342a64fe9d2df7605`.

## Result

The report schema is
`coordinate_coupled_doc_transition_v4_one_epoch_smoke_v1`. Execution and the
smoke gate both report `passed`:

- all 46 updates completed with finite gradients and values;
- the selected coordinate blocks and field heads had nonzero gradients and
  nonzero updates at both the first and final probes;
- defined-value counts remained identical;
- prediction/state nonfinite, water/carbon budget, and negative-inventory hard
  counts were all zero;
- no checkpoint was written.

The selected litter/POC/DOC loss fell `48.5539%` after one epoch. Gradient
norms were finite without clipping: minimum `0.1583`, median `0.5721`, and
maximum `2.7564`.

This is an execution and trainability result, not an absolute capability
result. DOC incoming amount improved only `2.6672%`, free-to-adsorbed transfer
improved `3.4735%`, and advective transfer improved `18.4918%` after one
epoch. DOC outgoing fraction became worse from its small initial loss. Those
field values neither reject nor promote v4 at one epoch; the frozen 100-epoch
gate must decide whether the joint representation converges.

## Resource Evidence

The run completed in `2247.59` seconds (`37:27.59`) including cold
compilation. The epoch history reports `833.84` seconds for its 46 updates and
evaluation after compilation. JAX reported peak bytes in use
`23,555,771,392` (`21.94 GiB`); `nvidia-smi` observed approximately
`25.5 GiB` including allocator/runtime overhead. The candidate therefore fits
one 32-GiB V100 under the accepted microbatch boundary.

The historical matched v3 100-epoch job used the same one-V100/four-CPU
boundary and required `05:34:10` of a six-hour allocation. V4 must therefore
use an eight-hour ceiling rather than risk another near-limit exit. The
prepared script is
`scripts/hpc/slurm_coordinate_coupled_doc_transition_v4_one_point.sh`.

## Active Gate

The next operation is one fresh 100-epoch, 4,600-update isolated v4 diagnostic
on `gnall`. Promotion still requires at least `99%` aggregate reduction and,
for every initially dominant field, at least `95%` reduction or final loss at
most `1e-4`. It writes no checkpoint. Multi-point training, rollout, sealed
test, Gate D2, and product work remain closed regardless of smoke success.
