# Fresh litter joint-method admission

Free gln01 GPU0,1 integration COMPLETED0 at immutable7b9e0ae after2560.00s
(42.67min). Both full-network joint updates completed. The separate diagnostic
starts from seed20260911 and fresh AdamW; no checkpoint is inherited. Its
weights will not initialize the proposed paid training.

Evidence: `outputs/fresh-litter-v1-integration.json`,
`outputs/fresh-litter-v1-runtime.json`, `outputs/fresh-litter-v1-metadata.json`.
Remote report/log: `runtime/outputs/diagnostics/fresh-litter-v1-integration.{json,log}`.
Wrapper85836/Python85878 have exited. No new paid submission.

## What passed

- Two changing-data updates, each daily64/micro8 + oneT7 + oneT30.
- Daily and recurrent hard counters all zero, including litter validity,
  masks, discrete state, finite predictions/state and budget bookkeeping.
- Nonzero finite full mixed gradients; norms2499.27 and5352.16.
- AdamW advances through step2 and peers synchronize after each update.
- Update0/update2 fixed/excluded TRAIN and16-window development monitors
  have complete metric populations and pass physical checks.
- Saved checkpoint forward replay equals the live forward result.

These are bounded execution/physical checks, not final gradient coverage or
scientific acceptance. Existing unchanged native-AD evidence is reused; the
explicit four-term scalar/gradient composition is also tested locally. The
deployed JAX0.4.38 synthetic interface check passes in61.64s. Local relevant
checks total75 passing cases across runner/data/dual/litter, fresh composition,
panel selection and evaluator tests; static checks pass.

## Loss and learning interpretation

The first fixed TRAIN daily loss is312.6712. Its terms contribute approximately
base0.3981, canopy312.0335, litterpart0.0250 and litter0.2146. Thus the new
litter coefficient is actually used once. Large initial canopy loss dominates
the scalar; scalar shares are not gradient shares.

After two updates the fixed daily loss is311.83 and excluded TRAIN daily
loss582.24->581.09. Fixed litter above/below physical change RMSE moves
94.583->94.413 and443.061->442.175gC/m2/day. These remain large errors and
tiny changes. The two updates use different data, so their mixed losses
501.72 and876.38 are not a same-population learning curve.

Both gradients are clipped strongly (scales0.0004001 and0.0001868). This
does not establish undertraining, a bad clip threshold, or learning failure
by itself; two updates cannot settle the learning strategy. No accuracy
claim or coefficient adjustment follows from this integration.

## Resource evidence

Cold compilation: daily988.40s, T7262.38s, T30273.84s, total25.41min.
The initial bootstrap/monitor completes at671.6s; final monitors and persisted
replay account for additional overhead. First update37.00s, second7.52s.
Second-update daily/recurrent branches take7.033/6.720s, merge/check/AdamW
0.347s and synchronization0.069s. One warm sample is not sustained-throughput
proof. Keep the conservative5h ceiling for the proposed1000-update segment.

Measured device allocator peaks: GPU0~16.07GiB, GPU1~2.65GiB. Both reserve
~26.98GiB in the JAX pool; reservation is not actual peak usage. Compiled
daily gradient estimate14.42GiB, recurrent estimates~2.23GiB each. This is
no-OOM evidence for this exact batch/execution policy, not arbitrary batch
capacity or future longer-horizon certification.

## Next action

Finish the already-running48-window validation preparation on free test02;
its metadata-only selection succeeded at96fa02d before any candidate forward
on those windows. Seed20260921, eight windows per temporal/spatial/joint x
T7/T30 cell, four documented consumed panels excluded (239 distinct point-years).
The sealed test is untouched. This is fresh held-out point-year validation,
not a claim that every historical exploratory site has never been consulted.

The next paid proposal is one fresh1000-update segment of the frozen5415-update
method:2V100/8CPUs/64GiB/5h, maximum24.80yuan, no automatic extension. Checkpoints
and optimizer follow the same declared recipe at any later resource segment.
Assess physical quantities and the broader validation after training, without
selecting the best-looking early checkpoint as final. Follow
`fresh_litter_joint_protocol_20260921.md` for exact method and command.
