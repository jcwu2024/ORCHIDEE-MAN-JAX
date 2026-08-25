# Target-Oriented Update-251 Continuation Admission (2026-08-25)

## Status

The exact update-250 checkpoint handoff, one real update-251 optimizer step,
recovery write/reload, and matched bounded 7/30-day screen pass on free
`gln01`. Full free-CPU preparation of updates 251-1000 is now admitted. This
report does not authorize or promote paid training.

## Prepared Admission Batch

Free CPU node `test02` prepared only absolute update 251 from the accepted v6
train/train release:

- manifest:
  `runtime/outputs/training/target-oriented-prepared-1000-f69e322/chunks/251-251/prepared_manifest.json`;
- manifest SHA256:
  `7d67b26b549ca5e99a084eaca8e43612c96843e352fff75d8fc803ad3b0dbd9f`;
- artifact SHA256:
  `131fa2fb7add28a6584e3679baa1df9988739db0e0cc12c54b17dce185a3121f`;
- artifact size: 3,322,098 bytes;
- full 1000-update schedule SHA256:
  `ee1ff380a18e1ff3ea919c579455db5895fa95116f2f95d144d21a852da9e2a5`;
- runtime year: 1962;
- source commit: `f69e32292b1f07695cd597cd0512b7a8fdbdd3cc`.

The manifest covers exactly update 251 and declares no Teacher regeneration,
historical v5 use, or sealed-test read.

The first launcher invocation stopped before Python because CentOS Bash with
`set -u` rejects expansion of a declared but empty array. It performed no
preparation or GPU work. Commit `f69e322` replaces the optional-only array
with one always-nonempty complete argument list and adds a source regression.

## Accepted Free-GPU Run

The accepted run used free `gln01` V100 GPU 0 at commit
`fee229e1d3d12b79e9e22d063ab4f8e09d57c7bc`:

- output:
  `runtime/outputs/training/target-oriented-continuation-admission-fee229e`;
- report SHA256:
  `eff3ddec5a7d767b98f45b88f65ece49165bf044ff1cc471736fb129cf1ad1a1`;
- parent stage SHA256:
  `099d48c76277cafb04daf868c492200d0271fd6a4a2c6fa29ca6e4bdbe7989a5`;
- parent executable commit:
  `e363482a75cbec3d6eaaa46555cf7191f3f55350`;
- backend: GPU (`cuda:0`);
- report status: `completed`;
- elapsed time: 920.78 seconds;
- peak live JAX device use: 8,187,984,384 bytes.

The parent loader verified parameters, AdamW first/second moments, optimizer
step 250, all 250 prior update records, the frozen contract and validation
panel, and the exact stage-file hash.

Update 251 used the absolute learning rate `9.999997063589093e-5`. Its loss
was `8.7882637404`, unclipped gradient norm `284116.125`, clip scale `1.0`,
and parameter-step norm `0.1019078791`. Budget failures, discrete mismatches,
nonfinite predictions/states, negative stocks, and unexpected mask mismatches
were all zero. The 99 declared-dynamic mask mismatches remain report-only.
The update-251 recovery file was written and reloaded successfully.

The matched admission panel uses the same one window in each of six
slice/horizon cells before and after the update. Its macro changed from
`0.5522847075` at update 250 to `0.5019741924` at update 251, a `9.1095%`
relative improvement. `full_frozen_panel=false`, so this is engineering
direction evidence only and cannot promote or reject the checkpoint.

An earlier free run at commit `f69e322` completed the same valid update and
hard gates but compared the parent's complete 96-window result with the six
admission windows. Commit `fee229e` evaluates the matching six parent windows
before the update. Only the corrected report above controls admission.

## Next Operation

Prepare absolute updates 251-1000 on free CPU test nodes, merge the disjoint
chunks into one partial-range manifest, and independently admit its exact
coverage and hashes. The preparation may read the accepted update-1-to-250
prepared product to reuse retained exact inputs, but it must not include those
updates in the continuation manifest or training loop.

After the full prepared range passes, construct and test the paid Slurm
launcher against the exact manifest and checkpoint identities. Submission
still requires a separate explicit resource, command, wall-time, directory,
and worst-case cost approval.
