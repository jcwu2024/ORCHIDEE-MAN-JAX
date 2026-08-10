# Successor Small-Fit Checkpoint Handoff V1

Date: 2026-08-10.

Status: local implementation, fail-closed tests and free-`gln01` serialization
preflight accepted. No paid GPU fit is authorized.

## Reason For The Amendment

The accepted v1 bounded-fit contract deliberately wrote no checkpoint. The
production runner nevertheless named a separately authorized 7/30-day screen
as its next gate. A successful 500-update process would therefore discard both
trained parameter trees before that screen could consume them. Repeating the
fit would waste paid GPU time and would not preserve the exact evaluated
models.

The architecture, sampling, optimizer, loss coefficients and promotion gates
remain unchanged. This amendment changes only the successful production-fit
handoff. Submaximal probes and failed fits still write no checkpoint.

## Producer And Consumer Identity

The accepted 500-entry prepared manifest is immutable and remains attributed
to producer commit `d4936996635b02fcf04cd5f8c4926be3cc010124`, runner
contract SHA256 `8317fc74...d1640de`, schedule SHA256
`f6c0d36f...a74103b`, and prepared-manifest SHA256
`0182b7f7...d80462`.

The checkpoint-capable runner executes from a later clean commit. Admission
does not pretend that the old artifacts were produced by that commit. It
independently records and verifies:

- the immutable producer commit and runner contract;
- the prepared manifest, schedule, dataset, candidate and fast-owner hashes;
- all 500 manifest entries, their total byte count and retained trace
  signature through the existing prepared-manifest admission;
- the checkpoint consumer commit and this handoff contract hash.

The machine authority is
`manifests/coarse_graining/successor_small_fit_checkpoint_handoff_v1.json`,
SHA256 `a587b8a93b57bb69f06776aa80fd2268f6ebe2f2e13c07b70e45a3d6632315b0`.
It explicitly amends only the two prior `checkpoint_written=false` fields.

## Atomic Checkpoint Set

After both matched arms complete exactly 500 updates, the final zero-rate
evaluation, compile-reuse gate, loss-reduction gates and all hard gates must
pass. Only then does the runner transfer both parameter and optimizer trees to
host memory and stage:

```text
checkpoints/
  checkpoint_manifest.json
  source_routed_process_attention_v2.pkl
  source_routed_graph_mlp_control_v2.pkl
```

Every arm payload binds the complete execution identity, update count,
parameter count, final metrics, loss reductions, parameter-tree schema and
optimizer-tree schema. The manifest records both file hashes and sizes. The
staging directory is renamed to the final directory only after all files and
the complete manifest are flushed. A second-arm or manifest write failure
removes staging and leaves no published checkpoint directory.

The reader rehashes both files and checks exact arm coverage, identity,
updates, and the serialized and reconstructed tree schemas. It rejects
missing arms, path escape, hash drift, identity drift, nonfinite values and
shape/dtype changes.

## Local Acceptance

The focused successor suite passes `23/23`. It covers:

- frozen handoff contract admission;
- distinct producer and consumer Git identities;
- matched checkpoint write/read round-trip;
- parameter and optimizer tree-schema verification;
- rejection of failed and 499-update fits before publication;
- cleanup after an injected second-arm write failure;
- tampered checkpoint shape rejection;
- launcher requirements for the checkpoint contract and final manifest.

Ruff, Python compilation and `git diff --check` remain clean. The
serialization-only free GPU preflight passed at commit `c15cd86`: both arm
trees originated on GPU, persisted through host memory, reloaded exactly, and
left no checkpoint payload after the temporary directory closed. It performed
zero training updates and did not read Teacher or v5 data. Report SHA256 is
`a923d6371473bc19bef9ea89e3a390a4798b02ac13e23782d2520ab22616cf5a`;
see
[`successor_checkpoint_handoff_gpu_probe_acceptance_20260810.md`](successor_checkpoint_handoff_gpu_probe_acceptance_20260810.md).

The paid 500-update fit still requires a fresh resource and cost approval.
