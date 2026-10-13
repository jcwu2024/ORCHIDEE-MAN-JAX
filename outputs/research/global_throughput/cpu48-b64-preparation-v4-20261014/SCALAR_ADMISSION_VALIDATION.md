# Scalar Admission Validation

This is unpaid preparation evidence only. The scalar audit is an internal
comparison against the saved GPU reference; it is not scientific acceptance,
and it has not been bound into a CPU scalar admission record.

The final executed receipt is [scalar-audit-receipt-v3.json](scalar-audit/scalar-audit-receipt-v3.json), SHA-256
`cb1331edb7094e74ec1814c6813a0eee218c58ea4e89f1c0c1f630e981479431`.
It records recomputation of all 64 lane gates (61 pass, 3 tolerance-only
failures across 4 leaves), recomputed prepared lane identities, and verified
point order for 64 unique points. Frozen-reset arrays were not retained for
independent recomputation; the saved reset gate remains a sealed proof only.

The exact staged script used for that run is [audit_cpu_scalar_admission-v3.py](scalar-audit/audit_cpu_scalar_admission-v3.py), SHA-256
`90a1056f7ace3d7ecb7a3e322dbbd8f69e52302ed3ebe19b0caa111568985ff6`.
The reusable working-tree script later gained a checkout-root import bootstrap
and is a different source snapshot. [Execution notes](scalar-audit/audit-execution-notes.md)
preserve the actual absolute-path `srun` invocation, captured stdout and exit
status, the receipt's inaccurate shorthand launch-string field, and the
separate initial `PYTHONPATH` setup failure.

Verification evidence is preserved without rerunning unchanged tests:

- [Reader-inclusive, pre-ordering-change run](scalar-audit/reader-inclusive-tests-62.stdout.txt) ran collector, saved-reference metadata, and runner tests: 62 passed. The saved-reference metadata module and its implementation were unchanged by the later gate-order correction.
- [Post-ordering-change run](scalar-audit/collector-benchmark-tests-43.stdout.txt) ran collector and runner tests after the permutation/replacement early-stop order was restored: 43 passed. The synthetic sidecar test uses production gate emitters and sealing, and checks array, index, and hash tampering.

The post-change source keeps permutation and replacement checks immediately
after their model calls, records the frozen-reset check after warm repeats, and
seals candidate arrays afterward. Batch-only parent accounting validates only
the new batch directory; reused source-parent cost is recomputed from its
verified raw accounting row and kept separate from the new allocation.
