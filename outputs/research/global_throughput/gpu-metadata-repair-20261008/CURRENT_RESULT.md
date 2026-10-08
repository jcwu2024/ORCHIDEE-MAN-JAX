# Current result: causal lowering repair; B512 GPU measurement pending

The DIFFUCO PFT-mask host cache retained a `DynamicJaxprTracer` from an earlier trace. Canonical JAX 0.4.38 CPU sequential lowering reproduced B256 public/internal536/536 followed by B512536/673: one cached Boolean mask caused JAX to lift all137 constants. The immutable NumPy Boolean host-mask repair changes cache lifetime only. Both repaired stages lower536/536 with417 retained inputs and no lifted constants; earlier-stage metadata stays unchanged. See [causal evidence](CAUSAL_REPAIR.md) and [raw/signature summary](causal-evidence-summary.json).

The native shapes, dtypes, mask comparison and five insertion outputs remain exact. Canonical tiny natural reproduction and independent NumPy output checks pass. Local103 tests, changed-file Ruff and diff checks pass. Full-file DIFFUCO Ruff has9 pre-existing import errors outside the repair. Diagnostic CPU wall totals353.528515s of600s; real model compile/execution never occurred. Streamed canonical help/Python and actual Bash/direct+spool dry-runs passed without importing JAX, using the original immutable v3 source ROOT. Final checks from the new deployed checkout remain pending; the unchanged CUDA entry and repaired real GPU executable remain unvalidated.

The byte-pinned certificate records actual full source bytes separately from the canonical baseline science identity. Only the exact DIFFUCO helper replacement and two exact audited aggregate baseline representations (LinuxLF11b6db89… / Windows4151db25…) are admitted; no generic AST/EOL normalization is applied. [The104-file audit](checkout-byte-audit.json) matches each baseline file to immutable Git blobs or exact checkout CRLF. Current science identity11b6db89… admits both actual sealed comparators against FULL64, preserving original inputs, output schema,515 gates and760 native leaves. This is stored-reference admission, not new prepared-input or GPU-runtime validation; see [admission](b512-baseline-admission.json).

| Fixed ordered512 lanes;7168 lane-days | Core median, seconds | Prepared-wrapper median, seconds | Origin |
| --- | ---: | ---: | --- |
| B128,4 chunks |29.617320834|33.838030240|sealed job15600405|
| B256,2 chunks |27.024156772|30.468557430|sealed job15643868|
| B512,1 chunk |pending|pending|new proposal only|

B256 wrapper235.258923 lane-days/s costs an estimatedCNY2.833757201/2.928215775 per million lane-days, primary/conservative. Historical B128 lazy dispatch and B256 Python-bound dispatch differ, so the curve is not isolated batch-size causality. The target is the one-card wrapper cost/trajectory-day curve for future points×members;669-point timing needs separate tail/padding, initialization and IO evidence and is not a capacity limit. No global science, Fortran, annual,50-year,10× or long-horizon-gradient acceptance is implied.

The [new B512-only proposal](B512_PAID_PROPOSAL.md) imports B128/B256 without measuring or compiling them, reuses B256 actual compiler memory, and retains strict full-output/native/physical gates. It requests a separate45-minute4-CPU/1-V100 window atCNY1.80 primary/CNY1.86 conservative. Approval and submission are pending. Old attempts/budgets remain closed; no requeue, training, AD or automation was started. Code SHA and deployment receipt will be supplied by root after commit.
