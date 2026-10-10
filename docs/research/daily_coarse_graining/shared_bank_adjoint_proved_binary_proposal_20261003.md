# Reviewed actual-IR saved-binary T7: proposed500s numerical window

Status: explicitly approved and executed on2026-10-03; **scientifically terminal
and closed** at302.590292s/0.084052859 free card-hours/CNY0. First+3pairs and
independent NumPy complete: hard/loss/aux/all-parameter gates pass, states fail
11/10/10/10, warm1.009518x fails1.05. No adoption or further GPU time approved.
The following records the executed resource/method contract. Prior600s
allowance is closed. CPU analysis and independent review now admit the exact
saved candidate's actual IR structure:28 deduplicated field/scatter shapes,
12 multi-chunk F64 carry links and3 macro loops. This closes the parser gap;
the subsequent numerical/memory/speed verdict is recorded above and in the
linked bank-adjoint result report.

Approved resource envelope: one free gln01 physicalGPU1 window, total500s including setup, ordinary
repairs, result checks/writes and10s termination grace. Main cap490s, every
model call at most30s, each deserialize at most120s. Worst case one GPU,
0.1388888889 free card-hours, CNY0; no paid fallback. Inspect GPU1 occupancy
before starting and record node/load/affinity/visible GPU processes. OMP4,
OpenBLAS1, canonical JAX0.4.38/container and the existing LLVM flag remain.
Working directory is the immutable checkout:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1`
at `c599236bcf57b745e9bafed678298010f4c3b58f`.

Exact executed command:

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-bank-adjoint-accumulation-20261002/run_shared_bank_proved_binary_T7.sh --gpu 1 --candidate-sha256 f0e0a5eabd924533f94f8a636b173b433d61e82d20ff0cb21c326d7d89b99b01 --saved-IR-proof-sha256 d684cdb805a310a89bf6d5642f1b7b16b71065a50632fd36c34551c4e2e0cd1b
```

New output is
`runtime/outputs/diagnostics/shared-bank-adjoint-proved-binary-T7-20261003`.
Resume preserves the original target-clock origin/deadline, candidate and
certificate; exhausted budgets and terminal hard/scientific outcomes refuse
resume. The500s family completed its scientific measurement and will not resume.

The certificate checks bytes/JSON before GPU initialization. It hashes the
220,794,649-byte candidate, original compile report, actual capture report,
133,865,088-byte IR, captured sources, final offline analyzer/report and immutable
independent review. The final analyzer is `cb733504...e91f`, offline report
`094cc3bd...41414`, review `87c9cd50...61763`; exact pins are in the certificate.
The run reuses this proof with zero HLO-print/analyzer/review-wait calls. Both
binaries are then loaded once and their actual input ABI and memory estimates
are checked. No objective, lowering, backend, new preflight or JVP is built.

Pre-execution budget basis used earlier measurements: inputs ready6.2s, original deserialize
96.53s, candidate95.68s; prior startup55.7s gives a slower observed alternative.
Thus roughly240–290s of the490s main cap remain for full values, host checks and
writes after setup/loading, before certificate hashing overhead. The original
first call took2.775s and its saved bridge/write stage about0.77s. Candidate call
and complete paired output IO were then unmeasured;500s was a bounded measurement
attempt. It completed all pairs; actual results are in the bank-adjoint report.
Removing the actual10.61s capture,
30.9s analyzer and184s unsuccessful resident wait is the concrete saving.
The first results plus three paired full outputs require about2.1GB; keep all
completed arrays and classify timeout separately from numerical rejection.

Retain original T7/artifact742/update0, all64,884,481 parameters/300 leaves and
3854 states; FP32 projections, only four local axis/key/value banks widened F64
outside the original macro map, gathered values narrowed F32 before neural
logic, native AD/chunk128/source masks/physics unchanged. Strict
`rtol=1e-5, atol=1e-6` loss/auxiliary/all-parameter/all-state comparisons,
finite values,14 hard zeros,112 masks,21 exact-zero undefined gradients and
estimated/observed24GiB admission remain. No self-drift waiver.

Save first full candidate and three synchronized interleaved full-result pairs
even on numerical failure after hard admission. Hard failure saves/stops.
Original JVP33/199/428/200 supplements only. The1.05 timing screen is diagnostic;
no training, T30, complete-update throughput, architecture/loss/sampling or
tolerance change, policy adoption or paid job is authorized.

Readiness is
`outputs/diagnostics/shared-bank-adjoint-saved-binary-T7-600-20261003/proved-binary-preparation/readiness.json`.
Four actual-bundle metadata-only CPU checks pass on test01; five mocked500s
deadline/certificate controls pass locally. Actual Bash full/resume `--dry-run`
and syntax checks pass on test01, source staging hashes match. Eleven focused
strict-parser controls and independent proof/admission review pass. Existing
compile-pin/provenance, serialized-class import and canonical printer checks
are reused. These were preparation evidence; the complete executed model
measurement now rejects the route under its unchanged state and speed gates.
