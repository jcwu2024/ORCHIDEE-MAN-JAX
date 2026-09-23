# Actual-decoder dense-stage packing: bounded admission

Status: component evaluation completed; numerical checks pass, performance
benefit is modest and micro4 temporary memory increases. Not adopted. No
change to the default model, weights, loss, data, or training schedule.

## Completed paid result

Job15344733 COMPLETED0:0 in3m42s on ibc13b03n04. Estimated charge using the
approved rates:222/3600*(2.20+8*0.07)=0.1702yuan. Both micro stages completed.
Times below are median warm component output-and-gradient executions, not
complete daily updates. Temporary memory is compiler-estimated decimal MB,
not total live memory or the cumulative allocator peak.

| Micro | Variant | Median ms | Temporary MB |
| --- | --- | ---: | ---: |
| 1 | reference | 7.581 | 100.762 |
| 1 | packed-all | 6.835 | 100.740 |
| 1 | packed128 | 8.403 | 123.072 |
| 4 | reference | 14.695 | 145.131 |
| 4 | packed-all | 13.916 | 200.996 |
| 4 | packed128 | 15.869 | 294.318 |

Packed-all reduces time9.84% at micro1 and5.30% at micro4. Micro4 temporary
storage increases38.49%; packed128 increases time and memory at both sizes.
All output,14,372,865 component parameter-gradient coordinates and payload
gradient coordinates pass unchanged tolerances. Packed-all outputs match
reference exactly in these runs; parameter relativeL2 is2.16e-7/micro1 and
2.33e-7/micro4. This does not certify the full75.58M model.

Recorded compile-or-cache-load seconds for micro1 are36.49/16.34/17.29,
and micro4 are21.87/17.51/17.97, in reference/packed-all/packed128 order.
Do not extrapolate these component figures to full-model cold compilation.

Decision: retain as diagnostic evidence, do not integrate by default or launch
another expensive full-model compile. A5.3% component time reduction with
increased temporary storage does not resolve the current training-throughput
and memory regression. Next prepare an evidence-based decoder cost/architecture
proposal; architecture or capacity changes require discussion before building.
Reports preserved locally in outputs/shared-dense-stages-15344733-micro1.json
and outputs/shared-dense-stages-15344733-micro4.json and at remote paths below.

## Historical launch snapshot

Approved paid job **15344733** started on ibc13b03n04 at17:27:27CST
September23(server clock), status RUNNING at17:27:52. Allocation matches the
envelope below. Immediately before submission the node was IDLE, CPUAlloc0/112,
CPULoad0.01. Startup records host load0.03, GPU0MiB/0%, CPU affinity0-3,56-59.
Results: runtime/outputs/diagnostics/shared-dense-stages-15344733-micro{1,4}.json;
log: shared-dense-stages-15344733.log in the same directory. Collect component
numerical, memory and timing results before deciding on integration.

## Mechanism and reference

`shared_decoder_dense.py` keeps production field-local normalized attention
and field-local differentiable K/V banks. Within each coordinate block it
concatenates independent query rows after attention, applies the shared
residual normalization/FFN/normalization, then splits before the next block.
The readout is also row-packed. Optional dense row chunks bound intermediate
hidden arrays. Attention masks and carbon ownership are never concatenated
across fields. Layer normalization remains over feature width only.

Reference uses the unchanged production
`_run_coordinate_blocks_reusing_projections` and readout. Parameter/payload
gradients include projection construction inside the differentiated function.
This replaces the earlier generic outer-map proxy with actual decoder helpers;
it does not claim that reduced graph size automatically reduces live memory.

## Completed checks

- Four focused local tests pass (29.21s): unchunked and two padded chunk sizes,
  heterogeneous fields, GPP carbon route, all-masked attention row, inactive
  outputs, nonzero K/V biases, complete parameter/payload VJP and finite outputs.
- Main local benchmark width16/three fields passes reference, packed-all and
  packed8 output/parameter/payload comparisons.
- Actual JAX0.4.38 CUDA tiny benchmark also passes all three variants. Largest
  packed8 parameter difference4.77e-7, relativeL2 3.64e-7, zero failed
  coordinates; forward max2.98e-7, payload max2.24e-8. No tolerance changes.
- Runner shell syntax and actual paid-branch `--dry-run` expansion pass for
  both requested microbatch sizes. Ruff and diff whitespace checks pass.

Local reports: `outputs/shared-dense-stages-local.json` and
`outputs/shared-dense-stages-cuda-tiny-20260923.json`; deployed tiny equivalent
under runtime diagnostics. The first deployment attempt stopped before model
compilation because a transfer-directory script chose the server root package
instead of the declared checkout. The probe now honors `ORCHIDEE_REPO_ROOT`;
the corrected deployed CUDA execution above verifies actual import/execution.
A chunked-readout reshape issue was corrected during local testing, before GPU
execution. These were new prototype/launcher issues, not training-gradient faults.

## Prepared performance measurement

`probe_shared_dense_stages.py` uses production parameter initialization and
helpers at width768/two blocks, four heterogeneous fields of64/61/58/64queries.
It compares reference, packed-all, and packed128 at micro1, then micro4 only if
micro1 passes. Fixed forward rtol2e-5/atol2e-6 and gradient rtol5e-4/atol2e-5.
All variant timing/memory and numerical results are saved, including rejection.
Compiler temporary/argument/output bytes are reported separately from cumulative
process allocator statistics. Fragment estimates above12GiB stop before execution.

These synthetic payloads do not represent all161fields, the full2386-token GPP
bank, tokenizer, physical updater, retained tail or recurrent state. Passing
does not certify a complete update or demonstrate scientific learning. Adoption
still requires real-layout integration, parameter/state tests and saved-reference
daily/T7/T30/full-update checks. Do not launch a full-model cold compile merely
because this fragment compiles.

Both free GPUs were occupied at the performance-stage check. The tiny CUDA run
on GPU1 is numerical evidence only. Paid node ibc13b03n04 was IDLE with
CPUAlloc0/112 and CPULoad0.01; inspect again before an approved submission.

Prepared allocation: gnall,1V100,8CPU,32GiB,30min maximum, no node exclusivity.
Worst case: GPU2.20*0.5 + CPU8*0.07*0.5 = **1.38yuan**, using deployment rates.
Each micro stage has12min process ceiling within the30min allocation.
Working directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-dynamic-local-v2`
(unchanged f220013 production helpers). Four small overlay files are under
`runtime/tools/shared-dense-stages-20260923`; no new full checkout/data files.
Command from that working directory:

```bash
/rmprog/slurm/v22.05.7/bin/sbatch --nodelist=ibc13b03n04 /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/tools/shared-dense-stages-20260923/job_shared_dense_stages.sh
```

User explicitly approved this envelope; submitted once as15344733. Source scripts:
`scripts/hpc/job_shared_dense_stages.sh` and `run_shared_dense_stages.sh`.
If free GPU availability returns first, the same bounded probe can use it
without a paid submission; record contention and do not confuse noisy shared
timings with an idle-node performance result.
