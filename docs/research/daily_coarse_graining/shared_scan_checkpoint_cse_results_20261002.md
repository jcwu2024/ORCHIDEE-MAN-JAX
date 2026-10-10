# Outer scan checkpoint CSE screen

Completed 2026-10-02. **Close this `prevent_cse=False` treatment before public
T7/T30 or complete-update compilation.** The representative full expert5 T7
scan has no dependable speed improvement and fails the unchanged strict native
gradient comparison. No training, default change, precision change or paid job
was executed.

## Treatment and bounded scope

The source boundary is `rollout_loss.py::RolloutLoss._scan`, lines79–93. A
diagnostic JAX facade changes only the call to `jax.checkpoint(transition)` to
pass `prevent_cse=True` or `False`. It rejects other checkpoint names or keyword
arguments. Both arms retain default `policy=None`, one `remat2` and native JAX
AD. The project scan implementation is reused unchanged; no process logic was
ported or altered.

The representative step runs the **original full expert5 sample0 decoder**:
7542×1×384 query, 59 chunks128 including the original ten padded rows, decoder
width384/eight heads/two blocks, 3,548,928 decoder parameter coordinates. It
uses the preserved raw-bank payload and original component helper's map,
contractions, masks, gathers and scatters. No grouping or chunk reordering is
introduced. All neural values remain FP32. Dynamic inputs and raw banks have
18,683,520 derivative coordinates; the initial flattened query has2,896,128.

To create a bounded actual scan, the explicit diagnostic recurrence is
`q_next = q + 0.001*original_decoder(q, dynamic, raw_banks)` for seven steps.
The scalar diagnostic loss uses the original `sin(0.11*i)` synthetic cotangent
on every decoded output at each step. Native `value_and_grad` returns loss,
all seven decoded outputs, final query state, all decoder/dynamic/raw-bank
gradients and the complete initial-query derivative. Parameters and captured
floating inputs are dynamic executable arguments; masks/indices, seven day
indices and cotangent are fixed identically across arms.

This is **not the public recurrent objective or physical state**, and does not
certify T7/T30 science, physical hard gates or complete-update throughput. It
does cover the original 59-chunk body rather than a convenient first chunk.
Its recurrence/loss are explicitly representative diagnostics. AdamW is absent.

## Actual compiler evidence

Canonical GPU JAX0.4.38 reports `remat_opt_barrier=True`. Saved native source
`jax/_src/ad_checkpoint.py::_remat_lowering`, lines815–833, inserts an
optimization barrier for `differentiated and prevent_cse`; `False` bypasses
that branch. The compiled GPU results below establish an implemented treatment.
Zero final barriers does not prove the earlier barrier had no effect: final
fusion counts, memory and HLO differ.

| Evidence | True | False |
| --- | ---: | ---: |
| StableHLO optimization barriers | 1 | 0 |
| Optimized HLO barriers | 0 | 0 |
| `remat2` / default `policy=None` count | 1 / 1 | 1 / 1 |
| Optimized while / custom-call count | 6 / 57 | 6 / 57 |
| Optimized scatter / gather count | 8 / 16 | 8 / 16 |
| Optimized fusion count | 515 | 511 |
| Estimated live bytes | 2,394,760,620 | 2,376,650,412 |
| Backend compilation seconds | 10.663375 | 9.057331 |

The memory estimate decreases0.756%. Both arms are below the24GiB bound; this
is compiler-estimated live storage, not an independently measured peak.

## Seven synchronized pairs and correctness

Each arm runs once initially, then seven paired executions alternating order.
All timing calls synchronize complete native parameter/input AD results. A
compile guard rejects any new backend compile during these executions. Full
initial and paired results, executables, JAXPR, StableHLO and optimized HLO are
preserved. Reported gradient/output shapes and dtypes match between arms.

| Scope | Initial strict result | Seven paired strict results |
| --- | --- | --- |
| Loss, decoded outputs and final query state | Bitwise identical | Bitwise identical in every pair |
| Initial query derivative | Bitwise identical | Bitwise identical in every pair |
| All decoder parameter derivatives | 2012 failed coordinates; max abs3.814697e-5 | 1978–2144 failed coordinates in every pair; max abs up to4.577637e-5 |
| All captured dynamic/raw-bank derivatives | 8 failed coordinates; max abs1.907349e-6 | 1–9 failed coordinates in every pair; max abs up to2.413988e-6 |

Comparisons retain `rtol=1e-5, atol=1e-6`, including near-zero coordinates.
All initial values/derivatives are finite. All3,548,928 decoder parameter
derivatives and2,896,128 initial-query derivatives are nonzero;1,290,624 of
18,683,520 dynamic/raw-bank derivatives are nonzero in both arms.

Self-replay is diagnostic: both arms also fail their own strict derivative
comparisons (decoder failures1945–2159; raw-bank failures1–7). Thus the failed
cross-arm gate does **not isolate treatment-induced error**. Forty-one decoder
coordinates fail in all seven cross-arm pairs, while no raw-bank coordinate
fails every pair. This does not waive the gate or establish a causal numerical
bias. No tolerance or AD algorithm was changed to obtain passage.

| Stable native AD timing | True seconds | False seconds | Ratio |
| --- | ---: | ---: | ---: |
| All seven pairs, medians | 0.861735681 | 0.849175777 | 1.0147907x |
| True then False, four pairs | 0.868935017 | 0.847812911 | 1.0249136x |
| False then True, three pairs | 0.852472210 | 0.868381057 | 0.9816799x |

The small overall ratio reverses with execution order, so it is not dependable
speed evidence. The treatment changes compiled execution but is not admitted;
no public-model compile or complete-update measurement is justified by this
screen. This closes this specific outer-scan `prevent_cse=False` route, not all
checkpoint/execution optimization.

## Runtime and reproducibility

The free gln01 host was `ibc13b02n04`, using one TeslaV100-SXM2-32GB GPU1.
Immediately before preflight/component both GPUs had0MiB,0% utilization and no
compute processes. Host load at component start was0.17/0.10/0.31, and CPU
affinity was0–111. These are shared-node diagnostic measurements. Driver
525.85.12 and the accepted Singularity GPU runtime were retained.

Actual-environment `check_daily_training_runtime.py --backend gpu` passed in
23.232131s before the changed AD boundary; no Teacher data was read. Both
launchers have actual exit0. Component Python elapsed54.866865s, launcher
56.424245s; preflight launcher24.573444s. One-GPU launcher accounting totals
0.02249936 card-hours. Paid jobs0, training updates0. CPU-only saved-pair
analysis adds no GPU work. The component timeout was600s, per-arm backend
timeout240s; preflight timeout300s. Syntax, target `bash -n` and actual
`--dry-run preflight/component` argument expansion passed.

Source files:

- `scripts/hpc/probe_shared_scan_checkpoint_cse.py`
- `scripts/hpc/probe_shared_scan_checkpoint_cse_analysis.py`
- `scripts/hpc/run_shared_scan_checkpoint_cse.sh`

Local compact evidence directory:
`outputs/diagnostics/shared-scan-checkpoint-cse-20261002/`, especially
`component.json`, `analysis.json`, `preflight.json`, actual-exit records,
logs and both StableHLO/optimized HLO files. Source snapshots/manifest are
retained there. Large executable/full-result files stay on the remote shared
filesystem, avoiding another local3GiB copy.

Remote paths, all under the authorized project runtime:

- Immutable checkout: `runtime/checkouts/shared-split-width-v1` at
  `c599236bcf57b745e9bafed678298010f4c3b58f`.
- New overlay: `runtime/transfer/shared-scan-checkpoint-cse-20261002`.
- New results: `runtime/outputs/diagnostics/shared-scan-checkpoint-cse-20261002`.
- Existing helper overlay: `runtime/transfer/shared-command-buffer-loops-20261001`.
- Original payload: `runtime/outputs/diagnostics/shared-normalized-keys-payload-20261001.pkl`.
- Saved arrays: `component.{true,false}.initial.pkl` and
  `component.pair{0..6}.pkl`, with full paired results386,383,701bytes each.
- Saved binaries: `component.{true,false}.executable.pkl`; graph exports use
  the same stems with `.jaxpr.txt`, `.stablehlo.txt`, `.compiledhlo.txt`.

Run on the free GPU host with the absolute overlay launcher:
`bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-scan-checkpoint-cse-20261002/run_shared_scan_checkpoint_cse.sh preflight`,
then replace `preflight` with `component`. Output-preservation checks reject rerunning
over completed results; any future fresh diagnostic needs a distinct artifact
identity. The launcher does not contain or issue `sbatch`.
