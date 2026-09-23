# Split-width daily admission result

Collected after free launcher112232 and CPU preparation67596 exited0.
Code:a1fa82a; model:shared_transformer_split_width_v1.json; fresh seed20260921;
64,884,481 trainable parameters. No paid job or formal learning in this run.

## Completed evidence

- Daily GPU report `runtime/outputs/diagnostics/shared-split-a1fa82a-daily.json`:
  passed `one_day_two_update_technical_admission_only`,5072.24s (84.54min).
- All64,884,481 gradient coordinates finite. JVP11.1191374592 versus reverse
  projection11.1191364885, relative difference8.73e-8, unchanged tolerance passes.
- Two changing-data micro4 AdamW updates completed. Both updated forwards and
  final cross-sample replay have zero hard physical/numerical counters.
- Fixed-data micro4 value/gradient repetitions:0.600,0.527,0.526,0.522,0.518s;
  median0.526352s. This is four one-day rows, not32 rows, not a complete joint
  update, and not inference-only timing.
- Compiled gradient live estimate10.4869GiB; allocator actual cumulative peak
  bytes-in-use11.2171GiB. The reserved JAX pool is25.3913GiB, which must not be
  described as actual model memory. Previous width768 field-local daily32/micro4
  run reported23.405GiB actual peak, useful context but not a matched whole-run
  comparison (different runner/objective scope).
- Main reverse graph compile2992.5s (~49.9min); directional stage1253.3s
  (~20.9min). Cold compilation remains expensive; warm timing does not establish
  shorter startup or full500-update budget.
- CPU report `shared-split-a1fa82a-recurrent-cpu.json` passed in819.50s,
  with two actual TRAIN artifacts and T7/T30 forward/gradient shape traces.
  No multi-day model numerical execution or numerical gradient acceptance.

## Learning limits

Each immediate update lowers its own batch loss:0.35448->0.15473 and
0.66574->0.51663. After both updates, batch0 total loss is0.39232 versus initial
0.35448; batch1 is0.51663 versus initial0.97149. Thus do not present both updates
as uniform improvement or evidence of final accuracy. Both batches' aggregate
flux/stock penalties improve after two updates, while batch0 process penalty
regresses. Only eight daily rows are involved. No held-out or long-term accuracy
claim follows, and zero-RH coverage remains a limitation.

## Next operation

Run the already prepared native-AD GPU T7/T30 admission with the new daily report,
CPU preparation and split protocol. Then use local61a9671's
`run_shared_learning.py --technical-updates 2` to measure the actual32/micro4
plusT7/T30 full update before budgeting fresh500 learning. That timing extension
is not yet deployed. Keep accepted daily gradients and checkpoints; do not rerun
daily admission for report edits. No new task was launched while collecting
these results.
