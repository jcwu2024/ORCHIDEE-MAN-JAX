# Original native input-only VJP screen, 2026-10-01

Closed as an unadmitted bounded component. No public-loss split evaluation,
T7/T30, full optimizer update, training or precision adoption was started.

The candidate uses the original full128-chunk expert5 forward and native
`jax.vjp(lambda values: forward(parameters, values), inputs)`. Parameters,
inputs and cotangent remain dynamic; only parameter gradients are omitted
from this compiled closure. This changes the native derivative program and
is not a guarantee that compiled raw-input arithmetic remains equal.
No custom AD, stopped training gradient or saved-gradient replay is supplied
as a candidate derivative. Canonical source is
`shared_decoder_macro_packed._decode_mapped_rows` lines25–76 and
`shared_transformer._run_coordinate_blocks_reusing_projections` lines537–610.
Graph transport was closed; these are bounded exact-source claims.

The original expert5 sample0 capture retains all3,548,928 decoder parameters,
all21,579,648 raw-input coordinates, 59 chunks of128 with ten padded rows,
all2,896,128 component outputs and the original sin0.11 cotangent. The saved
original full VJP binary is replayed in the same process with zero baseline
compilations. This component cotangent is not the public scientific loss.

## Completed numbers

All outputs pass initial and seven strict comparisons (rtol1e-5/atol1e-6;
maximum1.430511e-6). Initial input gradients fail1,702 coordinates; seven
paired failures are1,709/1,715/1,715/1,715/1,722/1,722/1,719, with maximum
2.545118e-5. Both arms' input-gradient and output self comparisons all pass.
Inputs are finite, with4,186,752 nonzero gradient coordinates.

The raw interval diagnostic finds1,886 unique failing coordinates and1,585
persistent disjoint intervals beyond the largest strict allowance:
family_bank1,438; process_bank108; axis_bank39. Query, dynamic and local bank
coordinates all pass. At family_bank[0,7,11], all seven original values equal
0.8261800408363342 and all candidate values equal0.8262054920196533:
gap2.5451183319091797e-5 exceeds9.261800187232438e-6. Self stability therefore
does not explain this cross-program bias. No tolerance or noise waiver applies.

Original full VJP median is0.099285217s; native input-only is0.079865856s
(1.24315x for this omitted-output component). Input-only still consumes80.44%
of the original cost. An independent optimized parameter evaluator would
have to fit its entire forward and backward below0.019419361s to beat this
baseline. This is a measured cost budget; a combined split total was not run,
and it is not proof that every split policy must fail or cannot improve.

Candidate backend7.104047s; Python25.051543s; estimated live2,725,234,784bytes;
observed peak3,061,362,176bytes. Actual GPU0 runtime preflight passed:
checker23.474714s, launcher24.810516s (0.006891810card-hours). Component
launcher26.427460s (0.007340961card-hours). Total51.237976s,
0.014232771card-hours, one V100 per launch, paid0. Parallel GPU1 original T30
reference work is separate accounting and supplies no throughput gain here.
Diagnostic exit0 is the historical script's completed-report exit convention;
status remains `parity_failed`, and the executed launcher record is unchanged.

## Preserved artifacts

Local and remote component JSON, logs, launcher record, candidate binary,
original/candidate initial results, all14 repeat full raw trees and
candidate HLO are retained under
`outputs/diagnostics/shared-chunk-input-vjp-20261001` locally and
`runtime/outputs/diagnostics/shared-chunk-input-vjp-component-20261001-input.*`
on Explore1000. The reused original full-VJP binary remains at the original
remote `shared-command-buffer-loops-threshold-component-default-20261001.default.native_vjp.executable.pkl`
path, with its exact SHA recorded in the completed JSON; it is not duplicated
in this local output directory. `native-input-only-raw-intervals.json` contains every union
coordinate and per-arm seven-repeat interval. The local interval helper was
run once after its NumPy-index JSON serialization repair; exact paired counts
were verified against the completed JSON. Executed overlay remains immutable
at `runtime/transfer/shared-chunk-input-vjp-20261001`.

Disposition: close the component. No independently measured, numerically
admitted public loss split route is established.
