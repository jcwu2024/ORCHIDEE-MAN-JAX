# Original chunk program latency scheduler screen

Status: closed after the bounded component. No public-model compilation,
recurrent/full-update acceptance or training follows this result. It provides
no demonstrated throughput improvement and does not fix G2/V4 state bias.

The treatment retains the original59-chunk`lax.map`, canonical128-row
projection/normalization/attention/FFN shapes, FP32 arithmetic, padding,
source masks and native full parameter/raw-input VJP. It requests only
`xla_gpu_enable_latency_hiding_scheduler=true`; original command-buffer
threshold5 and LLVM parallelism false remain unchanged. The captured
expert5 sample0 raw banks and original unscaled`sin(0.11*i)` cotangent
match the saved original component. Its original forward/VJP binaries are
reused with zero baseline compilations. No group-size, unroll, normalized
key, precision, loss or sampling treatment is mixed into this screen.

The reason for screening scheduling was actual original optimized HLO with
independent attention/FFN/projection operations and queue0/empty-wait
metadata. Those queue labels alone do not establish lack of overlap. Actual
JAX0.4.38 accepts the true compiler option. Its exposed DebugOptions does
not offer this field, and the live default proto has no latency-hiding
entry. Thus this evidence does **not** certify a false→true default change
or attribute an observed HLO difference specifically to that flag.

## Numerical and timing result

All2,896,128 outputs are bitwise identical in the initial comparison and
all seven alternating pairs. All21,579,648 raw-bank/input coordinates pass
strict comparison initially and in every pair. Parameter coordinates
3,548,928 fail initially at2,073 and in the seven pairs at
`[2178,2056,2104,2061,2030,1913,2154]`; maximum pair errors are
1.287460e-5–2.145767e-5. Self scatter records and complete raw repeat trees
are retained. These failures are not waived as self noise; numerical status
is`parity_failed`.

Warm original→candidate fullVJP median is0.102995550→0.101178106 seconds
(1.017963x), forward0.025068192→0.024769960 (1.012040x). The respective
repeat ranges overlap, includingVJP excursions to roughly0.115 seconds
in both arms. This short timing supplies no convincing throughput benefit.
Candidate compilation is2.615377 seconds forward /7.271269 seconds VJP.
Memory estimates are identical at2,260,028,960 live bytes; process peak
2,633,146,880 bytes. All reported values and derivatives are finite and
meaningful. Launcher36.624215 seconds /0.010173393 free card-hours, one
GPU, paid0. The executed component wrapper completes exit0 after recording
numerical failure; that exit is not admission and is not rewritten.

## Optimized HLO limits

Original/candidate HLO has44 GEMMs with identical shape multisets,2 while
instructions,0 async-starts,20 command-buffer calls and queue0 for every
GEMM. Full HLO text differs. Ten distinct GEMM configuration records differ,
including cuBLAS selected algorithms such as22→21 and2→4. Therefore neither
full execution identity nor new overlap is established, and this comparison
does not isolate scheduler effects from compilation/autotuning choices.
There is no justification for an expensive public-loss compile here.

Original nativeVJP HLO SHA256:
`a95dbfce8b129a43db98cf1c40a37e80a27678063b317da950eddd8cc40a2d95`.
Candidate nativeVJP HLO SHA256:
`3124ba8d285082dc4e21c6505b4b8caeb3dbeffe76246a6a989276849021ca9d`.
Exact configuration/shape comparison:
`outputs/diagnostics/shared-original-schedule-20261001/original-map-scheduler-hlo-comparison.json`.

## Artifacts and ownership

Dedicated overlay:`runtime/transfer/shared-original-schedule-20261001`.
Remote stem:`runtime/outputs/diagnostics/shared-original-schedule-component-20261001-lhs`.
Local results/binaries/HLO/raw repeats:
`outputs/diagnostics/shared-original-schedule-20261001/`.
Entrypoint:`scripts/hpc/probe_shared_original_schedule_component.py`.
The new generic component source accepts a named execution variant; the
previous G2 executed snapshot remains untouched in its original overlay.
Actual target-shell dry-run expansion was exercised before launch.
