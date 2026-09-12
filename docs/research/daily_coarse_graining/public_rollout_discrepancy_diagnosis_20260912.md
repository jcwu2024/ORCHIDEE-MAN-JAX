# Diagnose the finite T2 derivative disagreement

Freeze before execution. Reuse the saved reference and streamed gradients from
the unchanged artifact322/window1/T2 fresh case. No training, new initialization
choice, Teacher regeneration, loss revision or relaxed acceptance threshold.
Paid job15048786 uses its independent12bac94 checkout and is untouched.

## Existing evidence

Loss agrees. Parameter relative L2 difference9.3602e-5 and initial-state
difference1.05105e-3 fail the declared component/coordinate checks. The four
parameter-coordinate exceptions are expert5/query/bias entries193/216/219/232.
Index2250, biomass[0,1,7,0] (PFT14 reserve carbon), contributes about99.78% of
initial-state discrepancy squared norm. Its fitted input scale must be read
before choosing physical perturbations; it is not part of the DOC log-tail.

## Two independent free-GPU diagnostics

1. Run the existing native `checkpointed` stage separately against the saved
   reference. JAX owns that reverse recurrence. Its agreement pattern and
   memory estimate help distinguish a host-adjoint problem from effects shared
   by rematerialized calculations, and assess a simpler final execution route.
   This diagnostic is allowed despite the failed core comparison; it does not
   bypass multi-day training admission.
2. `diagnose_public_rollout_discrepancy.py` observes the existing one-day VJP's
   already-returned primal and compares it to the saved forward trajectory.
   The optional observer cannot replace a state or cotangent; default behavior
   and numerical VJP are unchanged. Repeat streamed/direct stored-value checks
   detect whether this observation changes results. Visible primal equality
   alone cannot establish equality of hidden float32 intermediates.

The second diagnostic uses one combined JVP executable for parameters and
initial continuous state. It evaluates the normalized parameter-gradient
difference direction and a one-hot initial-state direction at the largest
discrepancy coordinate (2250 in the frozen evidence). Compare JVP projections
against both saved reverse methods. Then compare rounded central FD at three
scales, preserving every finished direction/epsilon in the report.

Parameter center epsilon is min(1e-4, max(1e-10, 0.01*max(abs(loss),1e-6)/
max(abs(reference_projection),abs(streamed_projection),1e-6))). State center
is min(fitted_safe_scale*1e-2, the same first-order loss-change bound), floored
at8 native float64 spacings of that state value. Probe4x,1x,0.25x,0.0625x and
0.015625x the center. The range was broadened before GPU numerical execution
after the CPU bootstrap measured reserve value9.98185e-9 and scale4.66846e-9:
normalized steps1e-4 alone can be too small for a float32 network's FD estimate.
This brackets truncation/quantization without making another expensive JVP graph.
Scales depend on saved sensitivity and actual input scaling, not FD outcomes.
Report raw state/mean/scale and float32 versus float64 standardized values.
Track physical gates and defined/discrete stability under every displacement.

These are diagnostic measurements, not an automatic pass/fail amendment. If
FD has no stable scale interval, report that limitation. If both reverse
methods are similarly consistent with independent evidence, investigate
conditioning and tolerances; if one systematically disagrees, localize that
path before changing code. Do not equate small relative error with harmlessness.

## Bounds and preparation

Local observer/projection tests first. The new combined parameter/state JVP
gets a real-data CPU abstract trace. Before either expensive GPU diagnostic,
run the exact-runtime tiny gate on that GPU. One free GPU per independent
diagnostic, at most90minutes each; no paid submission. Use durable results from
the existing runtime stage directory and new output names. Native checkpointed
uses the already CPU-traced numerical graph. The other diagnostic records the
JVP compiled-memory estimate before execution and stops above18GiB.
