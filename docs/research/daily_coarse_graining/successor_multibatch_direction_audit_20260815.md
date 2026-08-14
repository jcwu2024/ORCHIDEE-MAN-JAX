# Accumulated Multi-Batch Direction Audit, 2026-08-15

## Decision

Do not admit the ordinary equal-mean multi-batch direction for training under
the frozen all-component gate. Increasing the effective update batch tenfold
greatly reduces instability, but averaging gradients by component still hides
conflicts among point-year groups.

This is useful positive attribution: batch coverage is materially relevant.
It is not sufficient by itself, and it does not reject the Attention
architecture.

## Protocol

The frozen contract SHA256 is
`3c3cd06828f9300723e8e53e5708b7e02e386025daa0a9a717c10a948edfcd8d`.
At the frozen update-1000 Attention parameters, the audit accumulates the four
component gradients across ten scheduled mixed batches. The effective single
update therefore sees 40 point-years, 320 one-day anchors, and 40 independent
30-day windows. Sequential gradient accumulation is mathematically identical
to an equal-mean large batch at the same parameters without retaining ten
rollout graphs simultaneously.

The four accumulated component gradients are unit-normalized and combined by
the minimum-norm simplex rule. Virtual global parameter norms
`0.00003, 0.0001, 0.0003, 0.001` are evaluated on every contributing batch.
No update, checkpoint, validation value, or sealed-test value is retained.

## Result

The free-`gln01` run at commit `ab7f41a` completed with valid evidence:

- runtime report:
  `outputs/training/successor-multibatch-direction-ab7f41a/report.json`;
- report SHA256:
  `0601fcd6bc9f51b638bb2de14087be152320e65fa6430e5ddcab254990c0d847`;
- elapsed Python time: `930.29 s`;
- peak JAX device bytes: `13,023,250,688`;
- hard failures and nonfinite values: zero;
- optimizer updates and retained proposals: zero.

The aggregated direction has minimum component cosine `0.2736`, so the four
mean objectives have a clear common first-order direction. The simplex weights
are `0.4901` anchor, `0.3507` AGB tendency, `0.1593` BGB tendency, and zero
explicit NPP weight; NPP is already descended by the other components.

No step passes the group-aware gate. Norms through `0.0003` improve the median
NPP, AGB tendency, and BGB tendency, but only `4/10` groups improve every
component. At norm `0.0003`, the maximum group-component regression is only
`0.0323%`, compared with `19.11%` in the per-batch sequential smoke. Thus the
tenfold larger effective batch materially stabilizes the update but does not
make the mean direction common to individual groups. Norm `0.001` crosses a
nonlinear boundary and has a maximum regression of `28.89%`.

## Next Gate

Do not train the ordinary averaged-gradient direction. Freeze a zero-update
group-aware feasibility audit over the same 40 group-component gradients.
Normalize each group-component gradient separately and solve the 40-objective
minimum-norm simplex problem. This tests whether a true first-order direction
exists that respects the heterogeneous point-year groups, rather than merely
the four objectives after averaging groups.

If no positive group-aware common direction exists, close gradient-combination
optimizer work and redesign the objective or sampling unit. If one exists,
evaluate a bounded virtual step grid on all ten groups before considering any
sequential update. No architecture change, validation read, or training is
authorized by this audit.
