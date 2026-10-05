# Exact prepared-argument transport

The first Linux NEW60 rehydration failed strict identity at `215.0-115.0`.
The independently saved matched snapshots have identical 536-leaf structures,
state layouts and dtype/shape inventories. Eight FP64 leaves per block differ
in actual bytes; these differences are not a tree representation or ordering
issue. Preserve the comparison at
`outputs/research/global_throughput/scale-implementation-checks/point215-exact-input-comparison.json`.

The differences occur in regenerated preparation helpers:

| Numerical inputs | Frozen source preparation |
| --- | --- |
| DIFFUCO `lai_light.laitab` and `light` | `diffuco.py::diffuco_lai_light_table`, lines 2392–2421, exponential LAI/Beer light tables; Fortran `diffuco.f90::diffuco_trans_co2` 2214–2234 |
| Hydrol `mc_lin`, `k_lin`, `a_lin`, `b_lin`, `d_lin` | `hydrol.py::build_mineral_cwrr_tables`, lines 5875–6044; assembled by `orchestration.py::_compiled_hydrol_table_arrays` 10212–10225; Fortran `hydrol.f90::hydrol_init` 4150–4159 and 4172–4244 |
| Half-hour `forcing_days.swdown` | `orchestration.py::_paper_compiled_forcing_day` 10131–10196, `domain.py::_solarang_gswp` and solar redistribution 383–561; Fortran `solar.f90::solarang` 55–181 and `readdim2.f90::forcing_read_interpol` 1526–1550 and 1620–1639 |

The local preparation used Windows Python 3.11.15, JAX 0.10.1 and NumPy 2.4.6;
the Linux diagnostic used Python 3.11.15, JAX 0.4.38 and NumPy 2.2.6.
The evidence establishes actual low-bit regeneration differences;
it does not isolate a particular library, platform or compiler cause.

`scripts/dev/prepare_global_frozen_arguments.py` exports one consolidated
NEW60 collection from the saved completed local FULL64 input. Every full
first/second `BlockArguments` keeps its original module/class, nested container
order, array shape, dtype and exact bytes. No formula reruns, Day1 generation,
physical forward, casts or tolerance waiver are involved. Both original
per-point block digests and cold-state boundaries are checked before export.
Existing input SHA metadata remains provenance; the new transport file receives
its own SHA256. The four original remote anchors remain their existing inputs.

Remote rehydration requires `--frozen-arguments` and
`--frozen-arguments-sha256`. It checks transport SHA, FULL64 ordered selection,
source audit/config/prepared provenance, original static signatures, exact
audited VEG-only metadata mapping, source-rebuilt Linux static closures and
full numerical schema, original per-block digests, exact cold packet values and
state layouts, seven science days per block and 48 forcing steps per day.
Independently rebuilt Linux contexts/templates provide metadata and schema
checks; the scientific numerical inputs are the original frozen arrays.
The existing native-dtype/full64 reader screen remains required after binding.

This is input IO preservation, not numerical acceptance of regenerated values
and not scientific forward/gradient or throughput acceptance. Validation and
export details are recorded in the generated export and check reports under
`outputs/research/global_throughput/scale-implementation-checks/`.

## Local acceptance

The completed export is
`outputs/research/global_throughput/round4-inputs/frozen_arguments_new60.pkl`
(227,230,123 bytes, one payload file). Its SHA256 is
`9fce20a655505764bb840117f507ab7d56c2409815b64cf3d2a9a2e8fe5d23e3`.
The manifest and all 120 unchanged argument digests are in
`outputs/research/global_throughput/scale-implementation-checks/frozen-arguments-export.json`.
The export took 4.350475 seconds and passed the original FULL64 schema/native
dtype screen and every NEW60 cold packet value/layout/static identity check.

All 65 targeted tests passed (`test_global_frozen_arguments.py`,
`test_global_scale_inputs.py`, `test_global_scale.py`), including the full
remote adapter toy integration with actual frozen payload load/bind,
unchanged anchors and on-disk FULL64 reader admission. The standalone frozen
tests reject checksum, point/order, shape/dtype, value, static closure, state
layout/value/time boundary and forcing/day boundary tampering, and verify
original class/container/native dtype identity. No physical model executes in
these IO tests. Linux admission and GPU/scientific scale validation remain
separate, pending root execution.

## Explicit named-state representation migration

The first frozen Linux admission verified the original NEW60 input digest and
its own Linux static closure, then stopped before numerical GPU validation at
cross-anchor `state_layout` compatibility. Exact saved-state inspection shows
that all four original remote anchors use one ordering of the 93 fields in
`sechiba_finalize_state`, while NEW60 uses a different ordering. Component
order is unchanged. Every original anchor's positional state arguments match
its original packet; changing its field labels alone would misassociate
numeric values, and a tuple permutation changes both schema and digest.
The evidence is `scale-implementation-checks/round4-state-layout-comparison.json`.

Keep the original remote4 packets, arguments and reference identities
untouched. Each NEW60 lane first passes the original frozen argument digests,
static signature, cold values/layout and audited VEG mapping. Only then apply
the existing `prepare_global_scale_inputs.py::_restore_state_layout_order`
to that packet using the original remote anchor as the canonical template.
The adapter maps argument group0 (`values_by_component`) explicitly by
component, field and recursive nested path. It proves that classes,
dtype/shape/raw bytes and the cold packet association stay identical by name;
the other eight argument groups remain identical objects with unchanged
schema and numerical digest. Packet time remains 47. The lane packet,
argument slots and recomputed scientific static signature all use the same
canonical layout before cross-lane admission.

Original positional digests remain required IO provenance. Bound positional
digests describe the documented permutation and can differ; they are not
substituted for original hashes. Each remote point record retains original
and bound block digests, the complete field-index bijection, named nested
class/dtype/shape/byte inventories and source-backed packet migration proof
for both blocks. Duplicate, missing or extra names, wrong axes and changed
named values are rejected. No process formula, dtype, tolerance or value
changes are admitted.

All 75 targeted IO/scale tests passed, including nonzero equal-shape state
values: the correct named permutation preserves alpha=3 and beta=7; a
schema-identical value swap and a corrupted migration are rejected.
The actual saved NEW60 collection passed the full 60-point/two-block offline
proof in `scale-implementation-checks/round4-canonical-state-io-proof.json`.
All 120 original hashes were verified, all named-state/non-state proofs
passed, and all eight original remote4 argument hashes stayed unchanged.
This check uses local valid context metadata and the actual original remote4
state layout/schema; it does not prepare Linux contexts or execute a model.
Linux acceptance still requires the bounded root-run admission.
