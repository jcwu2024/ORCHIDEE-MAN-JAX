# Dominant T2 state-gradient discrepancy: PFT14 reserve carbon

Flat index2250 is `slowproc_stomate_previous_step_state.biomass[0,1,7,0]`,
the PFT14 reserve-carbon pool (`ICARBRES`, zero-based7), not padding or DOC.
Its raw derivative differs by1.54691 and contributes99.78% of the initial-state
gradient discrepancy squared norm. Native/streamed derivatives are1469.15114
and1470.69805, respectively.

Actual frozen-case input:

| Quantity | Value |
| --- | ---: |
| Initial reserve carbon | 9.98185346855e-9 |
| Fitted mean | 4.24204610866e-9 |
| Fitted scale | 4.66846019226e-9 |
| Float32 standardized arithmetic | 1.22948610783 |
| Float64 standardized arithmetic | 1.22948619534 |

Multiplying the raw native derivative by this input scale gives6.85867e-6 per
one fitted standard-deviation displacement; the corresponding derivative
difference is7.22169e-9. This is a diagnostic change of units, not a model or
tolerance change. A large dimensional gradient alone does not establish an
unstable or incorrect derivative. It does not establish that all other
coordinates agree, or that the raw discrepancy is harmless rounding.

## Source-backed interpretation

PFT14 maps to MTC2 in `configs/orchidee_man_250919.yaml:203`. The MTC2 path is
evergreen/pheno_model none. In its usual leaf-present, non-senescent allocation
branch, `ecureuil=0` leads to no new reserve allocation:
`fortran_source/ORCHIDEE/src_stomate/stomate_alloc.f90:763` and
`src_parameters/constantes_mtc.f90:709`. This is a conditional property, not
an unconditional fixed-zero theorem.

Reserve still participates in maintenance and biomass updates. When allocatable
carbon is nonpositive, the daily deficit is added to reserve in
`src_stomate/stomate_npp.f90:383`. Tissue pumping can also remove reserve.
The negative-stock repair in that subroutine, around lines497--516, tests
`biomass < zero` and restores negative biomass to `min_stomate=1e-8`, adjusting
respiration for the created carbon. It does not clamp every positive value
below1e-8. `jax_orchidee/stomate/carbon_kernels.py:3993` retains those no-alloc,
tissue-pump and negative-only repair semantics.

Thus9.98185e-9 is compatible with a previous negative-stock repair followed
by small maintenance depletion; its being below1e-8 is not itself a defect.
The precise preceding branch cannot be inferred from that single value.
This state must not be silently zeroed to eliminate a difficult derivative.

The four parameter-coordinate exceptions are in expert5, `litter_poc_doc`,
as ordered by `strong_baseline_axis_aligned_process_expert_v1.json` and
`strong_baseline_contract.py:41`. Shared context can connect biomass input to
that expert, so this location alone does not prove a litter/DOC process error.
The active replay/JVP/FD and native-checkpointed diagnostics test the cause.
