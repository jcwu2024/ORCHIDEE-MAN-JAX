# Original dual-AD slicing acceptance result

2026-10-02: the authorized three-update diagnostic completed with
`strict_parity_failed`. Whole-tree slicing remains unadmitted for the full
parameter/state gate. All updates were discarded; no training, production
adoption, default change or paid job occurred.

The result uses the original F4S/T7/T30 executables, each deserialized once and
shared by the leaf/tree arms. Both model-entry and timed-update compilation
counters are zero. The F4S filename contains `daily-dual-v4`, but its admitted
report is `mode=reference`, `variant=field_original`; no failed vmap candidate,
precision candidate, normalized-key, gather or checkpoint candidate is used.
Source checkout is immutable
`c599236bcf57b745e9bafed678298010f4c3b58f`.

The workload is daily32 as eight distinct micro4 slices, plus one T7 and T30,
full 64,884,481-parameter/300-leaf native AD, and the declared AdamW. The daily
state output has `[8,4,3854]` coordinates; T7/T30 each return 3854. State
gradients are diagnostics and never optimizer parameters. Neural/physical
precision remains FP32/FP64. The public daily accumulator and complete-update
runner are reused without rebuilding losses.

## Strict result

Every comparison uses unchanged coordinate allowance
`1e-6 + 1e-5 * abs(reference)`. Bootstrap and update1 use identical update0
parameters and optimizer in both arms. Later pairs follow separate changing
parameter/optimizer trajectories; their input equality is not assumed.

| Phase | Daily state failures / 123328 | T7 state failures / 3854 | T30 state failures / 3854 | Combined parameter-gradient failures / 64884481 | Updated parameters / AdamW |
|---|---:|---:|---:|---:|---|
| Bootstrap | 52 | 1 | 1 | 0 | pass / pass |
| Update1 | 60 | 1 | 0 | 0 | pass / pass |
| Update2 | 185 | 2 | 7 | 0 | pass / pass |
| Update3 | 143 | 148 | 2 | 6942 | pass / pass |

All auxiliary comparisons pass. Update scalars pass through update2 and fail
one coordinate at update3. The final combined-gradient maximum absolute
difference is `2.298504114151001e-5`, relative L2
`3.612566395552382e-5`; all 6942 failures remain rejected despite passing updated
parameters/AdamW.

Example actual state failures:

- Bootstrap daily index `[0,0,1882]`: tree `-39.178045838025405`, leaf
  `-39.17386195546312`, difference `.004183882562287522`, allowance
  `.00039273861955463124`.
- Update1 T7 coordinate33: tree `.3685086667537689`, leaf
  `.3686502277851105`, difference `.00014156103134155273`, allowance
  `.000004686502277851105`.
- Bootstrap T30 coordinate33: difference `.000028133392333984375`, allowance
  `.000027614103317260746`.

The same-weight bootstrap-to-update1 replay also fails strict state gates:
daily51 coordinates in the leaf arm and62 in tree; T7 coordinates33/34 in both
arms; T30 passes in both. T7 leaf coordinate33 drifts
`6.681680679321289e-5` against allowance `4.685834109783173e-6`; tree drifts
`8.207559585571289e-5` against `4.685907423496247e-6`. This establishes strict
drift during repeated use of the identical saved binary. It does not establish
a causal explanation for the leaf/tree differences or admit them. Independent
JVP diagnosis remains a separate owner's evidence and was not rerun here.

Complete failed-coordinate indices, both values, differences and allowances
for every phase and both self replays are retained in the execution JSON.
Their state failure counts and complete index lists were independently
recomputed from the eight actual downloaded state artifacts.

## Physical, mask and resource gates

All outputs and proposals are finite, all hard counts are zero, and measured
peak GPU memory is `14,310,737,664` bytes (13.328GiB), below24GiB.

The separate host-only state-mask analysis passes all24 component/phase/arm
rows, using the same prepared bundle and canonical `_defined` rule:
`isfinite(initial) & (abs(initial) < 0.5e20)`.

| Component | Defined initial coordinates | Undefined initial coordinates | Undefined-gradient exact zero |
|---|---:|---:|---|
| Daily32 | 122670 | 658 | pass in every phase/arm |
| T7 | 3833 | 21 | pass in every phase/arm |
| T30 | 3833 | 21 | pass in every phase/arm |

Fortran/source masks remain part of the prepared inputs: all leaves, including
boolean masks, are dtype/shape/byte equal between the eight leaf/tree slices
at starts0,4,...28. Byte equality correctly preserves masked NaN placeholders.
Both recurrent arms use the same prepared case objects and executable objects;
there is no recurrent slicing or mask transformation. The offline check adds
no model evaluation or compilation.

## Runtime interpretation

| Update | Leaf complete seconds | Tree complete seconds |
|---|---:|---:|
| 1 | 12.752980 | 11.639716 |
| 2 | 12.624451 | 11.690882 |
| 3 | 12.669365 | 11.662331 |
| Median | 12.669365 | 11.662331 |

The observed leaf/tree median ratio is1.086349x. It is a rejected diagnostic
timing, not accepted same-workload throughput evidence, because the complete
strict gate failed. The historical parameter-only1.097337x result is not
upgraded to full-state acceptance.

Actual deserialization times are daily13.849205s, T7 94.860325s and T30
95.556448s; corresponding pickle reads are .066723/.064060/.111568s. These
startup costs are distinct from model compilation and warm updates. Helper
bootstrap arms take49.656291/18.048499s; bootstrap comparison/preservation is
complete at300.438415s. The full successful-execution process takes400.042682s
within its1200s budget; launcher407.251758s equals .113125488free card-hours.

The node is `ibc13b02n04`, physical GPU1, with OMP4/OpenBLAS1 and initial CPU
affinity0-111. Launch load is .39/.77/.67; both GPUs were empty immediately
before this attempt. Original compiler flags are retained; no FFN precision
`operand_upcaster` flag is used.

Earlier unique attempts are preserved: the first stopped on masked-NaN slice
equality, v2 on a cached Python package attribute selecting the pre-overlay
learning module, and v3 was interrupted during loading after a failed-index
JSON defect was identified. The repaired v4 checks actual overlay signatures
before loads and uses ordinary Python integers in failure evidence. These are
runner/startup failures, not numerical conclusions. Four launchers total
887.451561s, .246514323free card-hours, cash0.

## Artifacts

Local family: `outputs/diagnostics/shared-original-dual-slice-update-20261002/`.
Main files are `shared-original-dual-slice-update-20261002-v4.json`,
`result-summary.json`, `shared-original-dual-slice-update-20261002-v4.state-mask-analysis.json`,
`artifact-validation.json`, all eight `.states.pkl` files, launcher/log records,
and `executed-snapshot-v4/`. The independent size manifest identifies eight
complete raw proposals (total8,314,036,560 bytes) retained under
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/`; none is a
training checkpoint. Every expected remote proposal exists with a positive
recorded size, and every downloaded state file matches its remote size.

Execution sources are `scripts/hpc/probe_shared_original_dual_slice_update.py`,
`scripts/hpc/run_shared_original_dual_slice_update.sh`, and host analysis
`scripts/hpc/analyze_shared_original_dual_slice_states.py`. The five current
research entry documents are maintained by the lead.
