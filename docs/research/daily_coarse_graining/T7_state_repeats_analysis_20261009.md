# Preserved T7 state-gradient repeat analysis — 2026-10-09

CPU-only extraction of 21 already saved full initial-state gradients confirms that the paid B4/B1 separation at sample 3 coordinates 434, 2382 and 2621 persists after including cold evaluations. All three paired differences at each coordinate exceed both the original strict allowance and the same-session B1 observed range width. This is an empirical repeat comparison; it does not identify cause, harmlessness, or a scientifically accepted reference.

## Evidence and unchanged gates

The extraction ran on test01 using the canonical CPU Python and plain NumPy/pickle. It performed no JAX import, model call, evaluation, compilation, optimizer update, GPU use or scheduler submission. Each native array is finite shape (4, 3854), already scaled by the four-sample mean; no additional scaling was applied. Only the extracted 2,012,580-byte NPZ and metadata JSON were transferred locally; the analysis JSON was produced locally and full parameter-gradient pickles remain remote.

The strict coordinate test remains `abs(B4-B1) <= 1e-6 + 1e-5*abs(B1)`. Full-state relative L2 and cosine are descriptive and cannot replace this gate. Metadata records source report/artifact paths, array hashes, native dtypes, losses and case identities. Resource host is not inferred when absent from the source report metadata; the separately preserved job receipts identify paid node ibc13b03n01 and later dispatch node ibc13b02n04.

| Session | Modes | Saved states | Official state failure counts |
| --- | --- | ---: | --- |
| paid_B4, same paid host/session | serial B1 and batch B4 | 8 | paired cross 9/9/11; B1 self 4/1/3; B4 self 4/3/3 |
| dispatch_B1, later separate host/session | block_each and dispatch_four B1 | 8 | paired cross 1/1/2; block self 2/3/1; dispatch self 2/3/3 |
| free_B1_recovery, older free session | serial B1 | 4 | self 1/5/4 |
| free_B1_compile, older free session | serial B1 | 1 | no repeat estimate |

These counts were recomputed from the extracted arrays and match the saved reports. Self comparisons use each mode's own cold-first/warmup result as reference. Every within-mode repeat pair in both directions also contributes to a descriptive affected-coordinate union: 14 coordinates. Those extra comparisons enlarge the observed envelope only; they do not replace official comparisons or pool sessions.

## Priority coordinates

Values below use all four paid observations per mode (cold-first plus pairs 1–3). Dispersion is population standard deviation of these four observed values, not an estimate of a population bound.

| Sample:coordinate | Saved field binding | B1 minimum–maximum; std | B4 minimum–maximum; std | Observed separation | Gap / B1 width |
| --- | --- | --- | --- | --- | ---: |
| 3:434 | `hydrol_previous_step_state.profil_froz_hydro_ns[0, 5, 3]` | 1.536172433e+00–1.536192311e+00; 7.795222023e-06 | 1.536335628e+00–1.536380212e+00; 2.134377264e-05 | actual_above, gap 1.433172729e-04 | 7.2098 |
| 3:2382 | `slowproc_stomate_previous_step_state.carbon_32l[0, 1, 1, 0]` | 2.871503498e+00–2.871527668e+00; 9.226773529e-06 | 2.871398311e+00–2.871423702e+00; 1.033788651e-05 | actual_below, gap 7.979577993e-05 | 3.3015 |
| 3:2621 | `slowproc_stomate_previous_step_state.deepC_peat[0, 3, 1]` | 3.677237779e-03–3.678951412e-03; 6.533929800e-07 | 3.684125841e-03–3.686740994e-03; 9.846682920e-07 | actual_above, gap 5.174428225e-06 | 3.0196 |

| Sample:coordinate / pair | Signed B4 − B1 | Strict allowance | Absolute difference / allowance | Absolute difference / paid B1 width |
| --- | ---: | ---: | ---: | ---: |
| 3:434 / 1 | 1.906284597e-04 | 1.636189584e-05 | 11.6508 | 9.5898 |
| 3:434 / 2 | 1.433172729e-04 | 1.636192311e-05 | 8.7592 | 7.2098 |
| 3:434 / 3 | 1.566016581e-04 | 1.636180978e-05 | 9.5712 | 7.8781 |
| 3:2382 / 1 | -8.399795974e-05 | 2.971507700e-05 | 2.8268 | 3.4753 |
| 3:2382 / 2 | -1.207441534e-04 | 2.971527668e-05 | 4.0634 | 4.9957 |
| 3:2382 / 3 | -9.517382457e-05 | 2.971516015e-05 | 3.2029 | 3.9377 |
| 3:2621 / 1 | 6.888061762e-06 | 1.036772378e-06 | 6.6438 | 4.0196 |
| 3:2621 / 2 | 5.681067705e-06 | 1.036789514e-06 | 5.4795 | 3.3152 |
| 3:2621 / 3 | 7.551163435e-06 | 1.036778189e-06 | 7.2833 | 4.4065 |

All nine rows strictly fail. The three B4 paired ranges also remain separated from the eight later dispatch B1 observations. These cross-session comparisons are kept descriptive:

| Coordinate | Later dispatch B1 minimum–maximum; width | B4 pair-range separation / gap | Older recovery B1 minimum–maximum; width |
| --- | --- | --- | --- |
| 434 | 1.536192348e+00–1.536251535e+00; 5.918741226e-05 | actual_above, 8.409260772e-05 | 1.536125919e+00–1.536234719e+00; 1.088008285e-04 |
| 2382 | 2.871501502e+00–2.871524807e+00; 2.330544216e-05 | actual_below, 7.779901899e-05 | 2.871508773e+00–2.871528055e+00; 1.928218377e-05 |
| 2621 | 3.677804023e-03–3.680352122e-03; 2.548098564e-06 | actual_above, 3.773719072e-06 | 3.678072244e-03–3.679212183e-03; 1.139938831e-06 |

## All affected coordinates

The complete 14-coordinate union includes coordinates affected only by self-repeat comparisons. All rows retain their actual three paired signed differences and allowances; a separated observed range does not itself imply a strict failure. Detailed per-session values, means, extrema, dispersion and repeat labels for every coordinate are in the analysis JSON.

| Sample:coordinate | B1 min–max; std | B4 min–max; std | All-paid range relation; gap | B4−B1 pairs 1/2/3 | Strict allowances 1/2/3 | Failed pairs | Differences exceeding B1 width |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0:182 | 1.285432726e-01–1.285441816e-01; 4.163501501e-07 | 1.285410225e-01–1.285421550e-01; 4.461025411e-07 | actual_below; 1.117587090e-06 | -2.726912498e-06; -2.026557922e-06; -2.384185791e-06 | 2.285441518e-06; 2.285441816e-06; 2.285434067e-06 | 1/3 | 1/2/3 |
| 0:508 | 4.249595702e-01–4.249615967e-01; 9.239621176e-07 | 4.249632061e-01–4.249670208e-01; 1.481373292e-06 | actual_above; 1.609325409e-06 | 2.205371857e-06; 5.334615707e-06; 5.424022675e-06 | 5.249615371e-06; 5.249598980e-06; 5.249615967e-06 | 2/3 | 1/2/3 |
| 0:509 | 1.241954923e+00–1.241959929e+00; 1.898714356e-06 | 1.241961837e+00–1.241973758e+00; 4.401874647e-06 | actual_above; 1.907348633e-06 | 1.490116119e-05; 1.442432404e-05; 8.702278137e-06 | 1.341958857e-05; 1.341954923e-05; 1.341957068e-05 | 1/2 | 1/2/3 |
| 1:434 | -1.155866326e+00–-1.155860787e+00; 1.983883939e-06 | -1.155887748e+00–-1.155882444e+00; 2.054147981e-06 | actual_below; 1.611793414e-05 | -2.430658787e-05; -1.981901005e-05; -2.165650949e-05 | 1.255863442e-05; 1.255866326e-05; 1.255860787e-05 | 1/2/3 | 1/2/3 |
| 1:509 | 2.236520499e-01–2.236632705e-01; 4.028146598e-06 | 2.236622125e-01–2.236756682e-01; 4.956142790e-06 | overlap; 0.000000000e+00 | 1.239776611e-05; 8.985400200e-06; 1.016259193e-05 | 3.236632705e-06; 3.236568481e-06; 3.236520499e-06 | 1/2/3 | 1 |
| 2:505 | 1.010878943e-02–1.010995824e-02; 4.276943658e-07 | 1.010940317e-02–1.011039130e-02; 3.702885004e-07 | overlap; 0.000000000e+00 | -3.445893526e-07; 8.754432201e-08; 4.936009645e-08 | 1.101099582e-06; 1.101096211e-06; 1.101093538e-06 | none | none |
| 2:514 | -1.827947283e+01–-1.827917910e+01; 1.128770347e-04 | -1.827965283e+01–-1.827950931e+01; 5.141335103e-05 | actual_below; 3.647804260e-05 | -2.863407135e-04; -1.657009125e-04; -3.302097321e-04 | 1.837936649e-04; 1.837943420e-04; 1.837917910e-04 | 1/3 | 3 |
| 2:1882 | 3.409319271e+01–3.409343304e+01; 8.552459389e-05 | 3.409301882e+01–3.409314436e+01; 5.166458844e-05 | actual_below; 4.835472897e-05 | -1.778237678e-04; -1.871706579e-04; -1.391902871e-04 | 3.419330509e-04; 3.419333153e-04; 3.419319271e-04 | none | none |
| 3:33 | 1.203081489e+00–1.203101754e+00; 7.518695369e-06 | 1.203147531e+00–1.203169346e+00; 8.370739735e-06 | actual_above; 4.577636719e-05 | 5.173683167e-05; 6.759166718e-05; 8.368492126e-05 | 1.303095794e-05; 1.303101754e-05; 1.303081489e-05 | 1/2/3 | 1/2/3 |
| 3:182 | 6.009466480e-03–6.010862067e-03; 5.056212544e-07 | 6.009291857e-03–6.011232734e-03; 7.900042210e-07 | overlap; 0.000000000e+00 | -1.024920493e-06; 3.706663847e-07; 1.701060683e-06 | 1.060103168e-06; 1.060108621e-06; 1.060094665e-06 | 3 | 3 |
| 3:434 | 1.536172433e+00–1.536192311e+00; 7.795222023e-06 | 1.536335628e+00–1.536380212e+00; 2.134377264e-05 | actual_above; 1.433172729e-04 | 1.906284597e-04; 1.433172729e-04; 1.566016581e-04 | 1.636189584e-05; 1.636192311e-05; 1.636180978e-05 | 1/2/3 | 1/2/3 |
| 3:515 | -2.898963451e+01–-2.898949909e+01; 4.997169669e-05 | -2.898930144e+01–-2.898912358e+01; 6.357902040e-05 | actual_above; 1.976490021e-04 | 2.305507660e-04; 4.107952118e-04; 3.161430359e-04 | 2.908953199e-04; 2.908963451e-04; 2.908955131e-04 | 2/3 | 1/2/3 |
| 3:2382 | 2.871503498e+00–2.871527668e+00; 9.226773529e-06 | 2.871398311e+00–2.871423702e+00; 1.033788651e-05 | actual_below; 7.979577993e-05 | -8.399795974e-05; -1.207441534e-04; -9.517382457e-05 | 2.971507700e-05; 2.971527668e-05; 2.971516015e-05 | 1/2/3 | 1/2/3 |
| 3:2621 | 3.677237779e-03–3.678951412e-03; 6.533929800e-07 | 3.684125841e-03–3.686740994e-03; 9.846682920e-07 | actual_above; 5.174428225e-06 | 6.888061762e-06; 5.681067705e-06; 7.551163435e-06 | 1.036772378e-06; 1.036789514e-06; 1.036778189e-06 | 1/2/3 | 1/2/3 |

## Full-state descriptive comparisons

| Session / modes / pair | Strict failed / 15,416 | Relative L2 to B1 | Cosine |
| --- | ---: | ---: | ---: |
| paid_B4 / batch vs serial / pair1 | 9 | 3.854810826e-06 | 0.99999999999466616 |
| paid_B4 / batch vs serial / pair2 | 9 | 3.932661718e-06 | 0.99999999999510525 |
| paid_B4 / batch vs serial / pair3 | 11 | 4.291136600e-06 | 0.99999999999256928 |
| dispatch_B1 / dispatch_four vs block_each / pair1 | 1 | 6.552057699e-07 | 0.99999999999992895 |
| dispatch_B1 / dispatch_four vs block_each / pair2 | 1 | 2.029244440e-07 | 0.99999999999998723 |
| dispatch_B1 / dispatch_four vs block_each / pair3 | 2 | 4.488195874e-07 | 0.99999999999990163 |

## Interpretation and informative seam

Observed B1 repeat variability is real and sometimes fails the existing strict gate. It does not explain away these three B4/B1 differences: each is larger than the entire same-paid B1 range and the B4/B1 ranges stay separated with cold values included. Four observations per mode cannot bound future variability or assign a causal mechanism. Later/older B1 sessions provide distinct descriptive corroboration, not an interchangeable same-host control.

Sample 3:434 is the most informative first coordinate for localization because it has the largest strict-allowance ratios, the largest gap relative to same-paid B1 width, consistent signed separation, and a concrete saved hydrology field binding (`profil_froz_hydro_ns[0,5,3]`). Current source reviewed by the parent shows potential mechanisms involving historical FP32 field/expert summaries and common normalization: `strong_baseline_architecture::_masked_summary` (957–973), `assemble_strong_baseline_context` (1111–1128, 1167–1172), `shared_transformer::_normalized_local_state` (800–826), and `shared_latent_reader::_prepare_field` (205–230). The exact metadata extension below establishes array memberships, mask and scale; the bounded tie extension resolves initial-input extrema/tie membership. Multiple derivative paths/fan-in remain potential mechanisms, not demonstrated causes or measured coordinate-specific gradient contributions. This CPU analysis does not execute or approve a new GPU localization experiment.

The named prior original-state FD evidence used 59 saved-binary calls and right-sided slopes at zero physical boundaries without a stable slope platform; quantization, nonlinearity and branches remained unresolved. Prior FP32 original-state JVP/VJP discrepancies and FP64 internal consistency are different architecture/artifact evidence; FP64 changes forward values and is not an adopted reference. Neither adjudicates candidate 299 here. Repeating FD is not the default next route, and no precision, science, tolerance or reference selection is made.

## Reproduction and verification

Remote extraction (test01; all paths under `/WORK/liwei_work/jcwu/`):

```text
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= <canonical CPU Python> -B <staged analysis script> extract
  --report paid_B4 /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-batch-20261009-paid.replacement.compile-and-compare.json
  --report dispatch_B1 /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-dispatch-20261009.saved-dispatch.json
  --report free_B1_recovery /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-batch-20261008.serial-saved.json
  --report free_B1_compile /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-batch-20261008.compile-and-compare.json
  --output-prefix /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-state-repeats-20261009
```

Extraction took 14.257s. Canonical CPU Python: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python`; staged script: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-state-repeats-20261009/analyze_shared_latent_T7_state_repeats.py`.

Local analysis and lightweight tests:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
conda run -n ORCJAX python -B scripts/hpc/analyze_shared_latent_T7_state_repeats.py analyze --input-prefix outputs/diagnostics/shared-latent-reader-20261004/T7-state-repeats-20261009 --output outputs/diagnostics/shared-latent-reader-20261004/T7-state-repeats-20261009.analysis.json
conda run -n ORCJAX python -B -m pytest -q -p no:cacheprovider tests/unit/test_shared_latent_T7_state_repeats.py
```

Four CPU tests passed: observed overlap/direction, dispersion, scale-safe whole-state metrics retaining strict failures, and paid/session separation. The only warning was the existing pytest `cache_dir` setting with cache provider disabled. The actual extracted evidence reproduces every official state failure count listed above.

Artifacts: `scripts/hpc/analyze_shared_latent_T7_state_repeats.py`, `tests/unit/test_shared_latent_T7_state_repeats.py`, and `outputs/diagnostics/shared-latent-reader-20261004/T7-state-repeats-20261009.{npz,metadata.json,analysis.json}`. The repeat extraction uses saved failure-record bindings; the separate exact-input metadata extension below reads the actual saved head layout and prepared inputs.

## Exact sample 3 input/head metadata extension

The separate CPU extension reads prepared artifact 299 at `continuous_state[window0,day0]` and the exact prepared shared head. Prepared-artifact, context and staged neural-module hashes match the paid report identities. The config-only candidate head replacement matches the saved candidate configuration and preserves the same input-normalization, layout and carbon-layout objects. No candidate parameters were loaded. Canonical serialized class/module identities were retained through the known exact overlay bootstrap.

All four coordinates have canonical defined mask true, source-summary defined mask true after the inactive-PFT mask, and continuous tail mask false (configured threshold 16). Both active PFT mask entries are true. The normalized values use NumPy FP32 casts, subtraction and division in the historical arithmetic sequence; no neural forward or JAX numerical call was computed.

| Coordinate | Raw initial value | Continuous mean | Continuous scale | Historical FP32 z | Field-pool row / family | Expert-axis memberships | Local field mappings | Carbon feedback / PFT-state |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 434 | 0.000000000e+00 | 5.807185880e-11 | 1.702165729e-07 | -3.411645594e-04 | 61 / state_water | 18 entries; experts [0, 1, 2, 3, 5, 6]; landpoint[0], soil_layer[5], soil_tile[3] | `endpoint.fast_state.hydrol_previous_step_state.profil_froz_hydro_ns` scalar 33 | none; no PFT-state positions |
| 2382 | 3.224057494e-08 | 3.001904076e-07 | 4.424219757e-07 | -6.056431532e-01 | 117 / state_carbon | 20 entries; experts [0, 1, 4, 5, 6]; carbon_pool[1], deep_layer[0], landpoint[0], pft[1] | `inventory.carbon.carbon_32l.outgoing_fraction` scalar 96; `inventory.carbon.carbon_32l.incoming_amount` scalar 96 | position 1055, PFT owner 1; no PFT-state positions |
| 2621 | 6.329964314e-07 | 4.281807523e-06 | 6.206373200e-06 | -5.879135728e-01 | 123 / state_carbon | 15 entries; experts [0, 1, 4, 5, 6]; deep_layer[3], landpoint[0], pft[1] | none | position 1294, PFT owner 1; no PFT-state positions |
| 200 (control) | 0.000000000e+00 | 1.459349358e-05 | 3.509934243e-04 | -4.157768562e-02 | 45 / state_water | 12 entries; experts [0, 1, 2, 3, 5, 6]; landpoint[0], pft[1] | `endpoint.fast_state.hydrol_previous_step_state.canopy2ground` scalar 1 | none; no PFT-state positions |

Coordinate 434 therefore has actual field/axis/local membership in the saved layout and no carbon-feedback membership. Coordinate 2382 has field/axis, two local inventory mappings and carbon-feedback membership. Coordinate 2621 has field/axis and carbon-feedback membership with no local field mapping. These array memberships establish eligibility under the sampled masks; they do not establish individual gradient contributions or causality.

Final extraction ran on `ibtest01` in 4.320710s. The bounded schema-only inspection before it took 39.436775s. No model, JAX numerical call, tracing, compilation, GPU work or optimizer update ran. No tests were rerun for this data-only extension. Artifact: `outputs/diagnostics/shared-latent-reader-20261004/T7-state-inputs-20261009.json`; script: `outputs/diagnostics/shared-latent-reader-20261004/extract_T7_state_inputs_20261009.py`.

```text
# Invoked via SSH on test01, through the cln02 endpoint override:
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python -B /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-state-repeats-20261009/extract_T7_state_inputs_20261009.py
```

This extension used supplied known asset paths and exact source fallback because graph transport was unavailable. The saved case hash is recorded from the paid report and was not recomputed through a JAX case reconstruction. Derivative contributions remain unresolved; initial-input extrema/ties are addressed by the following bounded extension.

## Initial-input masked summary extrema/ties

The source arithmetic evaluated on the exact prepared window0/day0 inputs contains tied pool extrema, but none of coordinates 434, 2382, 2621 or control 200 is on a tied extremum. Coordinates 2621 and 200 each are a unique field minimum; 434 and 2382 are neither field minimum nor maximum. None is either extremum in any of its 65 total expert-axis memberships. Every requested coordinate remains selected after pool membership, canonical defined and inactive-PFT masks.

| Coordinate | Field selected count | FP32 minimum / tie count | FP32 maximum / tie count | Coordinate at field extremum | Axis rows / rows with tied minimum / rows with tied maximum | Coordinate at axis extrema |
| --- | ---: | --- | --- | --- | --- | --- |
| 434 | 66 | -7.984456606e-03 / 1 | 0.000000000e+00 / 60 | neither | 18 / 2 / 1 | neither in all rows |
| 2382 | 192 | -7.737739682e-01 / 1 | 0.000000000e+00 / 159 | neither | 20 / 8 / 5 | neither in all rows |
| 2621 | 64 | -5.879135728e-01 / 1 | 0.000000000e+00 / 53 | unique minimum | 15 / 8 / 0 | neither in all rows |
| 200 | 2 | -4.157768562e-02 / 1 | 0.000000000e+00 / 1 | unique minimum | 12 / 6 / 2 | neither in all rows |

Source `_masked_summary` (strong_baseline_architecture.py:957–973) casts values to FP32 and selects `pool & defined`; source-defined masks additionally apply structural/PFT validity via `_mask_inactive_pft_elements` (984–990). Excluded entries use +infinity for `jnp.min` and -infinity for `jnp.max`. Ties here are counts of selected values exactly equal to the FP32 extremum, with no approximate tie tolerance. The actual helper returns mean, finite-zero-tangent square root of variance, minimum and maximum; only normalized inputs and extrema/ties were computed here, without mean/variance, neural projections or derivatives.

For example, the 434 field maximum 0 has 60 tied entries while 434 itself is a small negative normalized value. Carbon-pool axis rows containing 2382 have maximum 0 tied across 55 entries while 2382 is negative. Thus neither raw zero nor membership in a pool that contains ties demonstrates that the coordinate itself lies on a reduction tie.

The artifact preserves each expert-axis pool's selected count, min/max values, exact tie counts, tied coordinate indices, requested-coordinate mask and extrema membership, plus selected DOC-tail counts. Non-DOC features use historical FP32 arithmetic; DOC uses the configured FP64 log tail before FP32 conversion. NumPy evaluates this source arithmetic without a JAX/GPU forward. DOC log1p can differ by platform rounding, so extrema involving selected DOC are CPU arithmetic evidence; this is not a claim of sampled GPU bitwise values. The initial-input finding does not exclude later recurrent extrema/ties or establish T7 derivative contribution magnitudes or cause.

Extraction ran on `ibtest01` in 40.727965s with no model/JAX numerical/tracing/GPU execution. It reused the exact metadata bootstrap and hashes. Artifact: `outputs/diagnostics/shared-latent-reader-20261004/T7-state-input-ties-20261009.json`; analyzer: `outputs/diagnostics/shared-latent-reader-20261004/analyze_T7_state_input_ties_20261009.py`. Invocation is the preceding test01 CPU command with `--ties`. No prior tests, Teacher generation or gradient graph were repeated.
