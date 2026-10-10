# Current Work Order

Updated: 2026-10-10. First-day localization completed both graphs/24 JVPs.
Finite, defined next-state tangents differ for every direction, including control 200;
raw state coordinate 20 is nonfinite. Numerical exit 3, GPU released, scope
closed. Raw-neural v3-final saved-evidence audit is accepted with active numerical
failures; 24 results saved/GPU released/scope closed. Handoff complete; stop.

## Main task and final acceptance

Improve T1/daily, T7 and T30 effective training throughput before formal
training. The10x target is cumulative research. Final acceptance uses changing
weights, the same GPU count and daily32/eight distinct micro4 plus one T7 and
one T30: public losses, complete parameter/initial-state gradients, physics,
masks and AdamW. State derivatives do not enter AdamW but remain a gate.
Keep strict `rtol=1e-5, atol=1e-6`, finite values and24GiB. No automatic
candidate adoption, reference/tolerance change or formal training is authorized.

## Effective evidence

| Evidence | Result and limit |
|---|---|
| [Forward/AD timing](research/daily_coarse_graining/shared_forward_reverse_results_20260930.md) | Matched medians: forward2.104821s/18.57%, additional gradients/recompute9.228212s/81.43%; AdamW0.02–0.03s. Pure reverse and exact whole-update partition remain unmeasured. |
| [Original complete-update scheduling](research/daily_coarse_graining/shared_original_full_state_schedule_followup_results_20261005.md) | Same two cards9.810149→7.647413s/+28.28%; state gates fail. Specific common-leaf-mass cancellation repair is independently closed. No complete admission. |
| [Selected latent components and T7 B4](research/daily_coarse_graining/shared_latent_reader_results_20261004.md) | Corrected T7/T30 component throughput improves about28%. Same742/275/482/299 T7 inputs: serial5.025831→B4 1.967053s/2.555005x. Loss/all parameters/aux/physics/masks/memory pass; states fail9/9/11. Corrected daily/full updates remain unmeasured. |
| [Saved-B1 synchronization](research/daily_coarse_graining/T7_saved_B1_dispatch_results_20261009.md) | 1.024075x over three pairs, state cross/self fails; separate scope closed. Keep helper for later whole-workload use, without polishing this small gain now. |
| [Saved-state/input analysis](research/daily_coarse_graining/T7_state_repeats_analysis_20261009.md) | Three sample299 B4/B1 ranges stay separated beyond observed B1 variation. Inputs FP64/normalization FP32; selected masks valid/no DOC tail. None of four selected coordinates lies on an initial field/axis extrema tie; later recurrent ties and derivative contributions remain unproved. |
| [Current JVP completion](research/daily_coarse_graining/T7_B1_state_JVP_completion_results_20261009.md) | 32 B1/B4 JVPs/four full VJPs complete; cycles repeat exactly. B1/4–B4 fails at 434/2382, passes 2621/control 200; own JVP/VJP and full-state gates fail. Full parameter self repeats, primal/aux/physics/memory pass. Reverse repeat variability alone does not describe the whole discrepancy; cause and derivative reference remain unresolved. |
| [First-day localization](research/daily_coarse_graining/T7_first_day_JVP_localization_results_20261010.md) | Next-state tangents fail 200/16/14/16 coordinates for input 434/2382/2621/control 200 in every cycle. Finite primals/observations and exact auxiliaries pass; raw state 20 nonfinite remains a separate failure. Differences occur before later recurrence/loss; no cause or derivative reference/adoption inferred. |
| [Accepted raw-neural result](research/daily_coarse_graining/T7_raw_neural_JVP_results_20261010.md) | Active tangent coordinates fail 1769/10/36/21 for 434/2382/2621/control200; four active primal coordinates fail. All 161 masks exact; 16 nontrivial repeats exact and 12 B4 isolation checks pass. Numerical gate failed; scope closed/released. |

No complete-update optimization has passed every gate. Fixed-weight four-T7
batching is a component screen, not a changed formal sampling mix. T30 trace
and source MAC shares do not isolate pure AD/module time or a launch bottleneck.
V4/V7 direct substitution stays stopped; T30 chunk-vmap and decoder128 remain
unselected. A precision/reference change is not selected by the JVP result.

## Current action and next gate

The [accepted result](research/daily_coarse_graining/T7_raw_neural_JVP_results_20261010.md)
reproduces active differences in the separately compiled canonical neural prefix,
before physical transforms/update. First differing operation, integrated-graph
cause and B1 derivative truth remain unproved. Batch-dependent FP32 contractions
are a source/HLO hypothesis; native JAX mapped scalar execution is a future
discussion candidate with correctness/speed unmeasured. No repair is selected.
Keep strict gates, unchanged precision/weights, complete fields/masks and AD.
The one-V100 scope closed at2169.713308/5400 seconds/0.602698 card-hours/CNY0,
exit3; two graphs/24 JVPs, zero VJPs/updates. The saved evidence and
[handoff](research/daily_coarse_graining/daily_speed_handoff_20261010.md) are complete.
Stop as explicitly directed:
“不自动开启下一轮 GPU 诊断或长训练，不自行移动目录/切根分支/清理 runtime。”
No automatic next graph, scope question, precision/reference choice or adoption.

## Completed scope and assets

The [first-day scope](research/daily_coarse_graining/T7_first_day_JVP_localization_results_20261010.md)
closed at 2124.572662062943/5400 GPU seconds/0.5901590727952619 card-hours/CNY 0,
actual exit 3, GPU released. Two graphs compiled once each, 24 JVPs,
zero VJPs/updates; no retry. Unused 3275.427337937057 seconds grant no new run.
Independent NumPy analysis and seven target guards preserve failures; no
full-T7 loss/parameter VJP/fourteen-hard-gate or update-speed acceptance.
The [original/B4 recovery](research/daily_coarse_graining/T7_state_JVP_results_20261009.md)
closed at 2809.490106/3600 free GPU seconds/CNY 0. The separately approved
[B1 completion](research/daily_coarse_graining/T7_B1_state_JVP_completion_proposal_20261009.md)
closed at 1780.7436698693782/3600 seconds/0.4946510194081606 card-hours/CNY 0,
zero updates. Numeric status is `numerical_failed`, ledger child exit 3;
[exit/release receipt](../outputs/diagnostics/shared-latent-reader-20261004/T7-B1-state-jvp-completion-20261009.exit-release.json)
confirms GPU 1 at 0 MiB/0%, no compute processes and completed PIDs absent. Independent
test01 NumPy analysis completed in 4.87128541059792s, exit 0, six guards passed.
Exact measurements/commands/receipts and native/scaled comparisons are in the
[completion report](research/daily_coarse_graining/T7_B1_state_JVP_completion_results_20261009.md).

Preserve both successful JVP graphs, saved VJPs and full arrays remotely under
`runtime/outputs/diagnostics/shared-latent-reader-20261004/`. Compact analysis
and CPU receipts are local under the same project-relative output directory.
Keep the accepted source archive and false/true approval records. Completed
commands must not be rerun; no Teacher regeneration.

All earlier windows and unused balances remain closed/separately accounted.
New paid work requires approved resources, time, directory, command and
worst-case CPU+GPU cost. Architecture/capacity, precision/reference, loss,
sampling, tolerance or recurrence choices need an explicit scientific decision.
Work only in jcwu's remote tree; never compute on login nodes.
