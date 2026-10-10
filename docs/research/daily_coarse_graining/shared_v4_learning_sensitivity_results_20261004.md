# V4/V7 bounded learning sensitivity — 2026-10-04

The approved three-arm, 16-update diagnostic finished normally. Candidate learning differs observably from both original trajectories at this fixed initialization and data. Under the common **original evaluator**, its joint loss is lower at step8 and higher at step16. At step16 candidate loss is **0.3907121399**, versus **0.3880653216 / 0.3880681301** for the two originals, differences **+0.0026468183 / +0.0026440098**. Original-repeat difference is **+2.8085427e-6**. These descriptive differences do not establish mathematical derivative error, universal learning damage, harmlessness, generalization or production acceptance. No scientific threshold was invented and reference drift does not waive strict parity.

Current disposition: retain the measured16.56–16.72% throughput improvement,
but stop direct default integration of the unchanged V4/V7 pair under the
existing strict gates. The learning observation gives no basis to dismiss
the numerical differences as harmless. Proceed to a different concrete,
function-preserving execution candidate under [NEXT_STEPS](../../NEXT_STEPS.md),
without extending this diagnostic or changing the scientific method.

## Actual execution and evidence scope

The actual exit is **0**, with **16 unique updates per arm / 48 total**, **580 started and completed native model calls**, monitors at **0/4/8/16**, and zero `compile_or_get_cached`/`backend_compile` model counts. The GPU family took **1192.757057 s / 0.331321404800 card-hours**, one free V100 GPU1, CNY0, under the approved1800s cumulative window. Parent execution owner confirmed GPU1 was released. Peak recorded active GPU memory is **14.114414 GiB**, below the unchanged24GiB ceiling. No failed raw files were found.

All arms start from the same saved pair1-reference parameters and AdamW (step2), artifact742, immutable source HEAD `c599236bcf57b745e9bafed678298010f4c3b58f`. Candidate uses V4 daily and V7 T7 with original T30. The fixed daily32/eight distinct micro4, T7/T30 losses, neural FP32/physical FP64, clipping/AdamW, sampling and full-network scope remain the declared diagnostic. This is an exploratory saved initialization, not the final fresh-initialization training recipe.

Raw preservation is exact: **12 full snapshots** at1/4/8/16 contain parameters, both moments, mixed gradients, state derivatives, losses/auxiliaries and memory; **ten monitors** contain original-evaluator loss/auxiliary/state/physics/memory records; **48 update rows** contain loss/norm/clip/component timing metadata. Full raw parameter gradients are not retained for every one of the48 updates or580 calls. Source hard gates run on each call; independent CPU array validation covers the preserved snapshots/monitors, not unsaved per-call arrays.

## Common original-evaluator results

| Step | Original A | Candidate | Original B | C − A | C − B | B − A |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.6371876799 | 0.6371876799 | 0.6371876799 | 0 | 0 | 0 |
| 4 | 0.4609767661 | 0.4609769957 | 0.4609767003 | 2.29618547e-07 | 2.953379297e-07 | -6.571938277e-08 |
| 8 | 0.3725799912 | 0.3716865703 | 0.3725666246 | -0.0008934208845 | -0.0008800542849 | -1.336659953e-05 |
| 16 | 0.3880653216 | 0.3907121399 | 0.3880681301 | 0.002646818308 | 0.002644009765 | 2.808542666e-06 |

All arms improve joint loss from initialization by step16, but all have a higher joint score at16 than8. The candidate/original ordering changes between8 and16. This is not evidence of a monotonic candidate degradation curve or monotonic optimization.

The final component changes from the shared baseline are:

| Loss | Initial | Δ A | Δ C | Δ B | C − A | C − B | B − A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| joint | 0.6371876799 | -0.2491223583 | -0.24647554 | -0.2491195498 | 0.002646818308 | 0.002644009765 | 2.808542666e-06 |
| daily | 0.5846060272 | -0.2070259766 | -0.2043122834 | -0.2070246869 | 0.002713693202 | 0.002712403419 | 1.289782592e-06 |
| T7 | 0.0110281886 | -0.006588616208 | -0.006615382936 | -0.00658681197 | -2.676672794e-05 | -2.857096566e-05 | 1.804237723e-06 |
| T30 | 0.04155346412 | -0.03550776549 | -0.03554787366 | -0.03550805097 | -4.010816617e-05 | -3.982268852e-05 | -2.854776494e-07 |

The higher candidate final joint score is dominated by its higher **daily** score (+0.0027136932/+0.0027124034). Its T7 and T30 aggregate scores are lower than both originals. The public normalized component losses show mixed effects:

| Public normalized loss term | Initial | Δ A | Δ C | Δ B | C − A | C − B | B − A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T30/flux | 0.06077636299 | -0.0523951139 | -0.05244836597 | -0.05239549536 | -5.325207234e-05 | -5.287060864e-05 | -3.81463707e-07 |
| T30/stock | 0.3132048141 | -0.2671747755 | -0.267482497 | -0.2671769634 | -0.0003077214232 | -0.000305533588 | -2.187835138e-06 |
| T30/living_stock | 0.3014363897 | -0.2480811094 | -0.2483977166 | -0.2480833767 | -0.0003166072326 | -0.0003143399499 | -2.267282621e-06 |
| T30/nonliving_stock | 0.3249732385 | -0.2862684417 | -0.2865672773 | -0.2862705501 | -0.0002988356137 | -0.0002967272261 | -2.108387654e-06 |
| T7/flux | 0.006089800048 | -0.002126465816 | -0.002140391359 | -0.002124051436 | -1.392554296e-05 | -1.633992263e-05 | 2.414379679e-06 |
| T7/stock | 0.09316389739 | -0.05717108005 | -0.05739805506 | -0.05715725629 | -0.0002269750085 | -0.0002407987683 | 1.382375983e-05 |
| T7/living_stock | 0.0366884294 | -0.03238882802 | -0.03235215471 | -0.03237953867 | 3.667331357e-05 | 2.738396179e-05 | 9.289351777e-06 |
| T7/nonliving_stock | 0.1496393654 | -0.08195333208 | -0.08244395541 | -0.08193497392 | -0.0004906233305 | -0.0005089814984 | 1.835816788e-05 |
| daily/process_day | 1.443037635 | -0.5464294849 | -0.5379419347 | -0.5464215074 | 0.008487550166 | 0.008479572672 | 7.977494751e-06 |
| daily/flux | 0.181783233 | -0.03457213045 | -0.03469336406 | -0.03457364497 | -0.0001212336087 | -0.0001197190862 | -1.514522491e-06 |
| daily/stock | 0.750558106 | -0.1893732047 | -0.1902913827 | -0.1893840146 | -0.0009181780731 | -0.0009073681547 | -1.080991843e-05 |

In particular, candidate daily process-day loss is higher while daily flux/stock losses are lower. T7 living-stock loss is higher, while T7 aggregate stock/nonliving-stock and flux losses are lower. T30 flux/living/nonliving terms are lower. These are normalized **loss terms**, not physical flux amounts or field-resolved accuracy; they do not locate an individual GPP/canopy process error. `learning-effects.json` retains all48 exact descriptive rows at4/8/16, including every arm's change, candidate vs both references and reference-repeat drift.

## Strict first update and later trajectories

At the first common-input update, unchanged `rtol=1e-5`, `atol=1e-6` comparisons give candidate **978 parameter-gradient failures** and **287 state-gradient failures**. Actual updated parameters, full optimizer, loss/auxiliary, norm and clipping pass. Original B vs A parameter gradients pass, but **seven state coordinates fail**. This repeat variation is retained and does not relax the candidate contract. The earlier independent same-input experiment's original **977** gradient failures remain historical evidence; this fresh run's978 count does not replace them.

The CPU analyzer independently reproduces the retained first-step strict classification. It verifies300 numeric leaves /64,884,481 coordinates /two static `None` nodes **and paths, shapes, dtypes and `None` positions** against the common initial parameter ABI for every stored parameter/gradient/first/second-moment tree. Stored steps are3/6/10/18 as specified. All preserved finite, physical hard-counter, source-defined zero-derivative and24GiB memory checks pass. Original masks are reused from the unchanged declared input-state mask artifact; no Teacher/prepared-model input is read by the analysis.

Cumulative parameter movements below are saved weights minus the common step2 weights. Operands are cast to float64 **before subtraction**; no new optimizer update is reconstructed or executed.

| Step | ‖Δ A‖ | ‖Δ C‖ | ‖Δ B‖ | ‖C − A‖ | ‖C − B‖ | ‖B − A‖ | C − A / ‖Δ A‖ | Cosine Δ A, Δ C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.03823569639 | 0.03823547319 | 0.03823569635 | 4.479779579e-06 | 4.480316208e-06 | 1.621072475e-07 | 0.0001171622332 | 0.9999999932 |
| 4 | 0.1423456944 | 0.1423454571 | 0.1423456965 | 8.248255944e-06 | 8.256648566e-06 | 9.262253427e-07 | 5.794524365e-05 | 0.9999999983 |
| 8 | 0.2308126498 | 0.2307426898 | 0.2308106885 | 0.0008026278736 | 0.0007773737364 | 3.445411501e-05 | 0.003477399849 | 0.999993998 |
| 16 | 0.3700486242 | 0.3707509879 | 0.3700616649 | 0.002847572372 | 0.002781796146 | 9.081854661e-05 | 0.007695130275 | 0.9999722464 |

By step16 candidate/original cumulative-parameter relative differences are **0.769513% / 0.751712%**, whereas original-repeat relative difference is **0.0245423%**. Final direction angles are0.00745033 /0.00727591 rad for candidate against the originals and0.000242876 rad for original-repeat. Direction and norm metrics are descriptive; no accepted trajectory-error threshold follows from them.

The readout module results retain local evidence rather than hiding it beneath the full network norm:

| Parameter module | Pair | Reference cumulative norm | Cross-error norm | Relative cross-error | Cosine |
| --- | --- | --- | --- | --- | --- |
| readout_first | A → C | 0.01856774364 | 0.0001194288378 | 0.006432059819 | 0.9999832113 |
| readout_first | B → C | 0.01856836762 | 0.0001167601221 | 0.00628811991 | 0.999984031 |
| readout_first | A → B | 0.01856774364 | 3.724409281e-06 | 0.000200584915 | 0.9999999804 |
| readout_second | A → C | 0.0009604361346 | 5.788455525e-06 | 0.006026903108 | 0.9999874124 |
| readout_second | B → C | 0.0009604681462 | 5.676995159e-06 | 0.00591065428 | 0.9999879932 |
| readout_second | A → B | 0.0009604361346 | 1.794229984e-07 | 0.0001868140857 | 0.9999999831 |

At step16, the largest absolute cumulative coordinate difference A→C is parameter `shared_transformer/output_axis_position_embedding/3[2,300]`: A movement5.311658606e-5, C3.673485480e-5, difference1.638173126e-5 (30.8411% of that local A movement). Readout_first's largest coordinate difference is kernel[261,211],3.501772881e-6 (6.42515% of local movement); readout_second kernel[43,0],1.309439540e-6 (1.21270%). Indices identify parameter coordinates, not physical state labels. No scientific importance ranking or damage conclusion follows from local percentages. Per-module results against both references and original-repeat are in each `parameters-step-*.json`.

Stored gradients at4/8/16 are also quantified in `trajectory-gradients-step-*.json`. They are gradients at **different learned parameters**, so their differences are trajectory divergence, not same-input AD error. The source audit remains bounded: native AD/formula/mask preservation does not establish the unique compiled floating cause or an efficient repair.

## Runtime measurements and offline analysis

| Arm | First update wall (s) | Median updates 2–16 (s) | Count after first |
| --- | --- | --- | --- |
| reference_a | 36.82420805 | 12.94082512 | 15 |
| candidate | 11.92154359 | 11.08715558 | 15 |
| reference_b | 13.32640025 | 12.92269366 | 15 |

For steps2–16, observed original/candidate median wall ratios are **1.167191× / 1.165555×**. This is a measurement during divergent trajectories with unchanged workload, not a new same-weight speed comparison. The existing same-input evidence remains13.103010→11.186190s (**1.171356×**). It is not an order-of-magnitude result. First original-A update includes a17.9262s optimizer/helper timing; first calls are reported separately and not blended into steady update ratios.

Recorded update walls sum **617.507291s**; raw monitor walls sum **161.037207s**. The remaining **414.212560s** of family time combines loading/startup, snapshot and strict-host IO, setup and other unpartitioned overhead. Individual startup/loading and IO durations were not measured separately in this driver; no isolated startup saving is inferred. Update walls exclude snapshot/strict-host IO. Official per-component timings and all15 after-first update walls per arm remain in `summary.json`.

The NumPy-only CPU analysis ran on **ibtest02**, with **54.790295s** reductions/IO after imports, **55.162556s** successful process wall, peak RSS **3.337708GiB** and **exit0**. The cumulative240s CPU allowance was preserved across an ordinary inert-unpickler repair: total **126.727725s**. Attempt0 stopped before analysis because unused initial science records had three additional namedtuple classes; source-backed inert mappings repaired deserialization without importing model/JAX modules. Attempt sources/logs and `cpu-attempt-history.json` preserve both attempts. There were zero CPU-analysis model/GPU/AD/optimizer/Teacher calls.

Independent saved-raw monitor reconstruction matches the driver effects exactly; saved update rows match the model ledger; first-step strict subsets match the retained strict record. Complete exit/counts/milestones/arm coverage/compile/memory checks all pass. No whole-array rerun followed successful verification.

## Artifacts and limits

Local analysis directory: `outputs/diagnostics/shared-v4-learning-sensitivity-20261004/analysis/`; remote mirror under `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-v4-learning-sensitivity-20261004/analysis/`. Main records are `summary.json`, `learning-effects.json`, `first-step-strict-rechecked.json`, four parameter and three trajectory-gradient JSONs, `actual-CPU-exit.json`, attempt history/logs/source snapshots, and the executable `analyze_saved_learning.py` / finite `run_cpu_analysis.py`. The actual GPU exit record is embedded unchanged in the analysis summary; parent retains the original model/exit/source records and12GB-scale raw arrays remotely.

The accepted three driver repairs and exact original/staged source evidence remain in the preparation review. Graph generation2026-10-03T16:22:23Z had new files not tracked and relevant shared metadata changed; exact source reads were used, not an exhaustive graph/branch-coverage claim. The analysis uses an inert saved optimizer class and three source-declared science namedtuples; it executes no model methods.

This one fixed-data diagnostic demonstrates measurable short-run trajectory sensitivity and a mixed component outcome. It supplies evidence for the parent's scientific decision. It does not authorize adoption, relax strict parity, establish generalization/long-run stability or justify claiming mathematical derivative wrongness or universal learning harm. No architecture, precision, loss, sampling or default was changed by the analysis.
