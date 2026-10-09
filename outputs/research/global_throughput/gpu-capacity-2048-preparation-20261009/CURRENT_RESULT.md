# Fixed2048 terminal result: B1024 retained, B2048 execution OOM

Job15671732 ended **FAILED1:0** after3800 parent seconds on n03. The independent
budget is closed. Strict saved-array collection retained **one complete B1024x2 stage**;
B2048 has no complete checkpoint and no two-shape curve or speed ratio is published.
Exact execution pin remains `8aa2e05aa24b65dc4d30a27d4ee857bba6312221`, clean on the server.

B1024 measures the common28672 lane-days:64 genuine FULL64 points repeated32,
continuous Days2–8/9–15,48steps/day. Native dtype/full outputs/masks and fixed
atol1e-8/rtol1e-10 remain unchanged. Actual test02 collector passed all2051 gates,
760 native leaves and2048 saved-output/FULL64 comparisons; exit0,158.59s, canonical
CPU env/taskset0–3/OMPBLAS1/`-B`, with the execution checkout clean afterward.

| Accepted same2048 collection | Median,s [full five-sample range] | Lane-days/s | Warm CNY/lane-day, primary / conservative |
| --- | --- | ---: | ---: |
| B1024x2 core | 81.772918770 [81.765316296–81.787853347] | 350.629529 | 1.9013420473e-06 / 1.96472011554e-06 |
| B1024x2 wrapper | 95.848153921 [95.645140350–96.195542384] | 299.139825 | 2.22861220984e-06 / 2.3028992835e-06 |

Five core samples,s: 81.787853347, 81.765316296, 81.772918770, 81.770447772, 81.773654319.
Five wrapper samples,s: 95.645140350, 95.927785467, 96.195542384, 95.807518750, 95.848153921.
Warm costs use1GPU atCNY2.20/hour plus4 allocated CPUs atCNY0.05/hour (conservative
0.07):CNY2.40/2.48 per allocation-hour. They exclude startup/compile/validation/IO
and failed-candidate time. These repeated64-point capacity measurements are not
2048 unique grids, production IO or general scientific acceptance. Historical
fixed1024 timing was not doubled or imported as this measurement.

B1024 explicit lower99.942532621s/compile970.430264080s/first full2048 call81.978769975s;
input-reference reload4.259086482s/prepared reload4.514236858s/host stack9.991418548s/
device transfer2.145252170s. Checkpoint serialization0.810382809s/atomic write0.767389741s,
881624142bytes. These boundaries remain separate from warm timing and whole-job cost;
no incomplete setup accounting is used for K-amortization estimates.

B2048 compiled (explicit lower66.696225233s/compile1129.087921631s) but failed in the
prepared-wrapper call's second continuous block while requesting9161696464bytes:
`RESOURCE_EXHAUSTED: Out of memory while trying to allocate 9161696464 bytes.`
This is execution OOM, not preallocation guard refusal or a numerical tolerance failure.
Any unsealed first/core samples remain diagnostic and are not promoted to throughput.
It does not prove every B2048 layout impossible on this card or identify a VRAM peak.

The saved candidate initial decision records resident7871545400B/full output880836608B,
estimate27528440336B plus2147483648B slack against selected-card free33721155584B
(total34359738368B). Its calculated postcompile compiler analysis/guard/snapshot were
not persisted before the failed operation; they cannot be reconstructed from the
OOM request. B1024's complete provenance retains actual compiler temp4571419856B,
whole bound22957020592B plus2GiB against originalfree33721155584B, and separate
post-transfer24864882688B/postcompile24703401984B free telemetry. None is peak VRAM.

Parent-only raw9 accounting:3800s = 1.055555556 GPU-card-hours and
4.222222222 allocated CPU-hours; estimated full-job
CNY2.533333333 primary/CNY2.617777778 conservative additive, below
approved4.80/4.96. Billing remains unverified,28s queue is unbilled; batch/extern/child
rows are preserved without adding their time again. B1024 child completed0:0 in2287s;
B2048 child failed1:0 in1501s. No retry, extension, new job or automation occurred.

Full original checkpoint/report/manifest/orchestration/log/diagnostics are preserved
locally and remotely. The881MB pickle remains outside Git; bytes and completed
strict-collector SHA are indexed. See [terminal summary](terminal-summary.json),
[strict collector output](collected-fixed2048.json), [collector log](job-15671732-collector-log.txt),
[raw9 accounting](job-15671732-sacct-raw9.txt), [original failure log](job-15671732-terminal-log.txt),
[artifact index](terminal-artifact-index.json), [original approved proposal](B2048_PAID_PROPOSAL.md),
and [closed paid ledger](../server-cost-preparation/PAID_EXECUTION_LEDGER.json).

No CPU/GPU ratio, Fortran, annual/669-point50-year,10x, trainingAD, proven plateau
or general scientific acceptance follows from this bounded result.
