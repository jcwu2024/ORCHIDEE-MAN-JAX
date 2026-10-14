# GPU B64 Comparator Evidence

This note pins the previously collected GPU B64 result used as the same-workload comparator for the CPU48/B64 execution. It records existing evidence only; no model execution or new scheduler query was performed.

## Accepted B64 stage

- Job `15579869`, stage `B64_candidate_first_v1`, execution pin `2b14506992e9c9c7c851cef7b905606e55ac4e04`.
- Workload: 64 ordered unique points over 14 days, totaling 896 point-days. The comparison record says all 64 ordered point IDs and input digests match the canonical CPU comparison input.
- Allocation: one GPU and four allocated CPUs, node `ibc13b03n03`, partition `gnall`; recorded parent status `COMPLETED`, elapsed 2,110 seconds, exit `0:0`.
- Core median: 5.632416466251016 s across five samples, or 159.07914575720022 point-days/s.
- Wrapper median: 6.222332641482353 s across five samples, or 143.99744462818447 point-days/s.
- Use the wrapper rate as the CPU48 wrapper-throughput denominator. Use the core rate only for a comparison at the core timing boundary.

The collected five-sample arrays are retained in `server-cost-results.json` at `accepted_stages[stage=B64_candidate_first_v1].core.samples_seconds` and `accepted_stages[stage=B64_candidate_first_v1].wrapper.samples_seconds`:

- Core seconds: `[5.632416466251016, 5.631819838657975, 5.6331293396651745, 5.627945635467768, 5.635269954800606]`.
- Wrapper seconds: `[6.239071449264884, 6.189428551122546, 6.222332641482353, 6.311673682183027, 6.195937527343631]`.

The stage collector identifies the original GPU output directory as `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/global-throughput-v2-2b14506/gpu-b64/candidate-first`. Its embedded execution origin points to that directory's `report.json`; the accounting provenance points to its parent `launch-context.txt`. Those two original GPU files are referenced by the local records but are not present in this worktree. The stage samples, workload, execution environment and report path are retained in the collector output.

## Accounting and cost limits

The local collection contains `job-accounting-15579869.json` and `sacct-terminal.psv`. The latter retains the parent and step rows; the parent row records 2,110 seconds, four CPUs, `billing=4,cpu=4,gres/gpu=1,node=1`, node `ibc13b03n03`, and exit `0:0`. The normalized job record cross-references the scheduler parent row and launcher TRES. This is collected scheduler accounting evidence, not an invoice or verified billing statement.

The collector's whole-job estimates are CNY 1.406666667 primary and CNY 1.453555556 conservative, with billing marked unverified. They cover the parent job across B64 and B16, startup, compilation, gates, repeats and IO. They are not a hot B64 stage cost and must not be divided by the 896 point-days to derive a same-workload job cost. Separately modeled warm-stage prices are declared scenarios, with billing also unverified.

## Different-workload B1024 context

Job `15671732` measured 64 points repeated 32 times: 2,048 lanes and 28,672 lane-days. Its wrapper rate of 299.1398250995909 lane-days/s is a different-workload reference and is not the CPU48/B64 ratio denominator. Its 3,800-second parent failed after the B1024 child completed and B2048 ran out of memory; the CNY 2.533333333 primary / CNY 2.617777778 conservative estimates cover that failed whole parent job, not a warm B1024 stage. Billing is unverified. See the comparison record for full scope and limitations.

## Local source files and SHA-256

Hashes below identify the local evidence files used for this note.

| Local source | SHA-256 |
| --- | --- |
| `outputs/research/global_throughput/cpu-gpu-b1024-comparison-20261011/comparison.json` | `4d48c4af6d75eee522b4c7db26566f8a4d94f5b662aaf40ab9032aaea39eb804` |
| `outputs/research/global_throughput/cpu-performance-results-15591356/server-cost-results.json` | `cd22479a33c425979be6323f03f830fbcd56345d267a3eadb0b7f74b00538933` |
| `outputs/research/global_throughput/cpu-performance-results-15591356/job-accounting-15579869.json` | `57dbb98ae5d510817f4f1a87259d07e0e73b8210c0249b4f9507ed70f0e58f6e` |
| `outputs/research/global_throughput/cpu-performance-results-15591356/sacct-terminal.psv` | `d035d97e00234b4ceaf30e566189d462d161596ce933c2103a7a201f757ee13d` |
| `outputs/research/global_throughput/cpu-performance-results-15591356/terminal-measurements.json` | `bdc3d83328045e694fdb9bec346d81524a1bd78984aefbccc681a6a9fe9d4af8` |
| `outputs/research/global_throughput/cpu-performance-results-15591356/collection-spec.json` | `77a5e6f591e8e9a21f332aeeced66b0621e59529e682d3300dde8496c16facfa` |
| `outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009/terminal-summary.json` | `a29159b7879d701701c554f7b5d03d101acdf3ace11b8f6f4d388f40248f0c03` |

The collected GPU parent and step accounting rows are available locally. The source GPU stage report and launcher context are not locally mirrored, so their contents are represented through the hashed collector records rather than independently rehashed originals.
