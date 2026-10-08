# Fixed1024 job15662711 running; no validated new measurement

The next explicit workload is14336 lane-days (ordered64 genuine FULL64 points×16):
new B512×2 chunks then B1024×1 chunk in one process/job, independently sealed.
Old fixed512 behavior and the accepted B512 job15657822 evidence remain intact.
Old B512 timings are historical512-workload results and are not doubled into
the new1024-workload curve. B512 is the best currently measured chunk size;
its optimality and B1024 capacity/performance are unproven.

Implementation, launcher integration and local strict collector readiness
are complete under [the protocol](PROTOCOL.md). The new paid proposal
was explicitly approved and submitted once as job15662711 from gln01 via192.168.11.2. [Submission receipt](submission-receipt.json) records RUNNING at17:02:31UTC on approved n05, CPU4/GPU1/node1/task1,90min/no-requeue/no-exclusive. Submit/start were17:01:36/17:02:02UTC (01:01:36/01:02:02 Asia/Shanghai2026-10-09). The independent immutable exact-bd9 checkout is unchanged. No retry or automation is authorized.

The [canonical helper probe](canonical-helper-probe.json) passes successive
512/1024 public/internal10/10, exact five native outputs against independent
NumPy and unchanged first-stage signature, in6.905893840 seconds on test02 with
CPU affinity0–3 and OMP/BLAS1. This is tiny-helper evidence only. The
[new historical-origin importer](historical-b512-admission.json) accepts the
actual220430830-byte B512 checkpoint, exact SHA256/source/code/native schema
and compiler-temp/RSS evidence, returns no timing and performs no model work.
Neither check accepts real fixed1024 GPU execution or science.

Validation passed159 lightweight tests:111 capacity/mask/actual Bash-Python
interface cases,33 existing batch regressions, and15 collector synthetic
contract/control-flow fixtures. Changed implementation/tests/tools pass Ruff
and diff checks. Collector tests cover failed-after-seal B512 retention,
shared-origin failure withholding B1024, exact checkpoint IO bytes, and
uncapped terminal-parent accounting even when the approved wall is exceeded.
See [collector readiness](collector-readiness.json),
[preparation verification](verification.json), and [ledger](preparation-ledger.json).

The [canonical Bash receipt](canonical-launcher-interfaces.json) proves actual
argument expansion and direct/spooled equality in a PROJECT-only relocated
diagnostic fixture, retained as historical evidence. [Actual deployed interfaces](deployment-interface-receipt.json) now pass on the exact clean target on test02:10 subprocess checks plus deployed collector admission,3.206605205s, affinity0–3/BLAS1. No PROJECT relocation, JAX import, input read, model work or GPU initialization occurs. The [deployment receipt](deployment-receipt.json) records independent Git objects/no alternates and the preserved clean ad6 source. Execution code
is pinned to `bd9e4c04af68d77e79b984bfc8aa064977f893d3`; the verified local source
bundle and captured shell-source correspondence are recorded in
[final code pin](final-code-pin.json). The [separate paid proposal](B1024_PAID_PROPOSAL.md)
sets the CPU4/GPU1/90min ceilings at CNY3.60 primary/CNY3.72 conservative.
The refreshed15:40:15UTC node snapshot keeps n05 provisional (CPU96/112,load19.37,GPU0/8 allocated); no idle eligible node or reservation is claimed. The fresh17:01:13UTC preflight retained n05 (CPU96/112,load19.69,GPU0/8 allocated); exact checkout clean, run root absent, no duplicate job observed. Startup [log](job-15662711-startup-log.txt) records affinity24,25,80,81 (2 physical cores for4 allocated Slurm CPUs),load18.67,19.75,20.12 and an empty selected-card process list. [Physical GPU identity](job-15662711-gpu-identity.json) is global0,UUID GPU-51897aa5-dac4-9d20-523e-830c362cf7a6,V100-SXM2-32GB,driver525.85.12. These are startup metadata, not model acceptance. Latest17:06:45UTC [observation](job-15662711-observation.txt) remains RUNNING,283s; last saved event is B512 compile start (process113.800671s,RSS10609459200 bytes). These are phase evidence, not completed compiler/stage timing. No strict complete stage, new throughput or final cost is published; independent budget appears in the existing [paid execution ledger](../server-cost-preparation/PAID_EXECUTION_LEDGER.json), with old budgets untouched.

Automatic approval review blocked removal of two generated ignored `.pyc`
files under this preparation's `__pycache__`, with reason `blocked by policy`.
They remain; no alternate deletion method was used.
