# Fixed1024 capacity preparation; no new measurement or paid submission

The next explicit workload is14336 lane-days (ordered64 genuine FULL64 points×16):
new B512×2 chunks then B1024×1 chunk in one process/job, independently sealed.
Old fixed512 behavior and the accepted B512 job15657822 evidence remain intact.
Old B512 timings are historical512-workload results and are not doubled into
the new1024-workload curve. B512 is the best currently measured chunk size;
its optimality and B1024 capacity/performance are unproven.

Implementation, launcher integration and local strict collector readiness
are complete under [the protocol](PROTOCOL.md). The new paid proposal
is separate, requires explicit approval and has not been submitted. No new
paid checkout, job, retry or automation is created by this preparation.

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
diagnostic fixture. It is not deployment admission: the eventual immutable
final-SHA checkout's real-target interface has not been executed. Final code
SHA, source bundle, paid approval and any subsequent execution remain separate.

Automatic approval review blocked removal of two generated ignored `.pyc`
files under this preparation's `__pycache__`, with reason `blocked by policy`.
They remain; no alternate deletion method was used.
