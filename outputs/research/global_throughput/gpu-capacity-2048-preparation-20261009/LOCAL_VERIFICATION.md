# Fixed2048 bounded preparation checks

Runner/coordinator owner: 67 new tests passed, 94 combined new/fixed1024 tests passed,
and 45 historical512 tests passed (11 deselected). Evidence is in
`runner-coordinator-checks.json`. Collector owner evidence is in `collector-readiness.json`.

Launcher owner: 17 tests passed (10 new2048 plus7 historical1024); direct and Slurm-spooled
Bash expansion and CPU-only coordinator plan were exercised. Python Ruff and Bash syntax
checks passed. These checks use synthetic contracts and dry plans, without model input
regeneration, real-model lower/compile/run or GPU initialization. Existing cache_dir warning
appeared because pytest cacheprovider was disabled to preserve a clean worktree.

Graph Verify evidence uses project E-ORCHIDEE-MAN-JAX, generation2026-10-08T18:30:49Z.
The operated worktree/new paths are excluded/not tracked; one child coverage request
returned Transport closed. Parent confirmed the same excluded coverage and exact source
fallback was used. No graph completeness claim or reindex is made.

Actual immutable target interface receipt is recorded separately after deployment.
No paid job, environment installation, allocator change, cache deletion or science re-audit
was performed. Local checks do not certify a B2048 numerical result or scientific acceptance.
