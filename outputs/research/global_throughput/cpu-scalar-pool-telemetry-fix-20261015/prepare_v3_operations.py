"""Relocate only candidate/evidence names; preserve the Slurm v22.05.7 prefix."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPLACEMENTS = [
    ('cpu-scalar-pool-telemetry-fix-20261015', 'cpu-scalar-pool-telemetry-fix-20261015/v3'),
    ('global-throughput-cpu-scalar-pool-v2', 'global-throughput-cpu-scalar-pool-v3'),
    ('global-cpu-scalar-pool-v2-', 'global-cpu-scalar-pool-v3-'),
    ('codex/global-cpu-scalar-pool-v2', 'codex/global-cpu-scalar-pool-v3'),
    ('global_cpu_scalar_pool_telemetry_fix_v2_', 'global_cpu_scalar_pool_telemetry_fix_v3_'),
    ('.scalar-pool-v2.tmp', '.scalar-pool-v3.tmp'),
]
for name in ['verify-target.sh', 'run-free-verification.sh']:
    source = (BASE / name).read_text(encoding='utf-8')
    for old, new in REPLACEMENTS:
        source = source.replace(old, new)
    assert '/rmprog/slurm/v22.05.7/bin/' in source
    assert '/rmprog/slurm/v32' not in source
    (BASE / 'v3' / name).write_bytes(source.encode())
