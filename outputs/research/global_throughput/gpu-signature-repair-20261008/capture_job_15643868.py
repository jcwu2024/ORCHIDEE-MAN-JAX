"""Preserve saved-only evidence on test02; no model/JIT/GPU calls or job writes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

PROJECT = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
SOURCE = PROJECT / 'runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512'
BASE = PROJECT / 'runtime/outputs/research/global_throughput/gpu-signature-repair-20261008'
CODE = PROJECT / 'runtime/worktrees/global-throughput-gpu-sweetspot-v3'
COMMIT = '12b495b9425de3ad1dc16efa3a5a0e69d7e89222'
JOB = '15643868'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', choices=('B256', 'terminal'), required=True)
    args = parser.parse_args()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=CODE, text=True).strip()
    destination = BASE / f'job-{JOB}-{args.label}-evidence'
    for path in (SOURCE, BASE, CODE, destination):
        assert path.resolve().is_relative_to(Path('/WORK/liwei_work/jcwu').resolve())
    assert not destination.exists()
    destination.mkdir(parents=True)
    files = [SOURCE / name for name in (
        'launch-context.txt', 'gpu-identity.json', 'gpu-diagnostic-exits.tsv',
        'selected-gpu-inventory.stdout.txt', 'selected-gpu-inventory.stderr.txt',
        'selected-gpu-processes.stdout.txt', 'selected-gpu-processes.stderr.txt')]
    files += [SOURCE / 'capacity' / name for name in ('report.json', 'events.jsonl', 'stage_manifest.json')]
    files += sorted((SOURCE / 'capacity').glob('stage_*.pkl'))
    files += [PROJECT / f'runtime/logs/global_gpu_capacity_continue_{JOB}.txt']
    proofs = []
    for source in files:
        if not source.is_file():
            proofs.append({'source': str(source), 'missing': True})
            continue
        if source.is_relative_to(SOURCE):
            relative = source.relative_to(SOURCE)
        else:
            relative = Path(source.name)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        before = source.stat()
        with source.open('rb') as stream, target.open('xb') as output:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                output.write(block)
                digest.update(block)
                size += len(block)
        after = source.stat()
        sealed = source.suffix == '.pkl'
        if sealed:
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            assert size == after.st_size
        proofs.append({'source': str(source), 'path': str(relative), 'bytes': size,
                       'sha256': digest.hexdigest(), 'sealed_checkpoint': sealed,
                       'telemetry_may_continue_after_snapshot': not sealed})
    common = ['/rmprog/slurm/v22.05.7/bin/sacct', '-j', JOB, '--parsable2', '--noheader']
    raw9 = subprocess.check_output([*common, '--format=JobIDRaw,JobName,Partition,State,ElapsedRaw,AllocCPUS,AllocTRES,NodeList,ExitCode'], text=True)
    collector7 = subprocess.check_output([*common, '-X', '--format=JobIDRaw,State,ElapsedRaw,AllocCPUS,AllocTRES,ExitCode,NodeList'], text=True)
    (destination / 'sacct-raw9.txt').write_text(raw9)
    (destination / 'sacct-collector7.txt').write_text(collector7)
    proof = {'job_id': JOB, 'code_commit': COMMIT, 'capture_utc': datetime.now(timezone.utc).isoformat(),
             'source': str(SOURCE), 'destination': str(destination), 'label': args.label,
             'scope': 'saved-only evidence copy/hash; no model execution or compilation',
             'files': proofs, 'parent_collector7': collector7.strip()}
    (destination / 'capture.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({'job_id': JOB, 'label': args.label, 'files': len(proofs),
                      'checkpoint_count': sum(item.get('sealed_checkpoint', False) for item in proofs),
                      'destination': str(destination), 'parent_collector7': collector7.strip()}))


if __name__ == '__main__':
    main()
