"""Recover the complete checkout prefix of this interrupted archive.

The following worktree portion was interrupted; stop at its first header.
No source or existing archive is removed. The recovered archive must still
pass archive_runtime_checkouts --resume-packed verification before cleanup.
"""
import gzip
import tarfile
from pathlib import Path

root = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/archives')
source = root / 'inactive-checkout-working-files-20260922.tar.partial'
destination = root / 'inactive-checkouts-20260922.tar.gz'
members = 0
with destination.open('xb') as raw:
    with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|') as output:
            with tarfile.open(source, 'r|gz') as original:
                for item in original:
                    if item.name.startswith('worktrees/'):
                        break
                    if not item.name.startswith('checkouts/'):
                        raise ValueError(f'unexpected prefix: {item.name}')
                    output.addfile(item, original.extractfile(item) if item.isfile() else None)
                    members += 1
                else:
                    raise ValueError('expected complete checkout/worktree boundary not found')
print(f'Recovered {members} members to {destination}; verification/removal still pending', flush=True)
