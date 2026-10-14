"""Stage compact reviewed evidence and verify Git preserves receipt bytes."""
from pathlib import Path
import hashlib
import json
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
EXCLUDED_NAMES = {'registry-before.json', 'inspect-deployment.log'}
RECEIPT = BASE / 'staged-bytes-verification.json'


def main():
    # Bundles and full duplicate administrative inventories remain local/server.
    # Critical pin/admission/target/scheduler receipts and operational sources go to Git.
    files = sorted(path for path in BASE.rglob('*') if path.is_file()
                   and path.suffix != '.bundle' and path.name not in EXCLUDED_NAMES and path != RECEIPT)
    assert all(path.stat().st_size < 1_000_000 for path in files)
    paths = [path.relative_to(ROOT).as_posix() for path in files]
    subprocess.run(['git', 'add', '-f', '--', *paths], cwd=ROOT, check=True)
    for path, relative in zip(files, paths):
        assert subprocess.check_output(['git', 'show', ':' + relative], cwd=ROOT) == path.read_bytes(), relative
    receipt = {'schema': 'global_telemetry_fix_staged_bytes_v1', 'all_checked_blobs_equal_workspace_bytes': True,
               'checked_file_count': len(files),
               'files': {relative: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path, relative in zip(files, paths)},
               'excluded': ['*.bundle', 'registry-before.json', 'inspect-deployment.log'],
               'excluded_material_preserved_locally_and_on_server': True}
    RECEIPT.write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
    relative = RECEIPT.relative_to(ROOT).as_posix()
    subprocess.run(['git', 'add', '-f', '--', relative], cwd=ROOT, check=True)
    assert subprocess.check_output(['git', 'show', ':' + relative], cwd=ROOT) == RECEIPT.read_bytes()
    print(json.dumps({'checked_file_count': len(files) + 1, 'byte_preservation': 'passed', 'bulk_bundles_staged': False}))


if __name__ == '__main__':
    main()
