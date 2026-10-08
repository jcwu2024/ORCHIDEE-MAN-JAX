"""One-time proof behind the two exact aggregate byte pins."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
from scripts.dev.global_source_compatibility import audit_diffuco_bytes, source_identity

revision = '8b7732aaa0c75d58f1aa515a7fccbadd628f353b'
paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision, '--', 'jax_orchidee'], cwd=root, text=True).splitlines()
canonical, windows = hashlib.sha256(), hashlib.sha256()
rows = []
for path in paths:
    blob = subprocess.check_output(['git', 'show', revision + ':' + path], cwd=root)
    actual = (root / path).read_bytes()
    baseline = audit_diffuco_bytes(actual)[0] if path == 'jax_orchidee/sechiba/diffuco.py' else actual
    if baseline == blob:
        representation = 'exact immutable Git blob'
    elif baseline == blob.replace(b'\n', b'\r\n'):
        representation = 'exact immutable Git blob LF-to-CRLF checkout'
    else:
        raise ValueError('unapproved checkout bytes: ' + path)
    canonical.update(path.encode() + b'\0' + blob + b'\0')
    windows.update(path.encode() + b'\0' + baseline + b'\0')
    rows.append({'path': path, 'representation': representation, 'git_blob_sha256': hashlib.sha256(blob).hexdigest(),
                 'baseline_checkout_sha256': hashlib.sha256(baseline).hexdigest()})
assert canonical.hexdigest() == '11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5'
assert windows.hexdigest() == '4151db252c62cf24899300c7cb2eac783405888fce1384030332d4622aa2c196'
report = {'immutable_git_revision': revision, 'files': rows, 'canonical_sha256': canonical.hexdigest(),
          'exact_windows_baseline_sha256': windows.hexdigest(), 'current_execution_identity': source_identity(root),
          'method': 'each certified baseline file equals immutable Git blob or its exact LF-to-CRLF representation; admission uses two aggregate pins, no generic EOL normalization'}
Path(__file__).with_name('checkout-byte-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'files_verified': len(rows), 'canonical_sha256': canonical.hexdigest(), 'windows_baseline_sha256': windows.hexdigest()}))
