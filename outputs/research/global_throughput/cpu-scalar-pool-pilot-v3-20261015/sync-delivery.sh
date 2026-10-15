#!/bin/bash
set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$P/runtime/outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015
cd "$E"
sha256sum -c delivery-payload.sha256
"$P/.venvs/orcjax_cpu/bin/python" -B - "$E" <<'PY'
from pathlib import Path
import hashlib, json, sys, tarfile
e = Path(sys.argv[1]).resolve()
with tarfile.open(e / 'delivery-payload.tar.gz') as stream:
    members = stream.getmembers()
    for member in members:
        path = (e / member.name).resolve()
        assert e in path.parents and member.isfile(), member.name
    manifest = json.load(stream.extractfile('delivery-manifest.json'))
    for record in manifest['files']:
        existing = e / record['name']
        if existing.exists():
            assert hashlib.sha256(existing.read_bytes()).hexdigest() == record['sha256'], record['name']
    stream.extractall(e, filter='data')
PY
"$P/.venvs/orcjax_cpu/bin/python" -B "$E/verify_delivery.py"
