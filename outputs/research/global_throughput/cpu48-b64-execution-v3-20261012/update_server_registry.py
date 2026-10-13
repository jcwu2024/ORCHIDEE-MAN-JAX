import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("registry", type=Path)
parser.add_argument("entry", type=Path)
parser.add_argument("--key", required=True)
parser.add_argument("--expected-sha256", required=True)
args = parser.parse_args()

raw = args.registry.read_bytes()
before = hashlib.sha256(raw).hexdigest()
if before != args.expected_sha256:
    raise SystemExit(f"registry changed: expected {args.expected_sha256}, found {before}")
data = json.loads(raw)
if args.key in data:
    raise SystemExit(f"refusing to overwrite existing registry entry {args.key}")
entry = json.loads(args.entry.read_text(encoding="utf-8"))
entry["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
entry["registry_previous_sha256"] = before
data[args.key] = entry
data["updated_utc"] = entry["utc"]
data["branch_sync_status"] = (
    "Global exact CPU48 v3 pin 03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8 imported "
    "by verified non-forced bundle fetch; main/daily refs and historical executions preserved."
)
encoded = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
mode = stat.S_IMODE(args.registry.stat().st_mode)
fd, temporary = tempfile.mkstemp(prefix="registry-v3-", dir=str(args.registry.parent))
try:
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, args.registry)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
after = hashlib.sha256(encoded).hexdigest()
json.loads(args.registry.read_bytes())
print(json.dumps({"key": args.key, "utc": entry["utc"], "before_sha256": before,
                  "after_sha256": after, "registry_updated": True}, sort_keys=True))
