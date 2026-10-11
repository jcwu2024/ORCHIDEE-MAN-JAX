"""Save one authorized SSH/SCP command's actual stdout, stderr and metadata."""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("label")
parser.add_argument("--stdin", type=Path)
parser.add_argument("--timeout", type=int, default=180)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
root = Path(__file__).resolve().parent
base = root / args.label
if base.with_suffix(".command.json").exists():
    raise ValueError("command labels are single-use")
started = datetime.now(timezone.utc).isoformat()
run = subprocess.run(command, input=args.stdin.read_bytes() if args.stdin else None,
                     capture_output=True, timeout=args.timeout)
base.with_suffix(".stdout.txt").write_bytes(run.stdout)
base.with_suffix(".stderr.txt").write_bytes(run.stderr)
receipt = {"command": command, "stdin": str(args.stdin) if args.stdin else None,
           "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
           "returncode": run.returncode, "stdout_bytes": len(run.stdout), "stderr_bytes": len(run.stderr)}
base.with_suffix(".command.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt))
print(run.stdout.decode("utf-8", "replace")[-16000:])
if run.stderr:
    print(run.stderr.decode("utf-8", "replace")[-3000:])
raise SystemExit(run.returncode)
