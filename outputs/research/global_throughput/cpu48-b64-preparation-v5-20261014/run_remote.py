"""Capture exact authorized SSH/transfer operations; never retries an operation."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ssh", "upload", "download"])
    parser.add_argument("name")
    parser.add_argument("source")
    parser.add_argument("destination", nargs="?")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "operations"
    folder.mkdir(exist_ok=True)
    receipt_path = folder / f"{args.name}.json"
    if receipt_path.exists():
        raise SystemExit("operation name already captured; refusing to overwrite")
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=12"]
    if args.mode == "ssh":
        source = Path(args.source)
        payload = source.read_bytes()
        command = ["ssh", "-T", *options, "cln01", "bash -s"]
    else:
        payload = None
        if not args.destination:
            raise SystemExit("transfer destination required")
        remote = args.destination if args.mode == "upload" else args.source
        if not remote.startswith("cln01:/WORK/liwei_work/jcwu/"):
            raise SystemExit("remote transfer path outside approved root")
        command = ["scp", *options, *(["-r"] if args.recursive else []), args.source, args.destination]
    begin = time.time()
    record = {"mode": args.mode, "argv": command, "source": args.source,
              "destination": args.destination, "start_epoch": begin,
              "stdin_sha256": hashlib.sha256(payload).hexdigest() if payload is not None else None}
    receipt_path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    with (folder / f"{args.name}.stdout.txt").open("wb") as stdout, \
         (folder / f"{args.name}.stderr.txt").open("wb") as stderr:
        try:
            process = subprocess.run(command, input=payload, stdout=stdout, stderr=stderr,
                                     timeout=args.timeout, check=False)
            code = process.returncode
        except subprocess.TimeoutExpired:
            record["transport_timeout"] = True
            code = 124
    record.update(exit_code=code, end_epoch=time.time(), wall_seconds=time.time()-begin)
    for stream in ("stdout", "stderr"):
        path = folder / f"{args.name}.{stream}.txt"
        record[stream] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "bytes": path.stat().st_size}
    receipt_path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"name": args.name, "exit_code": code, "wall_seconds": record["wall_seconds"],
                      "stdout_bytes": record["stdout"]["bytes"], "stderr_bytes": record["stderr"]["bytes"]}))
    raise SystemExit(code)

if __name__ == "__main__":
    main()
