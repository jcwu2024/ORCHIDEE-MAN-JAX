"""Supervise one Linux diagnostic window and its shared, bounded GPU ledger."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import resource
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def checked_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to("/WORK/liwei_work/jcwu"):
        raise ValueError(f"path outside jcwu: {path}")
    return path


def group_metrics(pgid):
    """Snapshot only our child's process group; zombies consume no compute."""
    live, rss, hwm, cpu = [], 0, 0, 0.0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid or fields[0] == "Z":
                continue
            live.append(int(entry.name))
            cpu += (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
            rss += int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    hwm = max(hwm, int(line.split()[1]) * 1024)
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    return live, rss, hwm, cpu


def signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def finalize_interrupted_report(output, status):
    report = output / "report.json"
    if report.exists():
        result = json.loads(report.read_text())
        if result.get("status") in {"running", "stopped"}:
            original = output / "report_before_supervisor_finalization.json"
            if not original.exists():
                atomic_json(original, result)
            result["status"] = status
            result["external_termination"] = "see launch-status.json; completed stage artifacts preserved"
            atomic_json(report, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-dir")
    parser.add_argument("--budget-file")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 10 <= args.timeout_seconds <= 1800:
        raise ValueError("window must be 10..1800s including process cleanup")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("missing command")
    output = checked_path(args.output_dir)
    report_output = checked_path(args.report_dir) if args.report_dir else output
    output.mkdir(parents=True, exist_ok=True)
    budget_path = checked_path(args.budget_file) if args.budget_file else None
    lock = None
    ledger = None
    entry = None
    if budget_path:
        budget_path.parent.mkdir(parents=True, exist_ok=True)
        lock = budget_path.with_suffix(".lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = (
            json.loads(budget_path.read_text()) if budget_path.exists() else {"limit_card_seconds": 3600, "windows": []}
        )
        limit = ledger["limit_card_seconds"]
        if limit not in (1800, 3600):
            raise ValueError("supported declared GPU allowances are 1800 or 3600 card-seconds")
        if limit == 1800 and ledger["windows"]:
            raise ValueError("the 1800 card-second allowance permits exactly one GPU window; no retries")
        committed = sum(w["charged_card_seconds"] for w in ledger["windows"])
        if committed + args.timeout_seconds > limit:
            raise ValueError(f"insufficient remaining GPU budget: {limit - committed}s")
        entry = {
            "output_dir": str(output),
            "start_utc": datetime.now(timezone.utc).isoformat(),
            "window_limit_seconds": args.timeout_seconds,
            "charged_card_seconds": args.timeout_seconds,
            "status": "reserved_running",
        }
        ledger["windows"].append(entry)
        atomic_json(budget_path, ledger)

    started = time.monotonic()
    process = None
    over_budget = False
    interrupted = False
    peak_rss = peak_hwm = 0
    cpu_lower_bound = 0.0
    exit_code = None
    failure = None
    survivors = []

    def interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            atomic_json(
                output / "window-process.json",
                {
                    "supervisor_pid": os.getpid(),
                    "child_pid": process.pid,
                    "child_pgid": process.pid,
                    "command": command,
                },
            )
            while process.poll() is None:
                _, rss, hwm, cpu = group_metrics(process.pid)
                peak_rss = max(peak_rss, rss)
                peak_hwm = max(peak_hwm, hwm)
                cpu_lower_bound = max(cpu_lower_bound, cpu)
                elapsed = time.monotonic() - started
                if interrupted or elapsed >= args.timeout_seconds - 5:
                    over_budget = not interrupted
                    signal_group(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        signal_group(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.25)
            exit_code = process.wait(timeout=1)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        if process is not None:
            survivors = group_metrics(process.pid)[0]
            if survivors:
                signal_group(process.pid, signal.SIGKILL)
                time.sleep(0.1)
                survivors = group_metrics(process.pid)[0]
        elapsed = time.monotonic() - started
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        supervisor_usage = resource.getrusage(resource.RUSAGE_SELF)
        status = (
            "over_budget"
            if over_budget
            else ("stopped" if interrupted else ("completed" if exit_code == 0 and not failure else "failed"))
        )
        result = {
            "status": status,
            "exit_code": exit_code,
            "wall_seconds": elapsed,
            "window_limit_seconds": args.timeout_seconds,
            "cpu_group_seconds_lower_bound": cpu_lower_bound,
            "reaped_children_user_seconds": usage.ru_utime,
            "reaped_children_system_seconds": usage.ru_stime,
            "supervisor_user_seconds": supervisor_usage.ru_utime,
            "supervisor_system_seconds": supervisor_usage.ru_stime,
            "supervisor_CPU_affinity": sorted(os.sched_getaffinity(0)),
            "group_peak_sampled_RSS_bytes": peak_rss,
            "process_peak_RSS_bytes": max(peak_hwm, usage.ru_maxrss * 1024),
            "surviving_group_pids": survivors,
            "exception": failure,
            "end_utc": datetime.now(timezone.utc).isoformat(),
        }
        if entry is not None:
            entry.update(result)
            entry["charged_card_seconds"] = math.ceil(elapsed) if not survivors else args.timeout_seconds
            atomic_json(budget_path, ledger)
            result["remaining_card_seconds"] = limit - sum(w["charged_card_seconds"] for w in ledger["windows"])
        atomic_json(output / "launch-status.json", result)
        if status in {"over_budget", "stopped", "failed"}:
            finalize_interrupted_report(report_output, status)
        if lock:
            lock.close()
        print(json.dumps(result), flush=True)
    if survivors:
        raise RuntimeError(f"child group cleanup failed: {survivors}")
    return (
        124 if over_budget else (130 if interrupted else (exit_code if exit_code is not None and exit_code >= 0 else 1))
    )


if __name__ == "__main__":
    raise SystemExit(main())
