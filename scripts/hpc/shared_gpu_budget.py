"""Persistent occupied-card budget for the authorized free-node diagnostic.

Run on the compute host. The locked ledger reserves the entire session before
starting it; interrupted records continue to consume their reservation until
explicitly reconciled from process/exit evidence. TERM/KILL is inside the cap.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

LIMIT = 18000
SESSION_LIMIT = 7200
ORIGINAL_SCOPE = "original_18000"
ORIGINAL_ROUTE = "original_full_state_schedule_20261005"
ORIGINAL_STAGE = "original-full-state-schedule"
ORIGINAL_ROUTE_LIMIT = 1000
ORIGINAL_SESSION_LIMIT = 500
ORIGINAL_ENTRY = "probe_shared_original_full_state_schedule.py"
SCHEDULE_EXTRA_SCOPE = "original_schedule_additional_1800_20261005"
SCHEDULE_EXTRA_ROUTE = "original_full_state_schedule_extra_1800_20261005"
SCHEDULE_EXTRA_STAGE = "original-full-state-schedule-extra-1800"
SCHEDULE_EXTRA_LIMIT = 1800
SCHEDULE_EXTRA_SESSION_LIMIT = 900

def authorize_limit(data, approved_limit=None, approval_note=None):
    """Apply an explicitly approved increment without replacing any sessions."""
    current = data["limit_gpu_seconds"]
    requested = LIMIT if approved_limit is None else approved_limit
    if current < LIMIT or requested < current:
        raise ValueError("budget limit cannot be reset or reduced")
    if requested > LIMIT and not (approval_note and approval_note.strip()):
        raise ValueError("increased budget requires an explicit approval note")
    if requested > current:
        data.setdefault("budget_authorizations", []).append({
            "previous_limit_gpu_seconds": current,
            "limit_gpu_seconds": requested,
            "increment_gpu_seconds": requested - current,
            "approval_note": approval_note.strip(),
            "recorded_epoch": time.time()})
        data["limit_gpu_seconds"] = requested
    return requested

@contextmanager
def locked(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {
            "schema_version": "occupied_gpu_budget_v1", "limit_gpu_seconds": LIMIT,
            "session_limit_seconds": SESSION_LIMIT, "sessions": []}
        if data["limit_gpu_seconds"] < LIMIT or data["session_limit_seconds"] != SESSION_LIMIT:
            raise ValueError("budget identity changed")
        yield data
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(temporary, path)

def charge(row):
    return row.get("occupied_gpu_seconds", row["reserved_gpu_seconds"])


def _original_entry(command):
    return any(Path(str(value)).name == ORIGINAL_ENTRY for value in command)


def _extra_authorization(data):
    rows = [row for row in data.get("budget_authorizations", ())
            if row.get("previous_limit_gpu_seconds") == 25200
            and row.get("limit_gpu_seconds") == 27000
            and row.get("increment_gpu_seconds") == SCHEDULE_EXTRA_LIMIT]
    if (len(rows) != 1 or not rows[0].get("approval_note")
            or not isinstance(rows[0].get("recorded_epoch"), (int, float))):
        raise ValueError("additional original schedule requires its own25200→27000 user authorization record")
    return rows[0]


def _extra_row(row):
    marked = (row.get("funding_scope") == SCHEDULE_EXTRA_SCOPE
              or row.get("funding_route") == SCHEDULE_EXTRA_ROUTE
              or row.get("stage") == SCHEDULE_EXTRA_STAGE)
    if marked and (row.get("funding_scope") != SCHEDULE_EXTRA_SCOPE
                   or row.get("funding_route") != SCHEDULE_EXTRA_ROUTE
                   or row.get("stage") != SCHEDULE_EXTRA_STAGE
                   or row.get("cards") != 2
                   or not 20 <= row.get("session_cap_seconds", 0) <= SCHEDULE_EXTRA_SESSION_LIMIT):
        raise ValueError("additional route record has altered stage/scope/source/card cap")
    return marked


def original_funding_snapshot(data):
    """Audit the settled original prefix and conservatively charge scoped retries."""
    authorizations = [row for row in data.get("budget_authorizations", ())
                      if row.get("previous_limit_gpu_seconds") == LIMIT
                      and row.get("increment_gpu_seconds", 0) > 0]
    if len(authorizations) != 1 or not isinstance(authorizations[0].get("recorded_epoch"), (int, float)):
        raise ValueError("original funding requires one known increment authorization boundary")
    authorization = authorizations[0]
    boundary = authorization["recorded_epoch"]
    sessions = data["sessions"]
    baseline = []
    for row in sessions:
        ended = row.get("ended_epoch")
        if not isinstance(ended, (int, float)) or ended > boundary:
            break
        if row.get("status") != "finished" or "occupied_gpu_seconds" not in row or row.get("funding_scope"):
            raise ValueError("original funding historical prefix is not settled or has unknown source")
        baseline.append(row)
    if not baseline or any(isinstance(row.get("ended_epoch"), (int, float))
                           and row["ended_epoch"] <= boundary for row in sessions[len(baseline):]):
        raise ValueError("original funding boundary is not a complete contiguous historical prefix")
    ids = [row["id"] for row in baseline]
    if len(set(row["id"] for row in sessions)) != len(sessions):
        raise ValueError("duplicate budget session identity")
    source = {"baseline_session_ids": ids,
              "baseline_charged_gpu_seconds": sum(charge(row) for row in baseline),
              "authorization_boundary": dict(authorization)}
    recorded = data.get("funding_scopes", {}).get(ORIGINAL_SCOPE)
    if recorded is not None and any(recorded.get(key) != value for key, value in source.items()):
        raise ValueError("original funding baseline identity changed")
    funded = []
    for row in sessions[len(baseline):]:
        if _extra_row(row):
            _extra_authorization(data)
            continue
        belongs = (row.get("funding_scope") == ORIGINAL_SCOPE or row.get("funding_route") == ORIGINAL_ROUTE
                   or row.get("stage") == ORIGINAL_STAGE or _original_entry(row.get("command", ())))
        if belongs:
            if (row.get("funding_scope") != ORIGINAL_SCOPE or row.get("funding_route") != ORIGINAL_ROUTE
                    or row.get("stage") != ORIGINAL_STAGE or row.get("cards") != 2
                    or not 20 <= row.get("session_cap_seconds", 0) <= ORIGINAL_SESSION_LIMIT):
                raise ValueError("original route record has unknown or altered stage/scope/source")
            funded.append(row)
    costs = [charge(row) for row in baseline + funded]
    if any(not math.isfinite(value) or value < 0 for value in costs):
        raise ValueError("original funding charge is not finite/nonnegative")
    route_consumed = sum(charge(row) for row in funded)
    original_consumed = source["baseline_charged_gpu_seconds"] + route_consumed
    return source | {"limit_gpu_seconds": LIMIT, "route": ORIGINAL_ROUTE,
                     "route_limit_gpu_seconds": ORIGINAL_ROUTE_LIMIT,
                     "route_charged_or_reserved_gpu_seconds": route_consumed,
                     "original_charged_or_reserved_gpu_seconds": original_consumed,
                     "original_remaining_gpu_seconds": LIMIT - original_consumed,
                     "route_remaining_gpu_seconds": ORIGINAL_ROUTE_LIMIT - route_consumed}


def extra_schedule_funding_snapshot(data):
    """The additional route uses only its own1800 grant, preserving original/daily sources."""
    authorization = _extra_authorization(data)
    original_funding_snapshot(data)  # Verify the original settled source independently.
    source = {"authorization_source": dict(authorization)}
    recorded = data.get("funding_scopes", {}).get(SCHEDULE_EXTRA_SCOPE)
    if recorded is not None and recorded.get("authorization_source") != source["authorization_source"]:
        raise ValueError("additional schedule authorization identity changed")
    costs = [charge(row) for row in data["sessions"] if _extra_row(row)]
    if any(not math.isfinite(value) or value < 0 for value in costs):
        raise ValueError("additional schedule charge is not finite/nonnegative")
    consumed = sum(costs)
    return source | {"limit_gpu_seconds": SCHEDULE_EXTRA_LIMIT, "route": SCHEDULE_EXTRA_ROUTE,
        "route_limit_gpu_seconds": SCHEDULE_EXTRA_LIMIT,
        "route_charged_or_reserved_gpu_seconds": consumed,
        "route_remaining_gpu_seconds": SCHEDULE_EXTRA_LIMIT - consumed,
        "source_semantics": "only separately approved1800; original18000 and daily-only7200 are excluded"}


def funding_snapshot(data, scope):
    return (original_funding_snapshot(data) if scope == ORIGINAL_SCOPE
            else extra_schedule_funding_snapshot(data))


def check_funding(data, *, scope, stage, seconds, cards, command):
    """The original-only route cannot draw from the daily extension or reset on retry."""
    required = scope is not None or stage in (ORIGINAL_STAGE, SCHEDULE_EXTRA_STAGE) or _original_entry(command)
    if not required:
        return None
    if scope == SCHEDULE_EXTRA_SCOPE:
        if stage != SCHEDULE_EXTRA_STAGE or cards != 2 or not 20 <= seconds <= SCHEDULE_EXTRA_SESSION_LIMIT:
            raise ValueError("additional original schedule requires fixed stage, two cards and≤900 seconds")
        snapshot = extra_schedule_funding_snapshot(data)
        if snapshot["route_charged_or_reserved_gpu_seconds"] + cards*seconds > SCHEDULE_EXTRA_LIMIT:
            raise ValueError("additional fixed route1800 GPU-second allowance exhausted")
        return snapshot
    if scope != ORIGINAL_SCOPE or stage != ORIGINAL_STAGE:
        raise ValueError("original schedule entry requires its fixed original funding scope/stage")
    if cards != 2 or not 20 <= seconds <= ORIGINAL_SESSION_LIMIT:
        raise ValueError("original schedule requires two cards and at most500 occupied seconds")
    snapshot = original_funding_snapshot(data)
    reserved = cards * seconds
    if snapshot["original_charged_or_reserved_gpu_seconds"] + reserved > LIMIT:
        raise ValueError("original18000 allowance exhausted; daily extension cannot fund this route")
    if snapshot["route_charged_or_reserved_gpu_seconds"] + reserved > ORIGINAL_ROUTE_LIMIT:
        raise ValueError("original fixed route1000 GPU-second allowance exhausted")
    return snapshot


def run(args):
    if not 20 <= args.seconds <= SESSION_LIMIT or args.cards not in (1, 2):
        raise ValueError("invalid occupied session cap/card count")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("missing command")
    identifier = str(uuid.uuid4())
    started_wall = time.time()
    started = time.monotonic()
    with locked(args.ledger) as data:
        limit = authorize_limit(data, args.approved_limit_gpu_seconds, args.approval_note)
        consumed = sum(map(charge, data["sessions"]))
        reserved = args.cards * args.seconds
        if consumed + reserved > limit:
            raise ValueError(f"budget exhausted: charged/reserved {consumed}; request {reserved}")
        funding = check_funding(data, scope=args.funding_scope, stage=args.stage,
                                seconds=args.seconds, cards=args.cards, command=command)
        row = {"id": identifier, "stage": args.stage,
            "host": os.uname().nodename, "wrapper_pid": os.getpid(), "cards": args.cards,
            "session_cap_seconds": args.seconds, "reserved_gpu_seconds": reserved,
            "started_epoch": started_wall, "command": command, "status": "reserved"}
        if funding is not None:
            row.update(funding_scope=args.funding_scope, funding_route=funding["route"])
            data.setdefault("funding_scopes", {})[args.funding_scope] = funding
        data["sessions"].append(row)
        if funding is not None:
            data["funding_scopes"][args.funding_scope] = funding_snapshot(data, args.funding_scope)
    process = None
    status = 125
    interrupted = None
    def stop(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        if process is not None:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        process = subprocess.Popen(command, start_new_session=True)
        try:
            status = process.wait(timeout=max(0.1, args.seconds - 12 - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=1)
            status = 124
    finally:
        # Count any inherited worker cleanup in the occupied session too.
        # A completed launcher must not strand a CUDA child after accounting.
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                cleanup_until = min(started + args.seconds - 1, time.monotonic() + 10)
                while time.monotonic() < cleanup_until:
                    try: os.killpg(process.pid, 0)
                    except ProcessLookupError: break
                    time.sleep(0.1)
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            except ProcessLookupError:
                pass
        elapsed = time.monotonic() - started
        with locked(args.ledger) as data:
            row = next(row for row in data["sessions"] if row["id"] == identifier)
            row.update(status="finished", ended_epoch=time.time(), occupied_seconds=elapsed,
                occupied_gpu_seconds=elapsed * args.cards, exit_status=status,
                interrupted_signal=interrupted, within_session_cap=elapsed <= args.seconds)
            data["charged_gpu_seconds"] = sum(map(charge, data["sessions"]))
            data["remaining_gpu_seconds"] = data["limit_gpu_seconds"] - data["charged_gpu_seconds"]
            if args.funding_scope is not None:
                data["funding_scopes"][args.funding_scope] = funding_snapshot(data, args.funding_scope)
        print(json.dumps({"session": identifier, "seconds": elapsed, "cards": args.cards,
            "exit_status": status}), flush=True)
    return status

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--seconds", type=int, required=True)
    p.add_argument("--cards", type=int, required=True)
    p.add_argument("--approved-limit-gpu-seconds", type=int,
        help="explicit user-approved cumulative limit; default remains 18000")
    p.add_argument("--approval-note", help="human authorization provenance for an increment")
    p.add_argument("--funding-scope", choices=(ORIGINAL_SCOPE, SCHEDULE_EXTRA_SCOPE),
        help="fixed original route≤1000/500s or separately approved schedule extension≤1800/900s; two cards")
    p.add_argument("command", nargs=argparse.REMAINDER)
    raise SystemExit(run(p.parse_args()))

if __name__ == "__main__": main()
