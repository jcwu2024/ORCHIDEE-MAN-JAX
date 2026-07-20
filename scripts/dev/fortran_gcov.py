from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


LINE_RE = re.compile(r"^\s*([^:]+):\s*(\d+):(.*)$")
BRANCH_RE = re.compile(r"^branch\s+(\d+)\s+(?:taken\s+(\d+)|never executed)")


def parse_gcov(path: Path) -> dict[int, list[dict[str, int]]]:
    branches: dict[int, list[dict[str, int]]] = defaultdict(list)
    source_line: int | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line_match = LINE_RE.match(raw)
        if line_match is not None:
            source_line = int(line_match.group(2))
            continue
        branch_match = BRANCH_RE.match(raw.strip())
        if branch_match is not None and source_line is not None:
            branches[source_line].append(
                {
                    "branch_index": int(branch_match.group(1)),
                    "taken": int(branch_match.group(2) or 0),
                }
            )
    return dict(branches)


def parse_gcov_line_counts(path: Path) -> dict[int, int]:
    """Return executable source-line counts from a GNU gcov listing."""

    counts: dict[int, int] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE_RE.match(raw)
        if match is None:
            continue
        token = match.group(1).strip().replace("*", "")
        if token.isdigit():
            counts[int(match.group(2))] = int(token)
    return counts


def select_arm_branch(
    arm_id: str, branches: list[dict[str, int]]
) -> dict[str, int] | None:
    _, kind, arm = arm_id.rsplit(":", 2)
    if kind == "where":
        if len(branches) < 2:
            return None
        return branches[-2] if arm == "true" else branches[-1]
    if len(branches) < 2:
        return None
    return branches[-2] if arm in {"true", "case"} else branches[-1]


def locate_generated_span(source_bytes: bytes, span_bytes: bytes) -> int:
    offset = source_bytes.find(span_bytes)
    if offset < 0 or source_bytes.find(span_bytes, offset + 1) >= 0:
        raise RuntimeError("cannot uniquely locate extracted procedure in generated harness")
    return source_bytes[:offset].count(b"\n") + 1
