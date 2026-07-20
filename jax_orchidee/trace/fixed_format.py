"""Streaming reader for ORCHIDEE fixed-format trace records.

The server trace files use Fortran list-directed text records: a record starts
with a textual tag and continues across following numeric/logical lines until
the next textual tag. This module only parses that explicit trace format; it
does not compute, fill, or approximate model state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping


@dataclass(frozen=True)
class FixedTraceRecord:
    """One fixed-format trace record with raw and coerced values."""

    tag: str
    values: tuple[object, ...]
    raw_values: tuple[str, ...]
    start_line: int
    line_count: int
    path: Path


@dataclass(frozen=True)
class FixedTraceTagSchema:
    """Field contract for one record tag."""

    tag: str
    fields: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class FixedTraceSchema:
    """Field contract for a fixed-format trace file."""

    name: str
    trace_file: str
    tags: Mapping[str, FixedTraceTagSchema]
    provenance: tuple[str, ...]


def _is_numeric_token(token: str) -> bool:
    try:
        float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        return False
    return True


def _is_record_tag(token: str) -> bool:
    return token not in {"T", "F"} and not _is_numeric_token(token)


def _coerce_token(token: str) -> object:
    if token == "T":
        return True
    if token == "F":
        return False
    if any(char in token for char in ".EeDd"):
        try:
            return float(token.replace("D", "E").replace("d", "e"))
        except ValueError:
            return token
    try:
        return int(token)
    except ValueError:
        return token


def _make_record(
    *,
    path: Path,
    tag: str,
    raw_values: list[str],
    start_line: int,
    line_count: int,
) -> FixedTraceRecord:
    raw = tuple(raw_values)
    return FixedTraceRecord(
        tag=tag,
        values=tuple(_coerce_token(token) for token in raw),
        raw_values=raw,
        start_line=start_line,
        line_count=line_count,
        path=path,
    )


def stream_records(
    path: str | Path,
    *,
    tags: str | Iterable[str] | None = None,
    limit: int | None = None,
    encoding: str = "utf-8",
) -> Iterator[FixedTraceRecord]:
    """Yield fixed-format records, optionally filtered by tag and count.

    Trace provenance: designed for the `outputs/server_1961_trace_full_20260623`
    fixed-format trace package. It recognizes only explicit tag boundaries and
    streams records from disk, so large traces such as `orchjax_hydrol_main_
    trace.txt` are not loaded into memory.
    """

    trace_path = Path(path)
    if isinstance(tags, str):
        wanted: frozenset[str] | None = frozenset((tags,))
    elif tags is None:
        wanted = None
    else:
        wanted = frozenset(tags)

    yielded = 0
    tag: str | None = None
    start_line = 0
    line_count = 0
    raw_values: list[str] = []

    with trace_path.open("r", encoding=encoding) as handle:
        for line_number, line in enumerate(handle, start=1):
            tokens = line.split()
            if not tokens:
                continue
            if _is_record_tag(tokens[0]):
                if tag is not None and (wanted is None or tag in wanted):
                    yield _make_record(
                        path=trace_path,
                        tag=tag,
                        raw_values=raw_values,
                        start_line=start_line,
                        line_count=line_count,
                    )
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return
                tag = tokens[0]
                raw_values = list(tokens[1:])
                start_line = line_number
                line_count = 1
                continue
            if tag is None:
                raise ValueError(f"value line before first record tag at {trace_path}:{line_number}")
            raw_values.extend(tokens)
            line_count += 1

    if tag is not None and (wanted is None or tag in wanted):
        yield _make_record(
            path=trace_path,
            tag=tag,
            raw_values=raw_values,
            start_line=start_line,
            line_count=line_count,
        )


def read_records(
    path: str | Path,
    *,
    tags: str | Iterable[str] | None = None,
    limit: int | None = None,
    encoding: str = "utf-8",
) -> tuple[FixedTraceRecord, ...]:
    """Materialize a bounded set of fixed-format records."""

    return tuple(stream_records(path, tags=tags, limit=limit, encoding=encoding))


def first_matching_record(
    path: str | Path,
    *,
    tags: str | Iterable[str] | None = None,
    criteria: Mapping[str, object] | None = None,
    parser,
    scan_limit: int | None = None,
    encoding: str = "utf-8",
) -> dict[str, object] | None:
    """Return the first parsed record matching explicit key/value criteria.

    Trace provenance: this is a streaming helper for fixed-format truth traces.
    `scan_limit` bounds the number of tag-filtered records inspected; no trace
    is materialized in full.
    """

    wanted = criteria or {}
    for record in stream_records(path, tags=tags, limit=scan_limit, encoding=encoding):
        parsed = parser(record)
        if all(parsed.get(key) == value for key, value in wanted.items()):
            return parsed
    return None


def parse_record(record: FixedTraceRecord, schema: FixedTraceTagSchema) -> dict[str, object]:
    """Map one record to named fields using a declared tag schema.

    Trace provenance: schema fields must come from an explicit fixed-format
    trace contract. The function validates arity exactly and does not invent
    missing columns.
    """

    if record.tag != schema.tag:
        raise ValueError(f"record tag {record.tag!r} does not match schema tag {schema.tag!r}")
    if len(record.values) != len(schema.fields):
        raise ValueError(
            f"record {record.tag!r} at {record.path}:{record.start_line} has "
            f"{len(record.values)} values, expected {len(schema.fields)}"
        )
    return dict(zip(schema.fields, record.values, strict=True))
