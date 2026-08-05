"""Measure lossless Gate-E2 OK_LEAK layouts on the bounded Gate-C capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.daily_coarse_graining.typed_sidecar import (  # noqa: E402
    DEFAULT_CONTRACT_PATH,
    load_typed_sidecar_contract,
    read_typed_sidecar_shard,
    select_typed_sidecar_layout,
    write_typed_sidecar_shard,
)

DEFAULT_CAPTURE = (
    ROOT
    / "outputs"
    / "research"
    / "daily_coarse_graining"
    / "gate_c2_daily_flux_capture"
    / "daily_flux_labels.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "research"
    / "daily_coarse_graining"
    / "gate_e2_typed_sidecar"
    / "ok_leak_layout_benchmark.json"
)
FULL_POINT_DAYS = 12_208_581


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark(
    capture_path: str | Path,
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
) -> dict:
    capture_path = Path(capture_path).resolve()
    contract = load_typed_sidecar_contract(contract_path)
    fields = tuple(
        field for field in contract.fields if field.family == "ok_leak_transfer_daily_v1"
    )
    ok_contract = replace(contract, fields=fields)
    with np.load(capture_path, allow_pickle=False) as capture:
        values = {
            field.path: np.ascontiguousarray(capture[field.path], dtype=np.float64)
            for field in fields
        }
    defined = {
        field.path: np.isfinite(values[field.path])
        for field in fields
    }
    day_index = np.asarray([2], dtype=np.int32)
    raw_value_bytes = sum(value.nbytes for value in values.values())
    raw_mask_bytes = sum(mask.nbytes for mask in defined.values())
    selected = {
        field.path: select_typed_sidecar_layout(values[field.path])
        for field in fields
    }
    requested = {
        "dense_deflate": {field.path: "dense" for field in fields},
        "coo_deflate": {field.path: "coo" for field in fields},
        "hybrid_auto_deflate": selected,
    }
    layouts = {}
    with tempfile.TemporaryDirectory(prefix="gate-e2-ok-leak-") as directory:
        temporary = Path(directory)
        for name, layout in requested.items():
            path = temporary / f"{name}.npz"
            write_typed_sidecar_shard(
                path,
                contract=ok_contract,
                day_index=day_index,
                values=values,
                defined=defined,
                layouts=layout,
            )
            restored, masks, restored_days = read_typed_sidecar_shard(
                path,
                contract=ok_contract,
                expected_days=1,
            )
            bit_exact = np.array_equal(restored_days, day_index) and all(
                restored[field.path].tobytes() == values[field.path].tobytes()
                and np.array_equal(masks[field.path], defined[field.path])
                for field in fields
            )
            stored_bytes = path.stat().st_size
            layouts[name] = {
                "stored_bytes": stored_bytes,
                "value_compression_ratio": raw_value_bytes / stored_bytes,
                "value_plus_mask_compression_ratio": (
                    raw_value_bytes + raw_mask_bytes
                )
                / stored_bytes,
                "bit_exact_roundtrip": bit_exact,
                "full_669_projection_gib": (
                    stored_bytes * FULL_POINT_DAYS / 1024**3
                ),
            }
    nonzero = sum(
        np.count_nonzero(value.view(np.uint64)) for value in values.values()
    )
    elements = sum(value.size for value in values.values())
    layout_counts = {
        name: sum(layout == name for layout in selected.values())
        for name in ("dense", "coo", "constant_zero")
    }
    return {
        "schema_version": "gate_e2_ok_leak_layout_benchmark_v1",
        "status": "bounded_local_measurement",
        "capture_path": str(capture_path),
        "capture_sha256": _sha256(capture_path),
        "sidecar_contract_sha256": contract.sha256,
        "sample": {
            "point_days": 1,
            "scope": "Gate-C ordinary PFT14 paper-case day",
            "field_count": len(fields),
            "raw_value_bytes": raw_value_bytes,
            "raw_explicit_mask_bytes": raw_mask_bytes,
            "elements": elements,
            "bit_nonzero_elements": int(nonzero),
            "bit_nonzero_fraction": float(nonzero / elements),
            "mask_policy": (
                "finite-only storage-overhead proxy; the production pilot must "
                "compose parent capability masks"
            ),
        },
        "hybrid_layout_counts": layout_counts,
        "layouts": layouts,
        "decision": {
            "accepted_local_layout": "dense_deflate",
            "scientific_axes_preserved": True,
            "lossy_quantization": False,
            "full_generation_authorized": False,
            "production_mask_semantics_validated": False,
            "next_required_measurement": (
                "repeat dense and hybrid layouts on a representative multi-landpoint, "
                "multi-season typed capture before freezing the production storage budget"
            ),
        },
        "projection_warning": (
            "The full-669 projection repeats one point-day's compressed size and "
            "is a sizing signal, not an accepted production estimate."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = benchmark(args.capture, contract_path=args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
