"""Preserve source and exact fixture values across style-only cleanup."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/canopy"
FILES = (
    "scripts/dev/oracle_canopy_component_fixture.py",
    "scripts/dev/oracle_canopy_component_certify.py",
    "scripts/dev/oracle_canopy_component_report.py",
    "tests/unit/test_canopy_component_local_jit.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("before", "after"))
    args = parser.parse_args()
    # The caller uses python -m from the declared repository cwd.
    from scripts.dev.oracle_canopy_component_fixture import make_canopy_fixture

    records = {}
    backup = OUT / "style_before"
    backup.mkdir(exist_ok=True)
    for name in FILES:
        path = ROOT / name
        source = path.read_text(encoding="utf-8")
        if args.phase == "before":
            (backup / path.name).write_bytes(path.read_bytes())
        records[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ast_sha256": hashlib.sha256(ast.dump(ast.parse(source), include_attributes=False).encode()).hexdigest(),
        }
    fixture = make_canopy_fixture()
    values = {
        "case_names": fixture.case_names,
        "payload": fixture.diffuco_precall.payload,
        "static_kwargs": fixture.static_kwargs,
        "driver": fixture.driver,
        "albedo": fixture.albedo,
        "thermal": fixture.thermal,
    }
    raw = json.dumps(values, sort_keys=True, default=lambda x: x.tolist()).encode()
    if args.phase == "before":
        (backup / "fixture_values.json").write_bytes(raw)
    records["fixture_values_sha256"] = hashlib.sha256(raw).hexdigest()
    if args.phase == "after":
        before = json.loads((OUT / "style_before.json").read_text())
        records["fixture_values_exact_equal"] = raw == (backup / "fixture_values.json").read_bytes()
        records["old_source"] = before
        records["semantics"] = (
            "Imports/spacing only in test and report. Fixture grid/pft lambdas become identical "
            "nested defs returning the same np.full(shape,value,dtype=float64); all ten cases, "
            "payload/static/driver/albedo/thermal values compare exactly. Certifier adds actual "
            "module import path to the receipt and retains x64 side-effect import. "
            "No physical algorithm, coefficients, cases, masks or production source changed."
        )
        if not records["fixture_values_exact_equal"]:
            raise AssertionError("style cleanup changed component fixture values")
    (OUT / f"style_{args.phase}.json").write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps(
            {
                "phase": args.phase,
                "fixture_sha256": records[FILES[0]]["sha256"],
                "values_sha256": records["fixture_values_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
