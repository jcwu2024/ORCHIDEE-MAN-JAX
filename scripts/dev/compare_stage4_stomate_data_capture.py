"""Parse a Stage4 PFT14 capture and strictly compare its JAX owner result.

The expected values are exclusively the STAGE4_DATA records in the supplied
capture. This is not a numeric fixture. A log predating schema 3 is rejected
rather than completed from source defaults or a previous result.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "jax_orchidee").is_dir()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.source_helpers import (  # noqa: E402
    StomateDataConstants,
    stomate_data_owner,
)

SCHEMA_VERSION = 3
PFT_INDEX = 13
VECTOR_KEYS = {
    "bm_sapl_initial_pft14": 12,
    "pheno_gdd_crit_pft14": 3,
    "senescence_temp_pft14": 3,
    "bm_sapl_leaf": 4,
    "dia_coeff": 2,
    "maxdia_coeff": 2,
    "bm_sapl_pft14": 12,
}
BOOL_KEYS = {
    "is_tree_pft14",
    "ok_laidev_pft14",
    "natural_pft14",
    "is_grassland_manag_pft14",
    "ok_dgvm",
    "use_age_class",
}
STRING_KEYS = {"pheno_model_pft14", "senescence_type_pft14"}
OWNER_VECTOR_KEYS = (
    "migrate",
    "maxdia",
    "cn_sapl",
    "lai_initmin",
    "sla",
    "is_tree",
    "pheno_type",
    "ok_laidev",
    "natural",
    "is_grassland_manag",
    "sp_densitesem",
    "sp_pgrainmaxi",
    "tmin_crit",
    "tcm_crit",
    "pheno_model",
    "senescence_type",
    "senescence_hum",
    "nosenescence_hum",
    "leafagecrit",
)
REQUIRED = {
    "schema_version",
    "compiled_nvm",
    "fortran_pft_index",
    "bm_sapl_initial_pft14",
    "migrate_initial_pft14",
    "maxdia_initial_pft14",
    "cn_sapl_initial_pft14",
    "lai_initmin_initial_pft14",
    "nleafages",
    "ok_dgvm",
    "use_age_class",
    "pi",
}
REQUIRED.update(
    f"{key}_pft14"
    for key in OWNER_VECTOR_KEYS
    if key not in {"migrate", "maxdia", "cn_sapl", "lai_initmin"}
)
REQUIRED.update(field.name for field in fields(StomateDataConstants))


def _value(key: str, tokens: list[str]):
    if key in STRING_KEYS:
        return " ".join(tokens).strip()
    if key in BOOL_KEYS:
        return tokens == ["T"]
    values = [float(token.replace("D", "E")) for token in tokens]
    if key in VECTOR_KEYS:
        return values
    if len(values) != 1:
        raise ValueError(f"{key}: expected one value, got {len(values)}")
    return (
        int(values[0])
        if key in {"schema_version", "compiled_nvm", "fortran_pft_index", "nleafages"}
        else values[0]
    )


def parse_capture(path: Path) -> tuple[dict, dict]:
    config, data = {}, {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in {"STAGE4_SOURCE_CONFIG", "STAGE4_DATA"}:
            target = config if parts[0] == "STAGE4_SOURCE_CONFIG" else data
            if parts[1] in target:
                raise ValueError(f"duplicate {parts[0]} record: {parts[1]}")
            target[parts[1]] = _value(parts[1], parts[2:])
    missing = sorted(REQUIRED - set(config))
    if config.get("schema_version") != SCHEMA_VERSION or missing:
        raise ValueError(
            f"strict comparison requires schema {SCHEMA_VERSION}; missing source records: {', '.join(missing)}"
        )
    if config["compiled_nvm"] != 14 or config["fortran_pft_index"] != 14:
        raise ValueError("capture is not the compiled-NVM=14 PFT14 case")
    needed_data = {
        "migrate_pft14",
        "maxdia_pft14",
        "cn_sapl_pft14",
        "tmin_crit_pft14",
        "tcm_crit_pft14",
        "leaf_timecst_pft14",
        "lai_initmin_pft14",
        "bm_sapl_pft14",
    }
    if missing_data := sorted(needed_data - set(data)):
        raise ValueError(f"missing STAGE4_DATA records: {', '.join(missing_data)}")
    return config, data


def _pft_vector(config, name, dtype=float, fill=0):
    value = config[f"{name}_pft14"]
    values = np.full(14, fill, dtype=dtype)
    values[PFT_INDEX] = value
    return values


def compare(path: Path, rtol: float = 1e-12, atol: float = 1e-12) -> dict:
    c, expected = parse_capture(path)
    if not np.isclose(c["pi"], np.pi, rtol=rtol, atol=atol):
        raise ValueError("captured Fortran pi does not match the JAX owner's np.pi")
    bm = np.zeros((14, 12, 1))
    bm[PFT_INDEX, :, 0] = c["bm_sapl_initial_pft14"]
    pheno_gdd_crit = np.zeros((14, 3))
    pheno_gdd_crit[PFT_INDEX] = c["pheno_gdd_crit_pft14"]
    senescence_temp = np.zeros((14, 3))
    senescence_temp[PFT_INDEX] = c["senescence_temp_pft14"]
    constants = StomateDataConstants(
        **{field.name: c[field.name] for field in fields(StomateDataConstants)}
    )
    # The capture exposes the target PFT14 vectors only. Other PFT columns
    # cannot affect PFT14 here, but a neutral nonzero SLA keeps their isolated
    # bookkeeping path numerically defined while the comparison remains PFT14-only.
    result = stomate_data_owner(
        bm_sapl=bm,
        migrate=_pft_vector(c, "migrate_initial"),
        maxdia=_pft_vector(c, "maxdia_initial"),
        cn_sapl=_pft_vector(c, "cn_sapl_initial"),
        lai_initmin=_pft_vector(c, "lai_initmin_initial"),
        sla=_pft_vector(c, "sla", fill=1.0),
        is_tree=_pft_vector(c, "is_tree", bool),
        pheno_type=_pft_vector(c, "pheno_type", int),
        ok_laidev=_pft_vector(c, "ok_laidev", bool),
        natural=_pft_vector(c, "natural", bool),
        is_grassland_manag=_pft_vector(c, "is_grassland_manag", bool),
        sp_densitesem=_pft_vector(c, "sp_densitesem"),
        sp_pgrainmaxi=_pft_vector(c, "sp_pgrainmaxi"),
        tmin_crit=_pft_vector(c, "tmin_crit"),
        tcm_crit=_pft_vector(c, "tcm_crit"),
        pheno_model=_pft_vector(c, "pheno_model", object),
        pheno_gdd_crit=pheno_gdd_crit,
        senescence_type=_pft_vector(c, "senescence_type", object),
        senescence_temp=senescence_temp,
        senescence_hum=_pft_vector(c, "senescence_hum"),
        nosenescence_hum=_pft_vector(c, "nosenescence_hum"),
        leafagecrit=_pft_vector(c, "leafagecrit"),
        nleafages=c["nleafages"],
        ok_dgvm=c["ok_dgvm"],
        use_age_class=c["use_age_class"],
        constants=constants,
    )
    actual = {
        "migrate_pft14": result.migrate[PFT_INDEX],
        "maxdia_pft14": result.maxdia[PFT_INDEX],
        "cn_sapl_pft14": result.cn_sapl[PFT_INDEX],
        "tmin_crit_pft14": result.tmin_crit[PFT_INDEX],
        "tcm_crit_pft14": result.tcm_crit[PFT_INDEX],
        "leaf_timecst_pft14": result.leaf_timecst[PFT_INDEX],
        "lai_initmin_pft14": result.lai_initmin[PFT_INDEX],
        "bm_sapl_pft14": result.bm_sapl[PFT_INDEX, :, 0],
    }
    comparisons = {
        key: {
            "expected": np.asarray(expected[key]).tolist(),
            "actual": np.asarray(value).tolist(),
            "matches": bool(np.allclose(value, expected[key], rtol=rtol, atol=atol)),
        }
        for key, value in actual.items()
    }
    return {
        "capture": str(path),
        "schema_version": SCHEMA_VERSION,
        "strict_pft14_comparison": True,
        "passed": all(row["matches"] for row in comparisons.values()),
        "comparisons": comparisons,
        "note": "Comparison only; this is not a closure claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.capture)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
