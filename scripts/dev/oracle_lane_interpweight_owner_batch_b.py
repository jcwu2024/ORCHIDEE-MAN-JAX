"""Stage 4 Batch B evidence preparation for the two interpolation owners.

This lane deliberately does not translate the Fortran owner.  It extracts the
two original procedures byte-for-byte and records the scientific call boundary
that a Fortran harness must bind (NetCDF/MPI are IO boundaries; aggregate_p and
the extracted interpolation helpers remain scientific dependencies).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from jax_orchidee.driver.interpolation_core12 import (  # noqa: E402
    AggregatePacket,
    FortranUndefinedVariableError,
    InterpolationSource,
    InterpolationTarget,
    interpweight_2d,
)
from jax_orchidee.driver.interpolation_core4cont import (  # noqa: E402
    FortranUndefinedInputError,
    interpweight_2dcont_routed,
)

FAMILY = "interpweight_owner_batch_b"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/interpweight.f90"
OWNERS = {
    "pft14-owner-contract-252914bbabc7": "interpweight_2D",
    "pft14-owner-contract-372b01d64eab": "interpweight_2Dcont",
}
DEPENDENCIES = (
    "interpweight_get_varNdims_file",
    "interpweight_get_var3dims_file",
    "interpweight_get_var4dims_file",
    "interpweight_calc_resolution_in",
    "interpweight_modifying_input2D",
    "interpweight_modifying_input3D",
    "interpweight_modifying_input4D",
    "interpweight_masking_input2D",
    "interpweight_masking_input3D",
    "interpweight_masking_input4D",
    "interpweight_provide_fractions2D",
    "interpweight_provide_interpolation2D",
    "aggregate_p",
)
SOURCE_LEVEL_IMPOSSIBILITY = {
    "interpweight_2D": {
        "rank3_allocation": 603,
        "rank4_time_selection_allocation": 628,
        "rank4_full_allocation": 646,
        "unconditional_unallocated_bcast": 663,
        "unconditional_unallocated_writeback": 854,
        "symbols": ["invar2D", "invar3D", "invar4D"],
    },
    "interpweight_2Dcont": {
        "rank3_allocation": 1972,
        "rank4_time_selection_allocation": 1993,
        "rank4_full_allocation": 2011,
        "unconditional_unallocated_bcast": 2028,
        "unconditional_unallocated_writeback": 2189,
        "symbols": ["invar2D", "invar3D", "invar4D"],
    },
}


def _target() -> InterpolationTarget:
    return InterpolationTarget(
        lalo=np.asarray([[0.0, 179.75]]),
        resolution=np.asarray([[100.0, 100.0]]),
        neighbours=np.zeros((1, 8), dtype=np.int32),
        contfrac=np.asarray([1.0]),
    )


def _aggregate(request):
    return AggregatePacket(
        sub_index=np.zeros((1, request.nbvmax, 2), dtype=np.int32),
        sub_area=np.zeros((1, request.nbvmax)),
        ok=True,
    )


def _boundary_outcomes() -> list[dict[str, object]]:
    source3 = InterpolationSource(
        values=np.ones((2, 2, 14)),
        longitude=np.asarray([179.75, -179.75]),
        latitude=np.asarray([1.0, -1.0]),
        variable_name="maxvegetfrac",
    )
    source1 = InterpolationSource(
        values=np.ones((2,)),
        longitude=np.asarray([0.0, 1.0]),
        latitude=np.asarray([0.0, 1.0]),
        variable_name="invalid",
    )
    probes = (
        (
            "fortran_source/ORCHIDEE/src_global/interpweight.f90:593:case",
            "interpweight_2D",
            lambda: interpweight_2d(source3, _target(), [-1.0] * 14, _aggregate, varmin=0.0, varmax=1.0, noneg=False, masktype="nomask", maskvalues=(0.0, 0.0, 0.0)),
            FortranUndefinedVariableError,
            "source_undefined_unallocated_invar2D",
        ),
        (
            "fortran_source/ORCHIDEE/src_global/interpweight.f90:1967:case",
            "interpweight_2Dcont",
            lambda: interpweight_2dcont_routed(source3, _target(), _aggregate),
            FortranUndefinedInputError,
            "source_undefined_unallocated_invar2D",
        ),
        (
            "fortran_source/ORCHIDEE/src_global/interpweight.f90:657:default",
            "interpweight_2D",
            lambda: interpweight_2d(source1, _target(), [1.0], _aggregate, varmin=0.0, varmax=1.0, noneg=False, masktype="nomask", maskvalues=(0.0, 0.0, 0.0)),
            ValueError,
            "source_ipslerr_p_fatal",
        ),
        (
            "fortran_source/ORCHIDEE/src_global/interpweight.f90:2022:default",
            "interpweight_2Dcont",
            lambda: interpweight_2dcont_routed(source1, _target(), _aggregate),
            ValueError,
            "source_ipslerr_p_fatal",
        ),
    )
    outcomes = []
    for arm_id, procedure, probe, expected, source_outcome in probes:
        try:
            probe()
        except expected as error:
            outcomes.append({"arm_id": arm_id, "procedure": procedure, "source_outcome": source_outcome, "jax_outcome": type(error).__name__, "passed": True})
        except Exception as error:  # pragma: no cover - evidence must retain unexpected errors
            outcomes.append({"arm_id": arm_id, "procedure": procedure, "source_outcome": source_outcome, "jax_outcome": type(error).__name__, "passed": False})
        else:
            outcomes.append({"arm_id": arm_id, "procedure": procedure, "source_outcome": source_outcome, "jax_outcome": "returned", "passed": False})
    return outcomes
MICROCASES = (
    "periodic_dateline_axis_coordinates",
    "reversed_latitude_axis_coordinates",
    "rank2_rank3_rank4_time_dimensions",
    "nomask_mbelow_mabove_msumrange_and_var_masks",
    "zero_and_missing_overlap_weights",
    "boundary_cells_default_and_slopecalc_writebacks",
)


def run_oracle(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = {}
    source_bytes = SOURCE.read_bytes()
    for contract, procedure in OWNERS.items():
        span = extract_procedure_bytes(SOURCE, procedure)
        (output_dir / f"{procedure}.f90").write_bytes(span.span_bytes)
        extracted[contract] = {
            "fortran_procedure": procedure,
            "start_line": span.start_line,
            "end_line": span.end_line,
            "span_sha256": span.span_sha256,
            "extracted_file": f"{procedure}.f90",
        }
    boundaries = _boundary_outcomes()
    owner_evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": False,
        "records": [
            {"owner_contract": contract, **record, "direct_scientific_dependencies": list(DEPENDENCIES)}
            for contract, record in extracted.items()
        ],
        "source_level_impossibility": SOURCE_LEVEL_IMPOSSIBILITY,
        "pft14_forcing_route": {
            "file": "data/INPUTDIR_ZZ/PFT1860_mangr_025deg.nc",
            "variable": "maxvegetfrac",
            "dimensions": ["PFT", "lat", "lon"],
            "shape": [14, 720, 1440],
            "fortran_rank": 3,
            "caller": "src_sechiba/slowproc.f90::slowproc_readvegetmax",
            "source_lines": {"variable": 3367, "owner_call": 3393},
            "active_owner": "interpweight_3D",
            "preprocessed_source": ".config/ppsrc/sechiba/slowproc.f90",
            "preprocessed_lines": {"variable": 3378, "owner_call": 3404},
            "batch_b_disposition": "outside_this_owner_pair; no interpweight_2D or interpweight_2Dcont dispatch",
        },
        "boundary_microcases": boundaries,
    }
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": False,
        "required_arm_count": None,
        "covered_arm_count": 0,
        "microcases": list(MICROCASES),
        "boundary_microcases": boundaries,
        "remaining_closure": "Rank-2 executable owner harness and numerical comparison remain required. Rank-3 boundary cases are conditional for this owner pair, but the paper PFT14 vegetation input dispatches to interpweight_3D, not either Batch B owner.",
    }
    comparison = {
        "schema_version": 2,
        "family": FAMILY,
        "status": "partial",
        "source_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "procedure_span_sha256": {key: value["span_sha256"] for key, value in extracted.items()},
        "tolerance": {"rtol": 1e-12, "atol": 1e-14, "discrete": "exact"},
        "microcases": list(MICROCASES),
        "boundary_comparisons": boundaries,
        "remaining_closure": coverage["remaining_closure"],
    }
    for name, document in (("comparison.json", comparison), ("branch_coverage.json", coverage), ("owner_region_evidence.json", owner_evidence)):
        (output_dir / name).write_text(json.dumps(document, indent=2) + "\n", encoding="ascii")
    return comparison


if __name__ == "__main__":
    print(json.dumps(run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY), indent=2))
