"""Run-definition materialization helpers for paper-case runtime inputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping

from jax_orchidee.driver.init import parse_run_def

YEAR_DYNAMIC_KEYS = frozenset(
    {
        "ATM_CO2",
        "FORCING_FILE",
        "RESTART_FILEIN",
        "SECHIBA_restart_in",
        "STOMATE_RESTART_FILEIN",
    }
)

JOB_PROTOCOL_OVERRIDES = {
    "IMPOSE_VEG": "y",
    "LAND_COVER_CHANGE": "n",
    "FIRE_DISABLE": "y",
    "SECHIBA_VEGMAX__00001": "0.0",
    "SECHIBA_VEGMAX__00002": "0.0",
    "SECHIBA_VEGMAX__00003": "0.0",
    "SECHIBA_VEGMAX__00004": "0.0",
    "SECHIBA_VEGMAX__00005": "0.0",
    "SECHIBA_VEGMAX__00006": "0.0",
    "SECHIBA_VEGMAX__00007": "0.0",
    "SECHIBA_VEGMAX__00008": "0.0",
    "SECHIBA_VEGMAX__00009": "0.0",
    "SECHIBA_VEGMAX__00010": "0.0",
    "SECHIBA_VEGMAX__00011": "0.0",
    "SECHIBA_VEGMAX__00012": "0.0",
    "SECHIBA_VEGMAX__00013": "0.0",
    "SECHIBA_VEGMAX__00014": "1.0",
    "NSTM": "6",
    "NVM": "14",
    "PREF_SOIL_VEG__00014": "4",
    "PFT_TO_MTC__00014": "2",
    "VEGETATION_FILE": str(
        Path(
            os.environ.get(
                "ORCHIDEE_DATA_ROOT",
                str(Path(__file__).resolve().parents[2] / "data"),
            )
        )
        / "INPUTDIR_ZZ"
        / "PFT1860_mangr_025deg.nc"
    ).replace("\\", "/"),
    "RESTART_FILEIN": "NONE",
    "SECHIBA_restart_in": "NONE",
    "STOMATE_RESTART_FILEIN": "NONE",
}


def read_run_def_values(path: str | Path) -> dict[str, str]:
    """Read a run.def as normalized string values."""

    return {str(key): str(value).strip() for key, value in parse_run_def(path).items()}


def static_run_def_values(values: Mapping[str, str], *, excluded_keys: Iterable[str] = YEAR_DYNAMIC_KEYS) -> dict[str, str]:
    """Drop year-dynamic keys from archived case run.def values.

    Fortran/run provenance: ``Job0_bio`` lines 325-335 rewrites
    ``FORCING_FILE`` and ``ATM_CO2`` inside the yearly loop, and restart names
    are also year-mode dependent. Archived case ``run.def`` files can contain
    the final loop value, so these keys cannot be used as static case truth.
    """

    excluded = {str(key) for key in excluded_keys}
    return {str(key): str(value).strip() for key, value in values.items() if key not in excluded}


def materialize_run_def_values(
    base_values: Mapping[str, str],
    *override_layers: Mapping[str, str],
    job_protocol_overrides: Mapping[str, str] = JOB_PROTOCOL_OVERRIDES,
) -> dict[str, str]:
    """Merge materialized defaults with case/static override layers."""

    merged: dict[str, str] = {}

    def apply_layer(layer: Mapping[str, str]) -> None:
        for raw_key, raw_value in layer.items():
            key = str(raw_key)
            merged[key] = str(raw_value).strip()

    apply_layer(base_values)
    for layer in override_layers:
        apply_layer(layer)
    apply_layer(job_protocol_overrides)
    return merged


def materialize_case_run_def_values(
    *,
    base_used_run_def: str | Path,
    case_run_def: str | Path,
    override_run_defs: Iterable[str | Path] = (),
    base_is_fortran_used_truth: bool = False,
) -> dict[str, str]:
    """Return a complete run.def for one independent paper mosaic case.

    The case ``run.def`` supplies static zoom and sensitivity values, while the
    base ``used_run.def`` supplies materialized Fortran defaults consumed by
    ``getin_p``. Year-dynamic values are intentionally excluded from the case
    layer so the runtime can select forcing, CO2, and restart mode per year.
    """

    base = read_run_def_values(base_used_run_def)
    overrides = [read_run_def_values(path) for path in override_run_defs]
    if base_is_fortran_used_truth:
        # A per-landpoint z1/used_run.def records the final getin_p values and
        # already contains domain and sensitivity parameters. Reapplying the
        # archived submission run.def can change resolved values (for example
        # TIDES) and is therefore forbidden on this path.
        return materialize_run_def_values(base, *overrides)
    case_static = static_run_def_values(read_run_def_values(case_run_def))
    return materialize_run_def_values(base, *overrides, case_static)


def write_materialized_run_def(
    values: Mapping[str, str],
    output: str | Path,
    *,
    header_lines: Iterable[str] = (),
) -> Path:
    """Write sorted materialized run.def values under a caller-chosen path."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(line) for line in header_lines]
    if lines:
        lines.append("")
    for key in sorted(values):
        lines.append(f"{key} = {values[key]}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
