"""Independent DIM2 driver restart IO for the paper single-landpoint protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_orchidee.driver.dim2 import Dim2DriverRestart
from jax_orchidee.stomate.restart_io import (
    StomateRestartPhysicalState,
    StomateRestartWriteReport,
    create_stomate_restart_skeleton_from_schema,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRIVER_RESTART_SCHEMA = (
    ROOT / "docs" / "source_audits" / "driver_restart_netcdf_schema.json"
)

DRIVER_RESTART_VARIABLES = {
    "fluxsens": "fluxsens",
    "vevapp": "vevapp",
    "old_zlev": "zlev_old",
    "old_qair": "qair_old",
    "old_eair": "eair_old",
    "rau_old": "rau_old",
    "petAcoef": "petAcoef",
    "petBcoef": "petBcoef",
    "peqAcoef": "peqAcoef",
    "peqBcoef": "peqBcoef",
    "albedo_vis": "albedo_vis",
    "albedo_nir": "albedo_nir",
    "z0": "z0",
}


def read_dim2_driver_restart(path: str | Path) -> Dim2DriverRestart:
    """Read every DIM2 prognostic restart field in source order.

    Fortran provenance: ``dim2_driver.f90`` lines 722-796 and 1139-1157.
    """

    with Dataset(path) as dataset:
        missing = sorted(set(DRIVER_RESTART_VARIABLES.values()) - set(dataset.variables))
        if missing:
            raise KeyError(f"driver restart is missing variables: {missing}")
        values = {
            field: np.asarray(dataset.variables[name][0]).reshape(-1)
            for field, name in DRIVER_RESTART_VARIABLES.items()
        }
    return Dim2DriverRestart(**values)


def write_dim2_driver_restart(
    output_path: str | Path,
    *,
    state: Dim2DriverRestart,
    physical_state: StomateRestartPhysicalState,
    schema_path: str | Path = DEFAULT_DRIVER_RESTART_SCHEMA,
) -> StomateRestartWriteReport:
    """Construct and populate the complete DIM2 driver restart file.

    Fortran provenance: ``dim2_driver.f90`` lines 1411-1429.
    """

    if not isinstance(state, Dim2DriverRestart):
        raise TypeError("state must be Dim2DriverRestart")
    undefined = [field for field in DRIVER_RESTART_VARIABLES if getattr(state, field) is None]
    if undefined:
        raise ValueError(f"driver restart state is undefined for fields: {undefined}")
    output = create_stomate_restart_skeleton_from_schema(
        output_path,
        physical_state,
        schema_path=schema_path,
    )
    with Dataset(output, "r+") as dataset:
        for field, name in DRIVER_RESTART_VARIABLES.items():
            variable = dataset.variables[name]
            value = np.asarray(getattr(state, field))
            expected = int(np.prod(variable.shape[1:]))
            if value.shape != (expected,):
                raise ValueError(
                    f"{field} normalized shape must be {(expected,)}, got {value.shape}"
                )
            variable[0] = value.reshape(variable.shape[1:])
    return StomateRestartWriteReport(
        output_path=output,
        written_fields=tuple(DRIVER_RESTART_VARIABLES),
        validated_derived_fields=(),
        unsupported_fields=(),
    )
