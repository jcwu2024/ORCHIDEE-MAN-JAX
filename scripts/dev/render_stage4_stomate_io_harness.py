"""Render a full-ABI Fortran driver for ``stomate_io`` restart oracle work.

The generated source is an interface-faithful harness skeleton: it allocates
every original formal array, calls ``readstart`` and passes that exact state
packet to ``writerestart``. Restart-file opening remains an explicit real
IOIPSL initialization boundary and is deliberately not replaced here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INTERFACE = ROOT / "docs/source_audits/stage4_restart_data_oracle_interface.json"
DEFAULT_OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/stage4_restart_io" / "stage4_stomate_io_driver.f90"

TYPE = {"INTEGER": "integer(i_std)", "REAL": "real(r_std)", "LOGICAL": "logical"}


def _declaration(field: dict[str, object]) -> str:
    name = str(field["name"])
    kind = TYPE[str(field["fortran_type"])]
    dimensions = [str(value) for value in field["dimensions"]]
    if not dimensions:
        return f"  {kind} :: {name}"
    return f"  {kind}, allocatable :: {name}{'(' + ','.join(':' for _ in dimensions) + ')'}"


def _allocate(field: dict[str, object]) -> str | None:
    dimensions = [str(value) for value in field["dimensions"]]
    if not dimensions:
        return None
    return f"  allocate({field['name']}({','.join(dimensions)}))"


def _call(name: str, fields: list[dict[str, object]]) -> list[str]:
    arguments = [str(field["name"]) for field in fields]
    lines = [f"  call {name} &"]
    for start in range(0, len(arguments), 5):
        chunk = ", ".join(arguments[start:start + 5])
        suffix = ", &" if start + 5 < len(arguments) else ")"
        prefix = "       & (" if start == 0 else "       &  "
        lines.append(f"{prefix}{chunk}{suffix}")
    return lines


def render(document: dict[str, object]) -> str:
    interfaces = document["interfaces"]
    readstart = interfaces["readstart"]["arguments"]
    writerestart = interfaces["writerestart"]["arguments"]
    fields = {
        str(field["name"]).lower(): field
        for field in [*readstart, *writerestart]
        if str(field["name"]).lower() != "npts"
    }
    declarations = [_declaration(field) for _, field in sorted(fields.items())]
    allocations = [_allocate(field) for _, field in sorted(fields.items())]
    allocations = [line for line in allocations if line is not None]
    return "\n".join([
        "! Generated from stage4_restart_data_oracle_interface.json; do not hand-edit.",
        "program stage4_stomate_io_driver",
        "  use defprec",
        "  use stomate_io, only: readstart, writerestart",
        "  use stomate_data",
        "  use constantes",
        "  use constantes_soil",
        "  use control, only: control_initialize",
        "  use grid, only: grid_set_glo, grid_allocate_glo",
        "  use ioipsl, only: getin_name, restclo",
        "  use ioipslctrl, only: ioipslctrl_restini",
        "  use mod_orchidee_para",
        "  use mod_orchidee_para_var",
        "  use ioipsl_para",
        "  use pft_parameters",
        "  use vertical_soil",
        "  implicit none",
        "  integer(i_std), parameter :: npts = 1_i_std",
        "  integer(i_std) :: rest_id, rest_id_stom, itau_offset",
        "  real(r_std) :: date0_shifted",
        *declarations,
        "",
        "  call getin_name('run.def')",
        "  call Init_orchidee_para()",
        "  call control_initialize(1800._r_std)",
        "  call pft_parameters_main()",
        "  ! stomate.f90::stomate_init lines 8040-8042.",
        "  months_num = 360_i_std",
        "  call getin_p('MONTHS_NUM', months_num)",
        *allocations,
        "  index = 1_i_std",
        "  lalo = 0._r_std",
        "  resolution = 1._r_std",
        "  t2m = 273.15_r_std",
        "  iim_g = 1_i_std",
        "  jjm_g = 1_i_std",
        "  nbp_glo = 1_i_std",
        "  call grid_set_glo(iim_g, jjm_g, nbp_glo)",
        "  call grid_allocate_glo(4_i_std)",
        "  index_g = index",
        "  lon_g = 0._r_std",
        "  lat_g = 0._r_std",
        "  call Init_orchidee_data_para_driver(nbp_glo, index_g)",
        "  call init_ioipsl_para()",
        "  call ioipslctrl_restini(0_i_std, 0._r_std, 1800._r_std, rest_id, rest_id_stom, itau_offset, date0_shifted)",
        "  ! Exact handle handoff in stomate.f90::stomate_init line 8231.",
        "  rest_id_stomate = rest_id_stom",
        "",
        *(_call("readstart", readstart)),
        "",
        *(_call("writerestart", writerestart)),
        "  if (is_root_prc) call restclo()",
        "  write(*, '(A)') 'stage4 stomate_io full ABI call completed'",
        "end program stage4_stomate_io_driver",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", type=Path, default=INTERFACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    document = json.loads(args.interface.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(document), encoding="ascii")
    print(f"WROTE {args.output} readstart={len(document['interfaces']['readstart']['arguments'])} writerestart={len(document['interfaces']['writerestart']['arguments'])}")


if __name__ == "__main__":
    main()
