"""Actual growth values through standalone restart IO and a next component.

Six cells retain their independent original-source six-cell reference identity.
Each cell is written separately only because the maintained paper restart
schema has identity scatter for one landpoint. No singleton oracle is claimed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.driver import orchestration  # noqa: E402
from jax_orchidee.driver.restart_export import (  # noqa: E402
    stomate_full_restart_states_from_day_end_packet,
    write_stomate_restart_from_day_end_packet,
)
from jax_orchidee.parameters.pft_catalog import build_pft_run_layout, load_pft_catalog  # noqa: E402
from jax_orchidee.stomate import integration  # noqa: E402
from jax_orchidee.stomate.restart_io import (  # noqa: E402
    StomateRestartPhysicalState,
    read_stomate_readstart_states_from_template,
)
from scripts.dev import oracle_phase06_growth_arithmetic as growth  # noqa: E402
from scripts.dev.oracle_phase05_daily_memory import compare as numeric_compare  # noqa: E402
from scripts.dev.oracle_phase05_daily_memory import encode, observe, sha  # noqa: E402

OUT = (
    ROOT
    / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/scientific-restart"
)
REFERENCE = ROOT / "tests/fixtures/phase06_growth_arithmetic_reference.json"
COMPONENT = "slowproc_stomate_previous_step_state"
ENTRY_PRODUCERS = {
    "biomass": "turnover.biomass",
    "leaf_age": "vmax.leaf_age",
    "leaf_frac": "vmax.leaf_frac",
    "age": "turnover.age",
    "sla_calc": "npp.sla_calc",
    "pft_present": "prescribe.pft_present",
    "everywhere": "prescribe.everywhere",
    "ind": "gap.ind",
    "adapted": "constraints.adapted",
    "regenerate": "constraints.regenerate",
    "turnover_time": "turnover.turnover_time",
    "senescence": "turnover.senescence",
    "when_growthinit": "phenology.when_growthinit",
    "npp_daily": "npp.npp",
    "turnover_daily": "turnover.turnover",
    "resp_maint": "npp.resp_maint",
    "resp_growth": "npp.resp_growth",
    "bm_to_litter": "final.bm_to_litter",
}
NEXT_FROM_ENTRY = {
    "prescribe_inputs": ("biomass", "leaf_frac", "pft_present", "everywhere", "ind", "when_growthinit"),
    "constraints_inputs": ("adapted", "regenerate"),
    "phenology_inputs": ("leaf_age",),
    "alloc_inputs": ("leaf_age", "age", "senescence", "sla_calc"),
    "post_npp_inputs": ("age", "senescence", "turnover_time", "sla_calc"),
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)

    def metadata_default(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Unsupported evidence object: {type(obj).__name__}")

    path.write_text(json.dumps(encode(value), indent=2, default=metadata_default) + "\n", encoding="utf-8")


def compare(name, expected, actual):
    a, b = np.asarray(expected), np.asarray(actual)
    if a.dtype.kind == "b" or b.dtype.kind == "b":
        same_shape = a.shape == b.shape
        mismatch = int(np.count_nonzero(a != b)) if same_shape else 1
        return dict(
            name=name,
            shape=list(a.shape),
            passed=same_shape and mismatch == 0,
            failed_coordinates=mismatch,
            max_abs=int(mismatch > 0),
            gate="exact boolean",
        )
    return numeric_compare(name, a, b)


def references():
    fixture = json.loads(REFERENCE.read_text())
    first = fixture["cases"]["environment_partition"]
    second = fixture["cases"]["environment_partition_reentry"]
    x = growth.base.decode(first["inputs"])

    def expected(case):
        return {
            key: np.asarray(value["values"]).reshape(value["shape"], order="F")
            for key, value in case["expected"].items()
        }

    return fixture, x, expected(first), expected(second)


def create_seed(path):
    """Actual source missing/default branch, with explicit all-sentinel depth."""
    with Dataset(path, "w") as dataset:
        for name, size in (("time", 1), ("veget", 14), ("y", 1), ("x", 1)):
            dataset.createDimension(name, size)
        depth = dataset.createVariable("depth_deepsoil", "f8", ("time", "veget", "y", "x"))
        depth[:] = 999999.0


def physical(cell):
    return StomateRestartPhysicalState(
        nav_lon=np.array([[float(cell)]]),
        nav_lat=np.array([[10.0]]),
        nav_lev=np.array([1.0]),
        time=np.array([1.0]),
        time_steps=np.array([48.0]),
    )


def producer_packet(base, fields, cell):
    packet = {name: np.asarray(fields[source])[cell : cell + 1].copy() for name, source in ENTRY_PRODUCERS.items()}
    packet["pft_present"] = packet["pft_present"].astype(bool)
    packet["senescence"] = packet["senescence"].astype(bool)
    packet["gdd_midwinter"] = np.asarray(fields["phenology.gdd_midwinter"])[cell : cell + 1].copy()
    # The component does not produce the daily accumulator phase. Retain its
    # actual readstart defaults, with no claim of a natural day-end history.
    packet["daily_accumulators"] = {k: v for k, v in base.daily_state._asdict().items() if k != "provenance"}
    packet["gpp_daily"] = packet["daily_accumulators"]["gpp_daily"]
    return packet


def states_equal(checks, expected, actual, prefix):
    for group in ("entry_state", "season_state", "daily_state", "gas_state", "remainder_state"):
        a, b = getattr(expected, group), getattr(actual, group)
        for name in a._fields:
            if name == "provenance" or name.startswith("read_input_"):
                continue
            before, after = getattr(a, name), getattr(b, name)
            if before is None:
                checks.append(dict(name=f"{prefix}.{group}.{name}", passed=after is None))
            else:
                checks.append(compare(f"{prefix}.{group}.{name}", before, after))


def restored_carry(x, producer, entries, seasons):
    """Only persisted produced operands are replaced by mapper returns."""
    result = growth.base.carry_inputs(x, producer)
    joined = {
        name: np.concatenate([np.asarray(getattr(entry, name)) for entry in entries], axis=0)
        for name in ENTRY_PRODUCERS
    }
    for group, names in NEXT_FROM_ENTRY.items():
        for name in names:
            result[group][name] = joined[name]
    result["alloc_inputs"]["lai"] = np.asarray(growth.base.chain_fields(producer)["final.lai"])
    result["phenology_inputs"]["gdd_midwinter"] = np.concatenate(
        [np.asarray(season.gdd_midwinter) for season in seasons], axis=0
    )
    # LAI, cn_ind and sla_age1 have no direct restart variables in the source.
    # The next bounded component retains their declared source-local operands.
    # LAI is independently checked from restored biomass/SLA below.
    result["prescribe_inputs"]["firstcall"] = False
    result["prescribe_inputs"]["stomate_restart_none"] = False
    return result


def run(output_dir=OUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture, x, original, original_next = references()
    producer = growth.current(x, compiled=True)
    fields = growth.base.chain_fields(producer)
    checks = [compare("original_producer." + name, value, fields[name]) for name, value in original.items()]
    layout = build_pft_run_layout(
        load_pft_catalog(ROOT / "configs/pft_catalogs/orchidee_man_paper_250919.json"),
        layout_id="paper_250919_legacy14",
        fractions=[0.0] * 13 + [1.0],
    )
    seed = output_dir / "declared-source-default-seed.nc"
    if not seed.exists():
        create_seed(seed)
    entries, seasons, records = [], [], []
    for cell in range(6):
        base = read_stomate_readstart_states_from_template(seed, t2m=np.array([280.0]), nvm=14, nslm=11)
        assert base.report.source_undefined_fields == ()
        packet = producer_packet(base, fields, cell)
        merged, _ = stomate_full_restart_states_from_day_end_packet(packet, base_states=base)
        output = output_dir / f"scientific-cell-{cell}.nc"
        written, merge = write_stomate_restart_from_day_end_packet(
            output, packet, base_states=base, physical_state=physical(cell), pft_layout=layout
        )
        loaded = read_stomate_readstart_states_from_template(
            output, t2m=np.array([280.0]), nvm=14, nslm=11, source_pft_layout=layout, target_pft_layout=layout
        )
        states_equal(checks, merged, loaded, f"cell{cell}.actual_io")
        # Use the actual declared reader defaults as the mapper template, so
        # this check depends on its real overrides rather than equal aliases.
        slowproc = {**loaded.entry_state._asdict(), **loaded.season_state._asdict()}
        slowproc.pop("provenance", None)
        previous = orchestration.DriverPreviousStepStatePacket(
            tstep=48,
            fields_by_component={COMPONENT: slowproc},
            provenance_by_component={
                COMPONENT: (
                    "Actual compiled standalone growth return -> restart_export -> readstart; unproduced tail is declared source default.",
                )
            },
        )
        entry, gaps = orchestration._paper_later_day_stomate_entry_state_from_previous(
            template=base.entry_state, previous_state=previous
        )
        season, season_gaps = orchestration._paper_later_day_stomate_season_state_from_previous(
            template=base.season_state, previous_state=previous
        )
        assert entry is not None and season is not None and not gaps and not season_gaps
        entries.append(entry)
        seasons.append(season)
        for target, source in ENTRY_PRODUCERS.items():
            checks.append(
                compare(f"cell{cell}.mapper.{target}", fields[source][cell : cell + 1], getattr(entry, target))
            )
        checks.append(
            compare(
                f"cell{cell}.mapper.gdd_midwinter",
                fields["phenology.gdd_midwinter"][cell : cell + 1],
                season.gdd_midwinter,
            )
        )
        records.append(
            dict(
                cell=cell,
                output=output.name,
                file_sha256=sha(output),
                write_report=written,
                merge_report=merge,
                actual_readstart_report=loaded.report,
                actual_source_local_flags={
                    name: getattr(loaded.remainder_state, name)
                    for name in ("read_input_depth_organic_soil", "read_input_thawed_humidity")
                },
                declared_seed_report=base.report,
                produced_packet=packet,
                mapper_gaps=gaps,
                season_mapper_gaps=season_gaps,
            )
        )
    next_x = restored_carry(x, producer, entries, seasons)
    # Recalculate LAI from the actual serialized biomass/SLA using the real
    # source-backed setlai component; no stored endpoint is substituted.
    restored_biomass = np.concatenate([np.asarray(entry.biomass) for entry in entries], axis=0)
    restored_sla = np.concatenate([np.asarray(entry.sla_calc) for entry in entries], axis=0)
    from jax_orchidee.stomate.carbon_kernels import set_lai_from_biomass

    lai = set_lai_from_biomass(restored_biomass, restored_sla)
    checks.append(compare("actual_setlai_from_restart", fields["final.lai"], lai))
    next_x["alloc_inputs"]["lai"] = lai
    independent_carry = growth.base.carry_from_fields(x, original)
    for group, values in independent_carry.items():
        for name, value in values.items():
            if isinstance(value, np.ndarray):
                checks.append(compare("restored_next_operand." + group + "." + name, value, next_x[group][name]))
    with observe(
        integration, ("prescribe_step", "constraints_step", "phenology_step", "allocation_step", "turnover_step")
    ) as calls:
        eager = growth.current(next_x)
    public = growth.current(next_x, compiled=True)
    for mode, result in (("eager", eager), ("public_compiled", public)):
        next_fields = growth.base.chain_fields(result)
        for name, expected in original_next.items():
            checks.append(compare("original_next." + mode + "." + name, expected, next_fields[name]))
    for owner, group in (
        ("prescribe_step", "prescribe_inputs"),
        ("constraints_step", "constraints_inputs"),
        ("phenology_step", "phenology_inputs"),
        ("allocation_step", "alloc_inputs"),
        ("turnover_step", "post_npp_inputs"),
    ):
        operands = calls[owner][0][0]
        for name in NEXT_FROM_ENTRY.get(group, ()):
            if name in operands:
                expected_operand = (
                    original_next["npp." + name]
                    if owner == "turnover_step" and name in ("age", "sla_calc")
                    else next_x[group][name]
                )
                checks.append(compare("actual_next_consumer." + owner + "." + name, expected_operand, operands[name]))
    result = dict(
        schema_version=1,
        status="passed" if all(c["passed"] for c in checks) else "failed_preserved",
        stable_process_id="PROC:restart.active.day_and_year_handoff",
        checks=checks,
        cell_records=records,
        independent_reference=dict(
            path=REFERENCE.relative_to(ROOT).as_posix(),
            sha256=sha(REFERENCE),
            cases=["environment_partition", "environment_partition_reentry"],
            input_order="C",
            reference_order="F",
            source_identities=fixture["source_identities"],
        ),
        complete_original_input=growth.base.encode(x),
        complete_restored_next_input=growth.base.encode(next_x),
        original_producer_fields=original,
        actual_producer_fields=fields,
        original_next_fields=original_next,
        actual_next_fields=growth.base.chain_fields(public),
        observed_actual_next_calls=calls,
        source_default_policy="Missing-variable reader branches retained individually; depth_deepsoil is explicit rank4 all999999 -> source1535-1538 zero branch. No None filled silently.",
        source_local_operands=dict(
            cn_ind="prescribe return; no direct source restart variable",
            sla_age1="NPP return; no direct source restart variable",
            lai="real setlai from restored biomass/SLA",
        ),
        limit="Bounded standalone numerical join, unchanged prescribed next forcing/config/nonproduced operands. Not a natural restart, full coupled day, annual trajectory or fresh year-transition scientific acceptance.",
    )
    save(output_dir / "execution.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(
        json.dumps(
            dict(
                status=result["status"],
                checks=len(result["checks"]),
                failures=[c for c in result["checks"] if not c["passed"]],
            ),
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
