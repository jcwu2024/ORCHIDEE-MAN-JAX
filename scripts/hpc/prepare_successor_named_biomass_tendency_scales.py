"""Freeze train-only AGB/BGB tendency scales from existing prepared v6 batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.successor_small_fit_prepared import load_prepared_artifact
from scripts.hpc.successor_staged_prepared import (
    load_staged_prepared_manifest,
    stage_update_range,
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("asset_sha256", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _cpu_backend() -> tuple[Any, tuple[Any, ...]]:
    import jax

    devices = tuple(jax.devices("cpu"))
    if not devices or jax.default_backend() != "cpu":
        raise ValueError("biomass tendency scale preparation requires the CPU backend")
    return jax, devices


def _load_contract(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = str(payload.get("contract_sha256", ""))
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    observed = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        payload.get("schema_version") != "successor_named_biomass_tendency_v1"
        or payload.get("status") != "local_objective_foundation_no_training_authorized"
        or expected != observed
    ):
        raise ValueError("named biomass tendency contract identity drift")
    return payload


def _select_records(
    entries: Sequence[Mapping[str, Any]],
    *,
    point_year_count: int,
    seed: int,
) -> tuple[Mapping[str, Any], ...]:
    if point_year_count < 1:
        raise ValueError("biomass tendency scale sample count must be positive")
    ranked = sorted(
        entries,
        key=lambda record: (
            hashlib.sha256(
                (
                    f"{seed}|{record['update']}|{record['landpoint_id']}|"
                    f"{record['year']}|{record['sha256']}"
                ).encode()
            ).hexdigest(),
            int(record["update"]),
        ),
    )
    selected = []
    seen = set()
    for record in ranked:
        identity = (str(record["landpoint_id"]), int(record["year"]))
        if identity in seen:
            continue
        selected.append(record)
        seen.add(identity)
        if len(selected) == point_year_count:
            break
    if len(selected) != point_year_count:
        raise ValueError("prepared stage has too few unique point-years for scales")
    return tuple(sorted(selected, key=lambda record: int(record["update"])))


def _trajectory_assets(
    runtime: Mapping[str, Any],
    artifact: Any,
) -> tuple[Any, Any]:
    from research.daily_coarse_graining.daily_operator.successor_training import (
        extract_biomass_productivity_trajectories,
    )
    from research.daily_coarse_graining.markov_dataset import defined_numeric_mask

    jax = runtime["jax"]
    import jax.numpy as jnp

    batch = artifact.batch

    def expected_one(state, defined, gpp, gpp_defined, templates):
        return extract_biomass_productivity_trajectories(
            state,
            defined,
            daily_fast_gpp=gpp,
            daily_fast_gpp_defined=gpp_defined,
            pft_fraction=templates.pft.pft_fraction,
            active_pft_mask=templates.pft.active_pft_mask,
            contract=runtime["contract"],
        )

    expected = jax.vmap(expected_one)(
        batch.expected_next_state,
        batch.expected_next_state_defined,
        batch.expected_fast_gpp,
        batch.expected_fast_gpp_defined,
        batch.rollout_inputs,
    )
    initial_state = np.asarray(batch.rollout_inputs.continuous_state[:, :1])
    initial_defined = defined_numeric_mask(initial_state)

    def initial_one(state, defined, templates):
        fraction = templates.pft.pft_fraction[:1]
        active = templates.pft.active_pft_mask[:1]
        return extract_biomass_productivity_trajectories(
            state,
            defined,
            daily_fast_gpp=jnp.zeros_like(fraction, dtype=jnp.float64),
            daily_fast_gpp_defined=active,
            pft_fraction=fraction,
            active_pft_mask=active,
            contract=runtime["contract"],
        )

    initial = jax.vmap(initial_one)(
        initial_state,
        initial_defined,
        batch.rollout_inputs,
    )
    return jax.device_get(initial), jax.device_get(expected)


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    contract = _load_contract(args.tendency_contract)
    sampling = contract["scale_sampling"]
    runtime = _load_extended_runtime(args, backend_initializer=_cpu_backend)
    from research.daily_coarse_graining.daily_operator.long_horizon_objective import (
        MaskedTrajectory,
    )
    from research.daily_coarse_graining.daily_operator.named_biomass_tendency import (
        NAMED_BIOMASS_TENDENCY_FIELDS,
        biomass_stock_identity_residual,
        fit_named_biomass_tendency_scales,
        named_biomass_tendency_objective,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        STAGED_SCHEMA_VERSION,
        build_successor_small_fit_schedule,
        load_successor_small_fit_contract,
        verify_successor_small_fit_schedule,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        derive_long_horizon_scales,
    )

    runner = load_successor_small_fit_contract(args.runner_contract)
    if runner.payload["schema_version"] != STAGED_SCHEMA_VERSION:
        raise ValueError("biomass tendency scales require staged-v3")
    references = runtime["index"].select(
        spatial_split="train",
        temporal_split="train",
    )
    schedule = build_successor_small_fit_schedule(
        references,
        runner,
        updates=int(runner.sampling["maximum_updates_per_arm"]),
    )
    verify_successor_small_fit_schedule(schedule, references, runner)
    start_update, end_update = stage_update_range(
        runner.sampling["stage_milestones"],
        2000,
    )
    manifest = load_staged_prepared_manifest(
        args.prepared_manifest,
        staged_training_contract_sha256=runner.sha256,
        full_schedule_sha256=schedule["schedule_sha256"],
        start_update=start_update,
        end_update=end_update,
    )
    selected = _select_records(
        manifest["entries"],
        point_year_count=int(sampling["point_year_count"]),
        seed=int(sampling["selection_seed"]),
    )

    initial_values = {name: [] for name in NAMED_BIOMASS_TENDENCY_FIELDS}
    initial_defined = {name: [] for name in NAMED_BIOMASS_TENDENCY_FIELDS}
    teacher_values = {name: [] for name in NAMED_BIOMASS_TENDENCY_FIELDS}
    teacher_defined = {name: [] for name in NAMED_BIOMASS_TENDENCY_FIELDS}
    identity_maximum = 0.0
    selected_report = []
    for index, record in enumerate(selected, start=1):
        update = int(record["update"])
        entry = schedule["entries"][update - 1]
        artifact = load_prepared_artifact(
            args.prepared_manifest,
            record,
            schedule_entry=entry,
        )
        initial, teacher = _trajectory_assets(runtime, artifact)
        for trajectories in (initial, teacher):
            identity = biomass_stock_identity_residual(trajectories.stocks)
            values = np.asarray(identity.value)[np.asarray(identity.defined)]
            if values.size:
                identity_maximum = max(
                    identity_maximum,
                    float(np.max(np.abs(values))),
                )
        for name in NAMED_BIOMASS_TENDENCY_FIELDS:
            initial_values[name].append(np.asarray(initial.stocks[name].value)[:, 0])
            initial_defined[name].append(
                np.asarray(initial.stocks[name].defined)[:, 0]
            )
            teacher_values[name].append(
                np.swapaxes(np.asarray(teacher.stocks[name].value), 0, 1)
            )
            teacher_defined[name].append(
                np.swapaxes(np.asarray(teacher.stocks[name].defined), 0, 1)
            )
        selected_report.append(
            {
                "update": update,
                "landpoint_id": str(entry["landpoint_id"]),
                "year": int(entry["year"]),
                "artifact_sha256": str(record["sha256"]),
            }
        )
        print(
            json.dumps(
                {
                    "event": "biomass_tendency_scale_point_year",
                    "completed": index,
                    "total": len(selected),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    initial_stocks = {
        name: MaskedTrajectory(
            np.concatenate(initial_values[name], axis=0),
            np.concatenate(initial_defined[name], axis=0),
        )
        for name in NAMED_BIOMASS_TENDENCY_FIELDS
    }
    teacher_stocks = {
        name: MaskedTrajectory(
            np.concatenate(teacher_values[name], axis=1),
            np.concatenate(teacher_defined[name], axis=1),
        )
        for name in NAMED_BIOMASS_TENDENCY_FIELDS
    }
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    statistics = fit_named_biomass_tendency_scales(
        initial_stocks,
        teacher_stocks,
        stock_scales=long_scales.stocks,
        floor_ratio=float(contract["scale_policy"]["stock_relative_floor_ratio"]),
        minimum_scale=float(contract["scale_policy"]["minimum_scale"]),
        horizons=tuple(contract["horizons_days"]),
    )
    zero = named_biomass_tendency_objective(
        teacher_stocks,
        teacher_stocks,
        scales=statistics.scales,
        coefficients={"tendency_mse": 1.0, "tendency_bias_squared": 1.0},
        horizons=tuple(contract["horizons_days"]),
    )
    teacher_teacher_loss = float(np.asarray(zero.total))
    identity_tolerance = float(
        contract["preparation_gates"]["agb_plus_bgb_identity_tolerance"]
    )
    passed = teacher_teacher_loss == 0.0 and identity_maximum <= identity_tolerance
    report = {
        "schema_version": "successor_named_biomass_tendency_scale_asset_v1",
        "status": "passed" if passed else "failed",
        "git_head": _git_head(),
        "identity": {
            "tendency_contract_sha256": contract["contract_sha256"],
            "staged_training_contract_sha256": runner.sha256,
            "full_schedule_sha256": schedule["schedule_sha256"],
            "prepared_manifest_sha256": manifest["manifest_sha256"],
        },
        "selection": {
            "policy": sampling["selection_policy"],
            "seed": int(sampling["selection_seed"]),
            "point_year_count": len(selected),
            "rollout_windows_per_point_year": int(
                sampling["rollout_windows_per_point_year"]
            ),
            "total_rollout_windows": int(
                sampling["rollout_windows_per_point_year"]
            )
            * len(selected),
            "records": selected_report,
        },
        "scales": {
            str(horizon): dict(values)
            for horizon, values in statistics.scales.items()
        },
        "scale_audit": statistics.audit,
        "gates": {
            "teacher_teacher_loss": teacher_teacher_loss,
            "teacher_teacher_loss_exact_zero": teacher_teacher_loss == 0.0,
            "maximum_total_minus_agb_bgb_absolute_residual": identity_maximum,
            "agb_plus_bgb_identity_tolerance": identity_tolerance,
            "agb_plus_bgb_identity_passed": identity_maximum <= identity_tolerance,
        },
        "backend": runtime["jax"].default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "elapsed_seconds": time.perf_counter() - started,
        "training_updates": 0,
        "checkpoint_written": False,
        "recovery_written": False,
        "validation_values_evaluated": False,
        "sealed_test_used": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "next_gate": (
            "local_candidate_objective_gradient_preflight"
            if passed
            else "repair_named_biomass_tendency_scale_asset"
        ),
    }
    report["asset_sha256"] = _canonical_sha256(report)
    _atomic_json(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--tendency-contract", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
