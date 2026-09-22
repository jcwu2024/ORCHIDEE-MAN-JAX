"""Bounded physical monitor for shared-Transformer recurrent learning.

The monitor is forward-only.  It reuses the public canonical rollout and the
source-owned science extractors, retains no full state trajectory in its
report, and applies physical integrity gates only.  The fixed TRAIN panel is a
learning diagnostic; the held-out development panel is not a sealed test and
does not inherit absolute accuracy gates from an older architecture.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

FIT_ARTIFACT_COUNT = 8
HORIZONS = (7, 30)
FIT_ROLE = "fixed_train_fit"
HELD_OUT_ROLE = "held_out_development"
HELD_OUT_SLICES = ("temporal", "spatial", "joint")


class SharedFitPool:
    """Expose window zero at T7/T30 while caching at most one TRAIN artifact."""

    def __init__(
        self,
        runtime: Mapping[str, Any],
        head: Any,
        prepared_manifest: str | Path,
        artifact_ids: Sequence[int],
    ) -> None:
        identifiers = tuple(int(value) for value in artifact_ids)
        if len(identifiers) != FIT_ARTIFACT_COUNT or len(set(identifiers)) != len(identifiers):
            raise ValueError("shared fit monitor requires eight unique TRAIN artifacts")
        self.runtime = runtime
        self.head = head
        self.prepared_manifest = Path(prepared_manifest).resolve()
        self.artifact_ids = identifiers
        self.rows = [
            {
                "artifact_id": identifier,
                "component": {7: "seven_day", 30: "thirty_day"}[horizon],
                "length": horizon,
                "horizon": horizon,
                "window_index": 0,
                "slice_id": FIT_ROLE,
                "spatial_split": "train",
                "temporal_split": "train",
            }
            for identifier in identifiers
            for horizon in HORIZONS
        ]
        self._cached_id: int | None = None
        self._cached_artifact: Any | None = None

    def _artifact(self, identifier: int) -> Any:
        if identifier not in self.artifact_ids:
            raise ValueError("fit-monitor row names an undeclared artifact")
        if self._cached_id != identifier:
            from scripts.hpc.shared_transformer_runtime import (
                _load_prepared_artifacts,
                validate_shared_artifact_metadata,
            )

            artifacts, _manifest = _load_prepared_artifacts(
                self.prepared_manifest, (identifier,)
            )
            validate_shared_artifact_metadata(
                self.runtime["index"], (identifier,), artifacts
            )
            self._cached_artifact = artifacts[0]
            self._cached_id = identifier
        return self._cached_artifact

    def case(self, jax: Any, row: Mapping[str, Any]) -> tuple[Any, ...]:
        """Return the common five-part prepared-window monitor boundary."""

        from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
            prepare_shared_window,
        )

        identifier = int(row["artifact_id"])
        horizon = int(row["length"])
        component = str(row["component"])
        expected_component = {7: "seven_day", 30: "thirty_day"}.get(horizon)
        if (
            expected_component != component
            or int(row.get("window_index", -1)) != 0
            or row.get("slice_id") != FIT_ROLE
        ):
            raise ValueError("fit-monitor row differs from the fixed window-zero contract")
        artifact = self._artifact(identifier)
        prepared = prepare_shared_window(
            self.head,
            getattr(artifact.batch, component),
            window_index=0,
            horizon=horizon,
        )
        del jax
        return (
            prepared.initial_continuous,
            prepared.initial_discrete,
            prepared.days,
            prepared.targets,
            prepared.loss_context.expected_discrete,
        )


def _validate_rows(
    rows: Sequence[Mapping[str, Any]], split_role: str
) -> tuple[Mapping[str, Any], ...]:
    selected = tuple(rows)
    if split_role not in (FIT_ROLE, HELD_OUT_ROLE):
        raise ValueError(f"split_role must be {FIT_ROLE!r} or {HELD_OUT_ROLE!r}")
    if not selected:
        raise ValueError("shared learning monitor panel cannot be empty")
    for row in selected:
        horizon = int(row["length"])
        if horizon not in HORIZONS:
            raise ValueError("shared learning monitor supports only T7 and T30")
        if row.get("component") != {7: "seven_day", 30: "thirty_day"}[horizon]:
            raise ValueError("monitor component/horizon mismatch")

    if split_role == FIT_ROLE:
        counts = {horizon: sum(int(row["length"]) == horizon for row in selected) for horizon in HORIZONS}
        identifiers = {int(row["artifact_id"]) for row in selected}
        pairs = {(int(row["artifact_id"]), int(row["length"])) for row in selected}
        if (
            len(selected) != 2 * FIT_ARTIFACT_COUNT
            or counts != {7: FIT_ARTIFACT_COUNT, 30: FIT_ARTIFACT_COUNT}
            or len(identifiers) != FIT_ARTIFACT_COUNT
            or len(pairs) != len(selected)
            or any(int(row.get("window_index", -1)) != 0 for row in selected)
        ):
            raise ValueError("fixed TRAIN fit panel must contain window zero at T7/T30 for eight artifacts")
    else:
        slice_ids = {str(row.get("slice_id")) for row in selected}
        if not slice_ids or not slice_ids <= set(HELD_OUT_SLICES):
            raise ValueError("held-out development rows must retain temporal/spatial/joint labels")
        counts = {
            (slice_id, horizon): sum(
                str(row.get("slice_id")) == slice_id and int(row["length"]) == horizon
                for row in selected
            )
            for slice_id in slice_ids
            for horizon in HORIZONS
        }
        if any(count != 8 for count in counts.values()):
            raise ValueError("each selected held-out development slice requires eight T7 and eight T30 windows")
    return selected


def _masked_sum(jnp: Any, left: Any, right: Any) -> Any:
    return type(left)(
        jnp.asarray(left.value) + jnp.asarray(right.value),
        jnp.asarray(left.defined, dtype=jnp.bool_) & jnp.asarray(right.defined, dtype=jnp.bool_),
    )


def _masked_difference(jnp: Any, left: Any, right: Any) -> Any:
    return type(left)(
        jnp.asarray(left.value) - jnp.asarray(right.value),
        jnp.asarray(left.defined, dtype=jnp.bool_) & jnp.asarray(right.defined, dtype=jnp.bool_),
    )


def _host_hard_counts(
    predicted_fluxes: Mapping[str, Any],
    expected_fluxes: Mapping[str, Any],
    predicted_stocks: Mapping[str, Any],
    expected_stocks: Mapping[str, Any],
    initial_stocks: Mapping[str, Any],
) -> dict[str, int]:
    counts = {
        "trajectory_prediction_nonfinite": 0,
        "trajectory_teacher_nonfinite": 0,
        "trajectory_mask_mismatch": 0,
        "trajectory_missing_valid_curve": 0,
        "shared_stock_negative": 0,
        "initial_stock_nonfinite": 0,
        "initial_stock_missing_valid": 0,
    }
    for key in predicted_fluxes:
        predicted, expected = predicted_fluxes[key], expected_fluxes[key]
        predicted_value = np.asarray(predicted.value)
        expected_value = np.asarray(expected.value)
        predicted_defined = np.asarray(predicted.defined, dtype=bool)
        expected_defined = np.asarray(expected.defined, dtype=bool)
        counts["trajectory_prediction_nonfinite"] += int(
            np.sum(predicted_defined & ~np.isfinite(predicted_value))
        )
        counts["trajectory_teacher_nonfinite"] += int(
            np.sum(expected_defined & ~np.isfinite(expected_value))
        )
        counts["trajectory_mask_mismatch"] += int(np.sum(predicted_defined != expected_defined))
        counts["trajectory_missing_valid_curve"] += int(not np.any(predicted_defined & expected_defined))
    for key, value in predicted_stocks.items():
        expected = expected_stocks[key]
        predicted_value = np.asarray(value.value)
        expected_value = np.asarray(expected.value)
        predicted_defined = np.asarray(value.defined, dtype=bool)
        expected_defined = np.asarray(expected.defined, dtype=bool)
        counts["trajectory_prediction_nonfinite"] += int(
            np.sum(predicted_defined & ~np.isfinite(predicted_value))
        )
        counts["trajectory_teacher_nonfinite"] += int(
            np.sum(expected_defined & ~np.isfinite(expected_value))
        )
        counts["trajectory_mask_mismatch"] += int(np.sum(predicted_defined != expected_defined))
        stock_value = np.asarray(value.value)
        stock_defined = np.asarray(value.defined, dtype=bool)
        counts["shared_stock_negative"] += int(
            np.sum(stock_defined & np.isfinite(stock_value) & (stock_value < -1.0e-12))
        )
        initial = initial_stocks[key]
        initial_defined = np.asarray(initial.defined, dtype=bool)[0]
        counts["initial_stock_nonfinite"] += int(
            np.sum(np.asarray(initial.defined, dtype=bool) & ~np.isfinite(np.asarray(initial.value)))
        )
        counts["initial_stock_missing_valid"] += int(not np.any(initial_defined))
        counts["trajectory_missing_valid_curve"] += int(
            not np.any(predicted_defined & expected_defined & initial_defined[None, ...])
        )
    return counts


def _identity_diagnostic(source: Any, derived: Any) -> dict[str, Any]:
    source_value = np.asarray(source.value)
    derived_value = np.asarray(derived.value)
    source_defined = np.asarray(source.defined, dtype=bool)
    derived_defined = np.asarray(derived.defined, dtype=bool)
    defined = source_defined & derived_defined
    finite = np.isfinite(source_value) & np.isfinite(derived_value)
    valid = defined & finite
    difference = np.where(valid, source_value - derived_value, 0.0)
    return {
        "unit": "gC m-2 day-1",
        "count": int(np.sum(valid)),
        "mask_mismatch": int(np.sum(source_defined != derived_defined)),
        "nonfinite_defined": int(np.sum(defined & ~finite)),
        "nonzero_residual_count": int(np.sum(valid & (difference != 0.0))),
        "maximum_absolute_error": float(np.max(np.abs(difference), initial=0.0)),
        "rmse": float(np.sqrt(np.sum(difference * difference) / max(int(np.sum(valid)), 1))),
        "classification": "diagnostic_only_not_a_physical_or_accuracy_gate",
    }


def _stock_change_arrays(
    predicted: Any, expected: Any, initial: Any
) -> dict[str, np.ndarray]:
    predicted_value = np.asarray(predicted.value)
    expected_value = np.asarray(expected.value)
    initial_value = np.asarray(initial.value)[0]
    initial_defined = np.asarray(initial.defined, dtype=bool)[0]
    common = (
        np.asarray(predicted.defined, dtype=bool)
        & np.asarray(expected.defined, dtype=bool)
        & initial_defined[None, ...]
    )
    predicted_trajectory = predicted_value - initial_value[None, ...]
    expected_trajectory = expected_value - initial_value[None, ...]
    predicted_previous = np.concatenate((initial_value[None, ...], predicted_value[:-1]), axis=0)
    expected_previous = np.concatenate((initial_value[None, ...], expected_value[:-1]), axis=0)
    change_defined = common.copy()
    if common.shape[0] > 1:
        change_defined[1:] &= common[:-1]
    return {
        "predicted_trajectory": predicted_trajectory,
        "teacher_trajectory": expected_trajectory,
        "predicted_daily_change": predicted_value - predicted_previous,
        "teacher_daily_change": expected_value - expected_previous,
        "defined": common,
        "daily_change_defined": change_defined,
    }


def _compact_curves(
    predicted_fluxes: Mapping[str, Any],
    expected_fluxes: Mapping[str, Any],
    predicted_stocks: Mapping[str, Any],
    expected_stocks: Mapping[str, Any],
    initial_stocks: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Mapping[str, np.ndarray]]]:
    fluxes = {}
    for key in predicted_fluxes:
        predicted = predicted_fluxes[key]
        expected = expected_fluxes[key]
        fluxes[key] = {
            "unit": "gC m-2 day-1",
            "predicted": np.asarray(predicted.value).tolist(),
            "teacher": np.asarray(expected.value).tolist(),
            "defined": (
                np.asarray(predicted.defined, dtype=bool)
                & np.asarray(expected.defined, dtype=bool)
            ).tolist(),
        }
    stocks = {}
    changes = {}
    for key in predicted_stocks:
        predicted = predicted_stocks[key]
        expected = expected_stocks[key]
        initial = initial_stocks[key]
        change = _stock_change_arrays(predicted, expected, initial)
        changes[key] = change
        stocks[key] = {
            "level_and_trajectory_unit": "gC m-2",
            "daily_change_unit": "gC m-2 day-1",
            "initial": np.asarray(initial.value)[0].tolist(),
            "initial_defined": np.asarray(initial.defined, dtype=bool)[0].tolist(),
            "predicted_level": np.asarray(predicted.value).tolist(),
            "teacher_level": np.asarray(expected.value).tolist(),
            "predicted_change_from_common_initial": change["predicted_trajectory"].tolist(),
            "teacher_change_from_common_initial": change["teacher_trajectory"].tolist(),
            "defined": change["defined"].tolist(),
            "predicted_daily_change": change["predicted_daily_change"].tolist(),
            "teacher_daily_change": change["teacher_daily_change"].tolist(),
            "daily_change_defined": change["daily_change_defined"].tolist(),
        }
    return {"fluxes": fluxes, "stocks": stocks}, changes


def _summary(records: Sequence[Mapping[str, Any]], summarize_windows: Any, summarize_errors: Any) -> dict[str, Any]:
    physical = summarize_windows([record["physical"] for record in records])
    for key in physical["stocks"]:
        changes = [record["changes"][key] for record in records]
        for label, predicted_key, teacher_key, mask_key in (
            (
                "trajectory_change_gC_m2",
                "predicted_trajectory",
                "teacher_trajectory",
                "defined",
            ),
            (
                "daily_change_gC_m2_day",
                "predicted_daily_change",
                "teacher_daily_change",
                "daily_change_defined",
            ),
        ):
            predicted = np.concatenate([value[predicted_key].reshape(-1) for value in changes])
            expected = np.concatenate([value[teacher_key].reshape(-1) for value in changes])
            defined = np.concatenate([value[mask_key].reshape(-1) for value in changes])
            physical["stocks"][key][label] = summarize_errors(predicted, expected, defined)
    rh = physical["fluxes"]["heterotrophic_respiration"]
    nonzero = int(rh["teacher_nonzero_totals"])
    return {
        "window_count": len(records),
        "physical": physical,
        "rh": {
            "teacher_nonzero_window_totals": nonzero,
            "zero_target_caveat": (
                "all defined Teacher RH totals are zero; RH errors are reported, but this "
                "panel does not demonstrate a learned nonzero RH response"
                if nonzero == 0
                else None
            ),
        },
    }


def build_shared_learning_monitor(
    runtime: Mapping[str, Any],
    head: Any,
    retained_template: Any,
    state_scales: Any,
    science_scales: Any,
    *,
    memory_limit_bytes: int = 24 * 2**30,
):
    """Build the canonical forward-only shared learning evaluator."""

    if int(memory_limit_bytes) <= 0:
        raise ValueError("shared learning monitor memory limit must be positive")
    jax = runtime["jax"]
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.canonical_rollout import make_canonical_step
    from research.daily_coarse_graining.daily_operator.forward_screen_metrics import (
        physical_window,
        summarize_errors,
        summarize_windows,
    )
    from research.daily_coarse_graining.daily_operator.recurrent_validation import physical_counts
    from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
    from research.daily_coarse_graining.daily_operator.shared_trajectory_loss import (
        FLUX_KEYS,
        STOCK_KEYS,
        shared_component_trajectories,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        extract_biomass_productivity_trajectories,
    )
    from scripts.hpc.probe_one_day_batch import _compiled_memory

    del science_scales  # Scales belong to training; this monitor reports physical units.

    def unused_loss(*_args: Any) -> Any:
        return jnp.asarray(0.0, dtype=jnp.float64)

    component = make_rollout_loss(
        jax,
        jnp,
        make_canonical_step(head, retained_template),
        unused_loss,
        limit_bytes=int(memory_limit_bytes),
    )
    forward = jax.jit(component.rollout)

    def extract(observation: Mapping[str, Any]) -> Mapping[str, Any]:
        shared = shared_component_trajectories(observation, head.contract)
        source = extract_biomass_productivity_trajectories(
            observation["state"],
            observation["defined"],
            daily_fast_gpp=observation["gpp"],
            daily_fast_gpp_defined=observation["gpp_defined"],
            pft_fraction=observation["fraction"],
            active_pft_mask=observation["active"],
            contract=head.contract,
        )
        autotrophic = _masked_sum(
            jnp,
            shared.fluxes["maintenance_respiration"],
            shared.fluxes["growth_respiration"],
        )
        identity_npp = _masked_difference(jnp, shared.fluxes["gpp"], autotrophic)
        return {
            "fluxes": {
                "gpp": shared.fluxes["gpp"],
                "npp": source.fluxes["npp"],
                "maintenance_respiration": shared.fluxes["maintenance_respiration"],
                "growth_respiration": shared.fluxes["growth_respiration"],
                "autotrophic_respiration": autotrophic,
                "heterotrophic_respiration": shared.fluxes["heterotrophic_respiration"],
            },
            "stocks": shared.stocks,
            "identity_npp": identity_npp,
        }

    extract = jax.jit(extract)
    hard_counts = jax.jit(
        lambda observations, targets, expected_discrete, templates: physical_counts(
            observations,
            targets,
            expected_discrete,
            templates,
            head.contract,
            state_scales,
        )
    )
    executables: dict[tuple[Any, ...], Any] = {}
    executable_metadata: dict[tuple[Any, ...], Mapping[str, Any]] = {}

    def argument_signature(arguments: tuple[Any, ...], horizon: int) -> tuple[Any, ...]:
        # eval_shape is available on the deployed JAX as well as local JAX.
        # Abstract the identity only: no model execution or device copies.
        abstract_arguments = jax.eval_shape(lambda values: values, arguments)
        leaves, tree = jax.tree_util.tree_flatten(abstract_arguments)
        descriptors = []
        for abstract in leaves:
            shape = tuple(int(value) for value in getattr(abstract, "shape", ()))
            dtype = str(abstract.dtype)
            weak_type = bool(getattr(abstract, "weak_type", False))
            descriptors.append((shape, dtype, weak_type))
        return (int(horizon), str(tree), tuple(descriptors))

    def compiled_for(arguments: tuple[Any, ...], horizon: int) -> tuple[Any, tuple[Any, ...]]:
        signature = argument_signature(arguments, horizon)
        if signature not in executables:
            began = time.perf_counter()
            executable = forward.lower(*arguments).compile()
            memory = _compiled_memory(executable)
            if memory["estimated_live_bytes"] > int(memory_limit_bytes):
                raise MemoryError("shared learning monitor forward exceeds its memory envelope")
            executables[signature] = executable
            executable_metadata[signature] = {
                "compile_seconds": time.perf_counter() - began,
                "compiled_memory": memory,
                "input_signature": {
                    "horizon": signature[0],
                    "leaf_count": len(signature[2]),
                    "sha256": hashlib.sha256(repr(signature).encode("utf-8")).hexdigest(),
                },
            }
        return executables[signature], signature

    def evaluate(
        parameters: Any,
        pool: Any,
        rows: Sequence[Mapping[str, Any]],
        *,
        panel_name: str,
        split_role: str,
    ) -> dict[str, Any]:
        selected = _validate_rows(rows, split_role)
        if not str(panel_name):
            raise ValueError("shared learning monitor panel_name cannot be empty")
        began = time.perf_counter()
        report: dict[str, Any] = {
            "schema_version": "shared_learning_monitor_v1",
            "panel": str(panel_name),
            "split_role": split_role,
            "physical_gate_passed": False,
            "scientific_acceptance": False,
            "accuracy_gates_applied": False,
            "windows": [],
            "horizons": {},
            "splits": {},
            "metadata": {
                "forward_only": True,
                "canonical_rollout": True,
                "public_ad_objective_modified": False,
                "full_state_trajectories_retained": False,
                "sealed_test_used": False,
                "panel_history": (
                    "held-out development data, separate from parameter updates but historically "
                    "available for method development"
                    if split_role == HELD_OUT_ROLE
                    else "fixed TRAIN fit diagnostic used only to establish bounded learning"
                ),
                "flux_units": "gC m-2 day-1",
                "stock_units": "gC m-2",
                "flux_keys": (
                    "gpp",
                    "npp",
                    "maintenance_respiration",
                    "growth_respiration",
                    "autotrophic_respiration",
                    "heterotrophic_respiration",
                ),
                "independent_training_flux_keys": tuple(FLUX_KEYS),
                "stock_keys": tuple(STOCK_KEYS),
                "stock_reference": "same actual day-zero stock and mask for prediction and Teacher",
                "npp_source": "source npp_daily extractor; GPP-RA is checked independently",
                "memory_limit_bytes": int(memory_limit_bytes),
            },
        }
        records: list[dict[str, Any]] = []

        for index, row in enumerate(selected, start=1):
            horizon = int(row["length"])
            split = str(row.get("slice_id", FIT_ROLE))
            print(
                f"shared_learning_monitor panel={panel_name} window={index}/{len(selected)} "
                f"split={split} horizon=T{horizon} stage=forward",
                flush=True,
            )
            initial, discrete, days, targets, expected_discrete = pool.case(jax, row)
            executable, executable_signature = compiled_for(
                (parameters, initial, discrete, days), horizon
            )
            forward_began = time.perf_counter()
            _, observations = executable(parameters, initial, discrete, days)
            jax.block_until_ready(observations)
            forward_seconds = time.perf_counter() - forward_began

            device_hard = hard_counts(observations, targets, expected_discrete, days[0])
            predicted = extract(observations)
            expected = extract(targets)
            first = jax.tree_util.tree_map(lambda value: value[0], days[0])
            assembled = head.input_assembler.assemble(first)
            initial_observation = {
                "state": jnp.asarray(initial)[None, ...],
                "defined": jnp.asarray(assembled.state_defined.other)[None, ...],
                "gpp": jnp.asarray(targets["gpp"][:1]),
                "gpp_defined": jnp.asarray(targets["gpp_defined"][:1]),
                "fraction": jnp.asarray(first.pft.pft_fraction)[None, ...],
                "active": jnp.asarray(first.pft.active_pft_mask)[None, ...],
            }
            initial_extracted = extract(initial_observation)
            jax.block_until_ready((device_hard, predicted, expected, initial_extracted))

            predicted_fluxes = predicted["fluxes"]
            expected_fluxes = expected["fluxes"]
            predicted_stocks = predicted["stocks"]
            expected_stocks = expected["stocks"]
            initial_stocks = initial_extracted["stocks"]
            hard = {key: int(np.asarray(value)) for key, value in device_hard.items()}
            hard.update(
                _host_hard_counts(
                    predicted_fluxes,
                    expected_fluxes,
                    predicted_stocks,
                    expected_stocks,
                    initial_stocks,
                )
            )
            predicted_identity = _identity_diagnostic(
                predicted_fluxes["npp"], predicted["identity_npp"]
            )
            teacher_identity = _identity_diagnostic(
                expected_fluxes["npp"], expected["identity_npp"]
            )

            physical = physical_window(
                predicted_fluxes,
                expected_fluxes,
                predicted_stocks,
                expected_stocks,
                initial_stocks,
            )
            curves, changes = _compact_curves(
                predicted_fluxes,
                expected_fluxes,
                predicted_stocks,
                expected_stocks,
                initial_stocks,
            )
            window = {
                "selection": dict(row),
                "split": split,
                "horizon": horizon,
                "prepared_day_indices": np.asarray(days[0].day_index).tolist(),
                "forward_seconds": forward_seconds,
                "hard": hard,
                "physical_gate_passed": not any(hard.values()),
                "physical_curves": curves,
                "source_npp_identity": {
                    "prediction": predicted_identity,
                    "teacher": teacher_identity,
                },
            }
            report["windows"].append(window)
            records.append(
                {
                    "horizon": horizon,
                    "split": split,
                    "physical": physical,
                    "changes": changes,
                    "executable_signature": executable_signature,
                }
            )
            print(
                f"shared_learning_monitor panel={panel_name} window={index}/{len(selected)} "
                f"split={split} horizon=T{horizon} stage=completed seconds={forward_seconds:.1f}",
                flush=True,
            )
            del observations, predicted, expected, initial_extracted
            del initial, discrete, days, targets, expected_discrete, first, assembled

        for horizon in HORIZONS:
            group = [record for record in records if record["horizon"] == horizon]
            if group:
                signatures = tuple(
                    dict.fromkeys(record["executable_signature"] for record in group)
                )
                compiled = [executable_metadata[signature] for signature in signatures]
                report["horizons"][str(horizon)] = {
                    "compile_seconds": sum(value["compile_seconds"] for value in compiled),
                    "compiled_memory": max(
                        (value["compiled_memory"] for value in compiled),
                        key=lambda value: value["estimated_live_bytes"],
                    ),
                    "executable_variants": compiled,
                    **_summary(group, summarize_windows, summarize_errors),
                }
        for split in sorted({record["split"] for record in records}):
            report["splits"][split] = {
                str(horizon): _summary(
                    [
                        record
                        for record in records
                        if record["split"] == split and record["horizon"] == horizon
                    ],
                    summarize_windows,
                    summarize_errors,
                )
                for horizon in HORIZONS
                if any(
                    record["split"] == split and record["horizon"] == horizon
                    for record in records
                )
            }
        report["physical_gate_passed"] = all(
            window["physical_gate_passed"] for window in report["windows"]
        )
        report["elapsed_seconds"] = time.perf_counter() - began
        return report

    return evaluate


__all__ = [
    "FIT_ROLE",
    "HELD_OUT_ROLE",
    "SharedFitPool",
    "build_shared_learning_monitor",
]
