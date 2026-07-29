"""Bounded 1/3/7-day training through the retained daily tail."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalModelConfig,
    canonical_model_apply,
    initialize_canonical_model,
    parameter_count,
)
from research.daily_coarse_graining.canonical_multistep import (
    CanonicalMultistepSequence,
    canonical_multistep_batch_loss,
)
from research.daily_coarse_graining.canonical_retained_tail import (
    bind_canonical_retained_tail_transition,
)
from research.daily_coarse_graining.canonical_rollout import (
    _packet_from_canonical_state,
)
from research.daily_coarse_graining.canonical_state_objective import (
    stabilized_state_delta_scale,
    state_process_weighting_from_contract,
)
from research.daily_coarse_graining.canonical_training import (
    fast_day_target_representation_from_contract,
    loss_weights_from_contract,
)
from research.daily_coarse_graining.canonical_training_run import (
    CHECKPOINT_SCHEMA_VERSION,
    _adam_init,
    _adam_update,
    verify_training_acceptance,
)
from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
    reconstruct_retained_tail_forcing_batch,
)
from research.daily_coarse_graining.daily_model_architecture import (
    CANONICAL_FLAT_V1,
    MODEL_ARCHITECTURES,
    STRUCTURED_PROCESS_FILM_V1,
    DailyModelDefinition,
    build_daily_model_definition,
    checkpoint_architecture_id,
    verify_checkpoint_architecture,
)
from research.daily_coarse_graining.markov_dataset import (
    MarkovShardRef,
    collate_windows,
    load_dataset_index,
    load_markov_shard,
    load_training_statistics,
)
from research.daily_coarse_graining.synthetic_operator_cost import (
    _tail_static_inputs,
)


@dataclass(frozen=True)
class CurriculumStage:
    horizon: int
    steps: int


@dataclass(frozen=True)
class LandpointRuntime:
    context: Any
    transition: Any
    static: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_curriculum(value: str) -> tuple[CurriculumStage, ...]:
    stages = []
    for raw in value.split(","):
        try:
            horizon_raw, steps_raw = raw.split(":", 1)
            stage = CurriculumStage(int(horizon_raw), int(steps_raw))
        except ValueError as error:
            raise ValueError("curriculum must use HORIZON:STEPS entries") from error
        if stage.horizon not in {1, 3, 7} or stage.steps < 1:
            raise ValueError("curriculum horizons must be 1, 3, or 7 with positive steps")
        stages.append(stage)
    if not stages or tuple(stage.horizon for stage in stages) != tuple(
        sorted({stage.horizon for stage in stages})
    ):
        raise ValueError("curriculum horizons must be unique and increasing")
    return tuple(stages)


def _load_plan(path: Path, expected_sha256: str) -> tuple[Mapping[str, Any], dict[str, Any]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if _canonical_json_sha256(plan) != expected_sha256:
        raise ValueError("generation plan hash does not match the dataset manifest")
    by_landpoint: dict[str, Any] = {}
    for entry in plan["entries"]:
        landpoint = str(entry["landpoint_id"])
        previous = by_landpoint.setdefault(landpoint, entry)
        for name in ("run_def", "reference_run_dir"):
            if str(previous[name]) != str(entry[name]):
                raise ValueError(f"plan {name} changes across years for {landpoint}")
    return plan, by_landpoint


@lru_cache(maxsize=12)
def _cached_shard(path: str):
    return load_markov_shard(Path(path))


def _model_config(contract, representation) -> CanonicalModelConfig:
    return CanonicalModelConfig(
        state_width=contract.continuous_state_width,
        forcing_width=contract.native_forcing.width,
        parameter_width=contract.static_conditions.parameter_width,
        landpoint_static_width=contract.static_conditions.landpoint_static_width,
        annual_condition_width=contract.static_conditions.annual_condition_width,
        fast_day_target_width=contract.fast_day_target_width,
        dynamic_undefined_width=len(representation.dynamic_undefined_indices),
    )


def _load_initial_parameters(
    path: Path | None,
    *,
    identity: Mapping[str, Any],
    config: CanonicalModelConfig,
    seed: int,
    model_definition: DailyModelDefinition | None = None,
):
    definition = model_definition
    if path is None:
        if definition is None:
            return initialize_canonical_model(config, seed=seed)
        return definition.initialize(seed=seed)
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported initialization checkpoint schema")
    observed = checkpoint.get("identity", {})
    for name in (
        "dataset_id",
        "teacher_git_head",
        "markov_contract_sha256",
        "statistics_sha256",
        "acceptance_sha256",
        "model_config",
    ):
        if observed.get(name) != identity[name]:
            raise ValueError(f"initialization checkpoint identity mismatch for {name}")
    source_parameters = jax.tree_util.tree_map(jnp.asarray, checkpoint["parameters"])
    if definition is None:
        if checkpoint_architecture_id(observed) != CANONICAL_FLAT_V1:
            raise ValueError("legacy training path requires a flat canonical checkpoint")
        return source_parameters
    source_architecture = checkpoint_architecture_id(observed)
    if source_architecture == definition.architecture_id:
        verify_checkpoint_architecture(observed, definition)
        return source_parameters
    if (
        source_architecture == CANONICAL_FLAT_V1
        and definition.architecture_id == STRUCTURED_PROCESS_FILM_V1
    ):
        return definition.initialize(
            seed=seed,
            canonical_parameters=source_parameters,
        )
    raise ValueError(
        "initialization checkpoint architecture cannot initialize requested model"
    )


def _atomic_pickle(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sample_batch(
    reference: MarkovShardRef,
    *,
    horizon: int,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    shard = _cached_shard(str(reference.path))
    choices = shard.days - horizon + 1
    if choices < 1:
        raise ValueError(f"shard {reference.path} is shorter than horizon {horizon}")
    starts = rng.integers(0, choices, size=batch_size)
    return collate_windows(tuple(shard.window(int(start), horizon) for start in starts))


def _compiled_forcing_batch(batch, contract, context):
    forcing_native = np.asarray(batch["forcing_native"])
    years = np.asarray(batch["year"])
    day_indices = np.asarray(batch["day_index"])
    if forcing_native.ndim < 3:
        raise ValueError("compiled forcing batch requires batch and horizon axes")
    batch_size, horizon = forcing_native.shape[:2]
    if years.shape != (batch_size, horizon) or day_indices.shape != (
        batch_size,
        horizon,
    ):
        raise ValueError("compiled forcing batch calendar shape drift")

    native_by_day = {}
    for batch_index in range(batch_size):
        for horizon_index in range(horizon):
            key = (
                int(years[batch_index, horizon_index]),
                int(day_indices[batch_index, horizon_index]),
            )
            native = forcing_native[batch_index, horizon_index]
            if key in native_by_day and not np.array_equal(native_by_day[key], native):
                raise ValueError(
                    "duplicate compiled forcing calendar key has different native data"
                )
            if key not in native_by_day:
                native_by_day[key] = native
    return reconstruct_retained_tail_forcing_batch(
        forcing_native,
        contract.native_forcing,
        context,
    )


def _sequence(batch, compiled_forcing) -> CanonicalMultistepSequence:
    return CanonicalMultistepSequence(
        forcing_native=jnp.asarray(batch["forcing_native"]),
        parameters=jnp.asarray(batch["parameters"]),
        landpoint_static=jnp.asarray(batch["landpoint_static"]),
        annual_conditions=jnp.asarray(batch["annual_conditions"]),
        year=jnp.asarray(batch["year"]),
        day_index=jnp.asarray(batch["day_index"]),
        teacher_state=jnp.asarray(batch["state_trajectory"][:, :-1]),
        teacher_fast_day_target=jnp.asarray(batch["teacher_fast_day_target"]),
        teacher_next_state=jnp.asarray(batch["teacher_next_state"]),
        retained_tail_inputs=compiled_forcing,
    )


def _make_runtime_and_compiled_forcing(
    *,
    landpoint_id: str,
    entry: Mapping[str, Any],
    config_path: Path,
    contract,
    batch,
    timing: dict[str, float] | None = None,
) -> tuple[LandpointRuntime, Any]:
    started = time.perf_counter()
    context = teacher.prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=entry["run_def"],
        reference_run_dir=entry["reference_run_dir"],
    )
    if timing is not None:
        timing["context"] = time.perf_counter() - started
    started = time.perf_counter()
    compiled_forcing = _compiled_forcing_batch(batch, contract, context)
    if timing is not None:
        timing["compiled_forcing"] = time.perf_counter() - started
    started = time.perf_counter()
    packet = _packet_from_canonical_state(
        batch["initial_state"][0],
        {name: value[0] for name, value in batch["initial_discrete_state"].items()},
        contract,
        tstep=47,
    )
    sample_forcing = jax.tree_util.tree_map(lambda value: value[0], compiled_forcing)
    static = _tail_static_inputs(
        context,
        packet,
        sample_forcing,
        first_day=False,
    )
    transition = bind_canonical_retained_tail_transition(
        config_path=config_path,
        context=context,
        contract=contract,
        static=static,
        runtime_year=1962,
    )
    if timing is not None:
        timing["runtime_static"] = time.perf_counter() - started
    del landpoint_id
    return (
        LandpointRuntime(context=context, transition=transition, static=static),
        compiled_forcing,
    )


def _make_runtime(
    *,
    landpoint_id: str,
    entry: Mapping[str, Any],
    config_path: Path,
    contract,
    batch,
) -> LandpointRuntime:
    runtime, _ = _make_runtime_and_compiled_forcing(
        landpoint_id=landpoint_id,
        entry=entry,
        config_path=config_path,
        contract=contract,
        batch=batch,
    )
    return runtime


def _make_train_step(
    *,
    statistics,
    representation,
    weights,
    transition,
    state_loss_weight: float,
    undefined_loss_weight: float,
    state_weights=None,
    state_delta_scale=None,
    state_increment_loss_weight: float = 0.0,
    model_apply=canonical_model_apply,
):
    def train_step(parameters, optimizer, initial_states, discrete_states, sequence, rate):
        def objective(value):
            return canonical_multistep_batch_loss(
                value,
                initial_states,
                discrete_states,
                sequence,
                statistics=statistics,
                representation=representation,
                fast_day_weights=weights,
                retained_tail_transition=transition,
                state_weights=state_weights,
                state_delta_scale=state_delta_scale,
                state_loss_weight=state_loss_weight,
                state_increment_loss_weight=state_increment_loss_weight,
                undefined_loss_weight=undefined_loss_weight,
                model_apply=model_apply,
            )

        loss, gradients = jax.value_and_grad(objective)(parameters)
        parameters, optimizer = _adam_update(
            parameters,
            gradients,
            optimizer,
            learning_rate=rate,
        )
        gradient_norm = jnp.sqrt(
            sum(jnp.sum(value * value) for value in jax.tree_util.tree_leaves(gradients))
        )
        return parameters, optimizer, loss, gradient_norm

    return jax.jit(train_step)


def train_multistep(args: argparse.Namespace) -> Mapping[str, Any]:
    if (
        args.model_architecture == STRUCTURED_PROCESS_FILM_V1
        and args.initialize_checkpoint is None
    ):
        raise ValueError(
            "structured_process_film_v1 requires --initialize-checkpoint so its "
            "zero-residual start preserves the accepted flat baseline"
        )
    dataset = args.dataset.resolve()
    statistics_path = args.statistics.resolve()
    acceptance = args.acceptance.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError("multistep output directory already exists")
    output_dir.mkdir(parents=True)
    verify_training_acceptance(acceptance, dataset, statistics_path)
    raw_manifest = json.loads(dataset.read_text(encoding="utf-8"))
    plan_path = (args.plan or Path(raw_manifest["plan"])).resolve()
    plan, entries = _load_plan(plan_path, str(raw_manifest["plan_sha256"]))
    index = load_dataset_index(dataset, verify_hashes=True)
    references = list(
        index.select(spatial_split="train", temporal_split="train")
    )
    if not references:
        raise ValueError("multistep training requires train/train shards")
    metadata = raw_manifest["markov_contract"]
    contract = daily_markov_contract_from_metadata(metadata)
    representation = fast_day_target_representation_from_contract(metadata)
    statistics = load_training_statistics(statistics_path, index=index)
    config = _model_config(contract, representation)
    model_definition = build_daily_model_definition(
        args.model_architecture,
        config,
        metadata,
    )
    curriculum = parse_curriculum(args.curriculum)
    if args.objective == "canonical_multistep_v1":
        state_weights = None
        state_delta_scale = None
        state_increment_loss_weight = 0.0
        state_objective = {
            "state_weighting": "uniform_per_continuous_state_value",
            "state_increment_loss_weight": 0.0,
        }
    else:
        process_weighting = state_process_weighting_from_contract(metadata)
        state_weights = jnp.asarray(process_weighting.weights)
        stabilized_scale, scale_audit = stabilized_state_delta_scale(
            statistics,
            floor_ratio=args.state_delta_floor_ratio,
        )
        state_delta_scale = jnp.asarray(stabilized_scale)
        state_increment_loss_weight = args.state_increment_loss_weight
        state_objective = {
            "state_weighting": process_weighting.metadata,
            "state_weighting_sha256": process_weighting.sha256,
            "state_delta_scale": scale_audit,
            "state_increment_loss_weight": state_increment_loss_weight,
        }
    identity = {
        "dataset_id": index.dataset_id,
        "teacher_git_head": index.teacher_git_head,
        "markov_contract_sha256": index.contract_sha256,
        "statistics_sha256": _sha256_file(statistics_path),
        "acceptance_sha256": _sha256_file(acceptance),
        "model_config": config._asdict(),
        "model_architecture": model_definition.identity(),
        "initialization_checkpoint_sha256": (
            None
            if args.initialize_checkpoint is None
            else _sha256_file(args.initialize_checkpoint.resolve())
        ),
        "training_objective": args.objective,
        "state_objective": state_objective,
        "generation_plan_sha256": str(raw_manifest["plan_sha256"]),
        "curriculum": [stage.__dict__ for stage in curriculum],
    }
    parameters = _load_initial_parameters(
        args.initialize_checkpoint.resolve() if args.initialize_checkpoint else None,
        identity=identity,
        config=config,
        seed=args.seed,
        model_definition=model_definition,
    )
    optimizer = _adam_init(parameters)
    weights = jnp.asarray(loss_weights_from_contract(metadata))
    rate = jnp.asarray(args.learning_rate, dtype=jnp.float32)
    rng = np.random.default_rng(args.seed)
    runtimes: dict[str, LandpointRuntime] = {}
    compiled_steps = {}
    history = []
    config_path = Path(plan["teacher_config"])

    for stage in curriculum:
        started = time.perf_counter()
        losses = []
        gradient_norms = []
        counts = defaultdict(int)
        order = rng.permutation(len(references))
        for step_index in range(stage.steps):
            if step_index and step_index % len(order) == 0:
                order = rng.permutation(len(references))
            reference = references[int(order[step_index % len(order)])]
            batch = _sample_batch(
                reference,
                horizon=stage.horizon,
                batch_size=args.batch_size,
                rng=rng,
            )
            runtime = runtimes.get(reference.landpoint_id)
            forcing = None
            if runtime is None:
                runtime, forcing = _make_runtime_and_compiled_forcing(
                    landpoint_id=reference.landpoint_id,
                    entry=entries[reference.landpoint_id],
                    config_path=config_path,
                    contract=contract,
                    batch=batch,
                )
                runtimes[reference.landpoint_id] = runtime
            if forcing is None:
                forcing = _compiled_forcing_batch(
                    batch,
                    contract,
                    runtime.context,
                )
            key = (reference.landpoint_id, stage.horizon)
            if key not in compiled_steps:
                compiled_steps[key] = _make_train_step(
                    statistics=statistics,
                    representation=representation,
                    weights=weights,
                    transition=runtime.transition,
                    state_loss_weight=args.state_loss_weight,
                    undefined_loss_weight=args.undefined_loss_weight,
                    state_weights=state_weights,
                    state_delta_scale=state_delta_scale,
                    state_increment_loss_weight=state_increment_loss_weight,
                    model_apply=model_definition.apply,
                )
            parameters, optimizer, loss, gradient_norm = compiled_steps[key](
                parameters,
                optimizer,
                jnp.asarray(batch["initial_state"]),
                jax.tree_util.tree_map(jnp.asarray, batch["initial_discrete_state"]),
                _sequence(batch, forcing),
                rate,
            )
            losses.append(float(loss))
            gradient_norms.append(float(gradient_norm))
            counts[reference.landpoint_id] += 1
        jax.block_until_ready(parameters)
        record = {
            "horizon": stage.horizon,
            "steps": stage.steps,
            "batch_size": args.batch_size,
            "mean_loss": float(np.mean(losses)),
            "last_loss": losses[-1],
            "mean_gradient_norm": float(np.mean(gradient_norms)),
            "seconds": time.perf_counter() - started,
            "landpoint_batches": dict(sorted(counts.items())),
        }
        history.append(record)
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "identity": identity,
            "parameters": jax.device_get(parameters),
            "optimizer": jax.device_get(optimizer),
            "completed_epochs": 0,
            "history": history,
        }
        _atomic_pickle(output_dir / "checkpoint.pkl", checkpoint)

    _atomic_pickle(output_dir / "best_checkpoint.pkl", checkpoint)
    report = {
        "schema_version": "canonical_multistep_training_report_v1",
        "status": "completed",
        "identity": identity,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "objective": args.objective,
        "model_architecture": model_definition.identity(),
        "parameter_count": parameter_count(parameters),
        "initialization_checkpoint": (
            None
            if args.initialize_checkpoint is None
            else {
                "path": str(args.initialize_checkpoint.resolve()),
                "sha256": _sha256_file(args.initialize_checkpoint.resolve()),
            }
        ),
        "state_objective": state_objective,
        "state_loss_weight": args.state_loss_weight,
        "state_increment_loss_weight": state_increment_loss_weight,
        "undefined_loss_weight": args.undefined_loss_weight,
        "history": history,
        "compiled_landpoint_horizons": [list(key) for key in sorted(compiled_steps)],
    }
    _atomic_json(output_dir / "training_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialize-checkpoint", type=Path)
    parser.add_argument("--curriculum", default="1:64,3:64,7:64")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--model-architecture",
        choices=MODEL_ARCHITECTURES,
        default=CANONICAL_FLAT_V1,
    )
    parser.add_argument(
        "--objective",
        choices=("canonical_multistep_v1", "process_increment_v2"),
        default="canonical_multistep_v1",
    )
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-increment-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-delta-floor-ratio", type=float, default=1.0e-3)
    parser.add_argument("--undefined-loss-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.batch_size < 1
        or args.learning_rate <= 0.0
        or args.state_loss_weight < 0.0
        or args.state_increment_loss_weight < 0.0
        or args.undefined_loss_weight < 0.0
        or not 0.0 < args.state_delta_floor_ratio <= 1.0
    ):
        raise ValueError("batch size, learning rate, and loss weights are invalid")
    report = train_multistep(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
