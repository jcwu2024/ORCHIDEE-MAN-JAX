"""Train matched one-step targets after detached model-generated prefixes."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.canonical_multistep import (
    CanonicalMultistepSequence,
    canonical_pushforward_prefix,
)
from research.daily_coarse_graining.canonical_multistep_training_run import (
    _atomic_json,
    _atomic_pickle,
    _compiled_forcing_batch,
    _load_initial_parameters,
    _load_plan,
    _make_runtime,
    _make_train_step,
    _model_config,
    _sample_batch,
    _sequence,
    _sha256_file,
)
from research.daily_coarse_graining.canonical_teacher_reentry import (
    query_teacher_day,
    teacher_reentry_templates,
)
from research.daily_coarse_graining.canonical_training import (
    fast_day_target_representation_from_contract,
    loss_weights_from_contract,
)
from research.daily_coarse_graining.canonical_training_run import (
    CHECKPOINT_SCHEMA_VERSION,
    _adam_init,
    verify_training_acceptance,
)
from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
)
from research.daily_coarse_graining.markov_dataset import (
    load_dataset_index,
    load_training_statistics,
)

OBJECTIVE = "on_policy_pushforward_v1"


@dataclass(frozen=True)
class PrefixStage:
    prefix_days: int
    steps: int


def parse_prefix_curriculum(value: str) -> tuple[PrefixStage, ...]:
    """Parse detached-prefix stages such as ``0:16,1:16,3:16,7:16``."""

    stages = []
    for raw in value.split(","):
        try:
            prefix_raw, steps_raw = raw.split(":", 1)
            stage = PrefixStage(int(prefix_raw), int(steps_raw))
        except ValueError as error:
            raise ValueError("prefix curriculum must use DAYS:STEPS entries") from error
        if stage.prefix_days not in {0, 1, 3, 7, 14} or stage.steps < 1:
            raise ValueError(
                "prefix days must be 0, 1, 3, 7, or 14 with positive steps"
            )
        stages.append(stage)
    prefixes = tuple(stage.prefix_days for stage in stages)
    if not stages or prefixes != tuple(sorted(set(prefixes))):
        raise ValueError("prefix days must be unique and increasing")
    return tuple(stages)


def _slice_sequence(
    sequence: CanonicalMultistepSequence, start: int, stop: int
) -> CanonicalMultistepSequence:
    return jax.tree_util.tree_map(lambda value: value[:, start:stop], sequence)


def _make_prefix_step(*, statistics, representation, transition):
    def one(parameters, initial_state, initial_discrete_state, sequence):
        return canonical_pushforward_prefix(
            parameters,
            initial_state,
            initial_discrete_state,
            sequence,
            statistics=statistics,
            representation=representation,
            retained_tail_transition=transition,
        )

    def batch(parameters, initial_states, initial_discrete_states, sequences):
        return jax.vmap(one, in_axes=(None, 0, 0, 0))(
            parameters,
            initial_states,
            initial_discrete_states,
            sequences,
        )

    return jax.jit(batch)


def _matched_training_sequence(
    batch: Mapping[str, Any],
    compiled_forcing,
    *,
    offset: int,
    terminal_states,
    teacher_fast_targets,
    teacher_next_states,
) -> CanonicalMultistepSequence:
    """Build one matched final-step target at the model-visited state."""

    day = slice(offset, offset + 1)
    return CanonicalMultistepSequence(
        forcing_native=jnp.asarray(batch["forcing_native"][:, day]),
        parameters=jnp.asarray(batch["parameters"][:, day]),
        landpoint_static=jnp.asarray(batch["landpoint_static"][:, day]),
        annual_conditions=jnp.asarray(batch["annual_conditions"][:, day]),
        year=jnp.asarray(batch["year"][:, day]),
        day_index=jnp.asarray(batch["day_index"][:, day]),
        teacher_state=jnp.asarray(terminal_states[:, None, :]),
        teacher_fast_day_target=jnp.asarray(teacher_fast_targets[:, None, :]),
        teacher_next_state=jnp.asarray(teacher_next_states[:, None, :]),
        retained_tail_inputs=jax.tree_util.tree_map(
            lambda value: value[:, day], compiled_forcing
        ),
    )


def _teacher_targets(
    *,
    batch: Mapping[str, Any],
    offset: int,
    terminal_states: np.ndarray,
    terminal_discrete_states: Mapping[str, np.ndarray],
    config_path: Path,
    runtime,
    templates,
    contract,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray]]:
    fast_targets = []
    next_states = []
    next_discrete = []
    for index in range(terminal_states.shape[0]):
        result = query_teacher_day(
            config_path=config_path,
            context=runtime.context,
            templates=templates,
            continuous=terminal_states[index],
            discrete={
                name: np.asarray(values[index])
                for name, values in terminal_discrete_states.items()
            },
            contract=contract,
            year=int(batch["year"][index, offset]),
            day_index=int(batch["day_index"][index, offset]),
        )
        fast_targets.append(result.fast_day_target)
        next_states.append(result.next_state)
        next_discrete.append(result.next_discrete_state)
    discrete_names = tuple(sorted(next_discrete[0]))
    if any(tuple(sorted(value)) != discrete_names for value in next_discrete):
        raise ValueError("Teacher re-entry returned inconsistent discrete schemas")
    return (
        np.stack(fast_targets),
        np.stack(next_states),
        {
            name: np.stack([value[name] for value in next_discrete])
            for name in discrete_names
        },
    )


def train_pushforward(args: argparse.Namespace) -> Mapping[str, Any]:
    dataset = args.dataset.resolve()
    statistics_path = args.statistics.resolve()
    acceptance = args.acceptance.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError("pushforward output directory already exists")
    output_dir.mkdir(parents=True)
    verify_training_acceptance(acceptance, dataset, statistics_path)
    raw_manifest = json.loads(dataset.read_text(encoding="utf-8"))
    plan_path = (args.plan or Path(raw_manifest["plan"])).resolve()
    plan, entries = _load_plan(plan_path, str(raw_manifest["plan_sha256"]))
    index = load_dataset_index(dataset, verify_hashes=True)
    references = list(index.select(spatial_split="train", temporal_split="train"))
    if not references:
        raise ValueError("pushforward training requires train/train shards")
    metadata = raw_manifest["markov_contract"]
    contract = daily_markov_contract_from_metadata(metadata)
    representation = fast_day_target_representation_from_contract(metadata)
    statistics = load_training_statistics(statistics_path, index=index)
    model_config = _model_config(contract, representation)
    curriculum = parse_prefix_curriculum(args.prefix_curriculum)
    identity = {
        "dataset_id": index.dataset_id,
        "teacher_git_head": index.teacher_git_head,
        "markov_contract_sha256": index.contract_sha256,
        "statistics_sha256": _sha256_file(statistics_path),
        "acceptance_sha256": _sha256_file(acceptance),
        "model_config": model_config._asdict(),
        "training_objective": OBJECTIVE,
        "generation_plan_sha256": str(raw_manifest["plan_sha256"]),
        "prefix_curriculum": [stage.__dict__ for stage in curriculum],
        "gradient_policy": "detached_prefix_final_transition_only",
        "target_policy": "teacher_at_model_visited_terminal_state",
    }
    parameters = _load_initial_parameters(
        args.initialize_checkpoint.resolve(),
        identity=identity,
        config=model_config,
        seed=args.seed,
    )
    optimizer = _adam_init(parameters)
    fast_day_weights = jnp.asarray(loss_weights_from_contract(metadata))
    learning_rate = jnp.asarray(args.learning_rate, dtype=jnp.float32)
    rng = np.random.default_rng(args.seed)
    config_path = Path(plan["teacher_config"])
    runtimes = {}
    templates_by_landpoint = {}
    prefix_steps = {}
    train_steps = {}
    history = []

    for stage in curriculum:
        started = time.perf_counter()
        losses = []
        gradient_norms = []
        teacher_seconds = 0.0
        query_count = 0
        counts = defaultdict(int)
        order = rng.permutation(len(references))
        for step_index in range(stage.steps):
            if step_index and step_index % len(order) == 0:
                order = rng.permutation(len(references))
            reference = references[int(order[step_index % len(order)])]
            batch = _sample_batch(
                reference,
                horizon=stage.prefix_days + 1,
                batch_size=args.batch_size,
                rng=rng,
            )
            runtime = runtimes.get(reference.landpoint_id)
            if runtime is None:
                runtime = _make_runtime(
                    landpoint_id=reference.landpoint_id,
                    entry=entries[reference.landpoint_id],
                    config_path=config_path,
                    contract=contract,
                    batch=batch,
                )
                runtimes[reference.landpoint_id] = runtime
                templates_by_landpoint[reference.landpoint_id] = (
                    teacher_reentry_templates(runtime.context)
                )
            compiled_forcing = _compiled_forcing_batch(batch, contract, runtime.context)

            if stage.prefix_days == 0:
                terminal_states = np.asarray(batch["initial_state"])
                terminal_discrete = {
                    name: np.asarray(value)
                    for name, value in batch["initial_discrete_state"].items()
                }
                teacher_fast = np.asarray(batch["teacher_fast_day_target"][:, 0])
                teacher_next = np.asarray(batch["teacher_next_state"][:, 0])
                teacher_next_discrete = {
                    name: np.asarray(value[:, 0])
                    for name, value in batch["teacher_next_discrete_state"].items()
                }
            else:
                key = (reference.landpoint_id, stage.prefix_days)
                if key not in prefix_steps:
                    prefix_steps[key] = _make_prefix_step(
                        statistics=statistics,
                        representation=representation,
                        transition=runtime.transition,
                    )
                full_sequence = _sequence(batch, compiled_forcing)
                prefix_sequence = _slice_sequence(
                    full_sequence, 0, stage.prefix_days
                )
                terminal = jax.device_get(
                    prefix_steps[key](
                        parameters,
                        jnp.asarray(batch["initial_state"]),
                        jax.tree_util.tree_map(
                            jnp.asarray, batch["initial_discrete_state"]
                        ),
                        prefix_sequence,
                    )
                )
                terminal_states = np.asarray(terminal.continuous_state)
                terminal_discrete = {
                    name: np.asarray(value)
                    for name, value in terminal.discrete_state.items()
                }
                query_started = time.perf_counter()
                teacher_fast, teacher_next, teacher_next_discrete = _teacher_targets(
                    batch=batch,
                    offset=stage.prefix_days,
                    terminal_states=terminal_states,
                    terminal_discrete_states=terminal_discrete,
                    config_path=config_path,
                    runtime=runtime,
                    templates=templates_by_landpoint[reference.landpoint_id],
                    contract=contract,
                )
                teacher_seconds += time.perf_counter() - query_started
                query_count += args.batch_size

            if set(terminal_discrete) != set(teacher_next_discrete):
                raise ValueError("on-policy discrete schemas do not match")
            training_sequence = _matched_training_sequence(
                batch,
                compiled_forcing,
                offset=stage.prefix_days,
                terminal_states=terminal_states,
                teacher_fast_targets=teacher_fast,
                teacher_next_states=teacher_next,
            )
            if reference.landpoint_id not in train_steps:
                train_steps[reference.landpoint_id] = _make_train_step(
                    statistics=statistics,
                    representation=representation,
                    weights=fast_day_weights,
                    transition=runtime.transition,
                    state_loss_weight=args.state_loss_weight,
                    undefined_loss_weight=args.undefined_loss_weight,
                )
            parameters, optimizer, loss, gradient_norm = train_steps[
                reference.landpoint_id
            ](
                parameters,
                optimizer,
                jnp.asarray(terminal_states),
                jax.tree_util.tree_map(jnp.asarray, terminal_discrete),
                training_sequence,
                learning_rate,
            )
            losses.append(float(loss))
            gradient_norms.append(float(gradient_norm))
            counts[reference.landpoint_id] += 1

        jax.block_until_ready(parameters)
        record = {
            "prefix_days": stage.prefix_days,
            "steps": stage.steps,
            "batch_size": args.batch_size,
            "mean_loss": float(np.mean(losses)),
            "last_loss": losses[-1],
            "mean_gradient_norm": float(np.mean(gradient_norms)),
            "teacher_query_count": query_count,
            "teacher_query_seconds": teacher_seconds,
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
        "schema_version": "canonical_pushforward_training_report_v1",
        "status": "completed",
        "identity": identity,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "state_loss_weight": args.state_loss_weight,
        "undefined_loss_weight": args.undefined_loss_weight,
        "history": history,
        "compiled_landpoint_prefixes": [list(key) for key in sorted(prefix_steps)],
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
    parser.add_argument("--initialize-checkpoint", type=Path, required=True)
    parser.add_argument("--prefix-curriculum", default="0:16,1:16,3:16,7:16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--undefined-loss-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.batch_size < 1
        or args.learning_rate <= 0.0
        or args.state_loss_weight < 0.0
        or args.undefined_loss_weight < 0.0
    ):
        raise ValueError("batch size, learning rate, and loss weights are invalid")
    report = train_pushforward(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
