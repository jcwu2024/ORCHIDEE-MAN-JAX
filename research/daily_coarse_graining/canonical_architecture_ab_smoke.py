"""Real-shard GPU smoke for the frozen 669-point architecture A/B."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.canonical_architecture_ab_run import (
    _atomic_json,
    _sha256_file,
    load_architecture_ab_experiment,
)
from research.daily_coarse_graining.canonical_daily_model import parameter_count
from research.daily_coarse_graining.canonical_training import (
    fast_day_target_representation_from_contract,
    loss_weights_from_contract,
    model_config_from_batch,
    prepare_canonical_batch,
)
from research.daily_coarse_graining.canonical_training_run import (
    _adam_init,
    _make_train_step,
    load_contract_metadata,
)
from research.daily_coarse_graining.daily_model_architecture import (
    build_daily_model_definition,
)
from research.daily_coarse_graining.markov_dataset import (
    collate_samples,
    load_dataset_index,
    load_markov_shard,
    load_training_statistics,
)

SCHEMA_VERSION = "canonical_architecture_ab_real_shard_smoke_v1"


def run_smoke(
    *,
    experiment_path: str | Path,
    dataset_path: str | Path,
    statistics_path: str | Path,
    acceptance_path: str | Path,
    output_path: str | Path,
) -> Path:
    experiment = load_architecture_ab_experiment(experiment_path)
    dataset_path = Path(dataset_path).resolve()
    statistics_path = Path(statistics_path).resolve()
    acceptance_path = Path(acceptance_path).resolve()
    for name, path in (
        ("dataset_manifest", dataset_path),
        ("training_statistics", statistics_path),
        ("acceptance_report", acceptance_path),
    ):
        expected = experiment.raw["artifacts"][f"{name}_sha256"]
        if _sha256_file(path) != expected:
            raise ValueError(f"architecture smoke {name} hash drift")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed":
        raise ValueError("architecture smoke requires an accepted dataset")
    if acceptance["dataset_manifest"]["sha256"] != _sha256_file(dataset_path):
        raise ValueError("architecture smoke acceptance dataset hash drift")
    if acceptance["training_statistics"]["sha256"] != _sha256_file(
        statistics_path
    ):
        raise ValueError("architecture smoke acceptance statistics hash drift")

    index = load_dataset_index(dataset_path)
    reference = index.select(spatial_split="train", temporal_split="train")[0]
    if _sha256_file(reference.path) != reference.sha256:
        raise ValueError("architecture smoke shard hash drift")
    shard = load_markov_shard(reference.path)
    batch_size = int(experiment.raw["training"]["batch_size"])
    if shard.days < batch_size:
        raise ValueError("architecture smoke shard is smaller than the frozen batch")
    raw = collate_samples([shard.sample(day) for day in range(batch_size)])
    statistics = load_training_statistics(statistics_path, index=index)
    contract = load_contract_metadata(dataset_path)
    representation = fast_day_target_representation_from_contract(contract)
    batch = prepare_canonical_batch(raw, statistics, representation)
    config = model_config_from_batch(batch)
    weights = jnp.asarray(loss_weights_from_contract(contract))
    rate = jnp.asarray(
        experiment.raw["training"]["learning_rate"],
        dtype=jnp.float32,
    )

    arms: dict[str, Any] = {}
    for arm in experiment.raw["arms"]:
        definition = build_daily_model_definition(
            str(arm["model_architecture"]),
            config,
            contract,
        )
        parameters = definition.initialize(seed=int(experiment.raw["training"]["seed"]))
        optimizer = _adam_init(parameters)
        train_step = _make_train_step(definition.apply)
        started = time.perf_counter()
        parameters, optimizer, compile_loss = train_step(
            parameters,
            optimizer,
            batch,
            weights,
            rate,
        )
        jax.block_until_ready(parameters)
        compile_seconds = time.perf_counter() - started
        started = time.perf_counter()
        parameters, optimizer, hot_loss = train_step(
            parameters,
            optimizer,
            batch,
            weights,
            rate,
        )
        jax.block_until_ready(parameters)
        hot_seconds = time.perf_counter() - started
        leaves = jax.tree_util.tree_leaves(parameters)
        if not all(np.all(np.isfinite(np.asarray(value))) for value in leaves):
            raise ValueError(f"architecture smoke produced non-finite {arm['id']} parameters")
        arms[str(arm["id"])] = {
            "model_architecture": definition.identity(),
            "parameter_count": parameter_count(parameters),
            "compile_update_seconds": compile_seconds,
            "hot_update_seconds": hot_seconds,
            "compile_loss": float(compile_loss),
            "hot_loss": float(hot_loss),
            "finite_parameter_leaves": len(leaves),
        }

    ratio = arms["axis_process"]["parameter_count"] / arms["flat"]["parameter_count"]
    gates = experiment.raw["screening_gates"]
    if not (
        float(gates["parameter_ratio_min"])
        <= ratio
        <= float(gates["parameter_ratio_max"])
    ):
        raise ValueError("architecture smoke parameter budget gate failed")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "experiment_sha256": experiment.sha256,
        "sample": {
            "landpoint_id": reference.landpoint_id,
            "year": reference.year,
            "spatial_split": reference.spatial_split,
            "temporal_split": reference.temporal_split,
            "shard_sha256": reference.sha256,
            "batch_size": batch_size,
            "days": [0, batch_size - 1],
            "sealed_test_used": False,
        },
        "arms": arms,
        "parameter_ratio": ratio,
    }
    return _atomic_json(Path(output_path).resolve(), result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run_smoke(
        experiment_path=args.experiment,
        dataset_path=args.dataset,
        statistics_path=args.statistics,
        acceptance_path=args.acceptance,
        output_path=args.output,
    )
    print(json.dumps({"status": "passed", "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
