"""Frozen experiment contract for the first formal neural process head."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "first_neural_process_head_candidate_v1"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "first_neural_process_head_candidate_v1.json"
)


def _contract_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FirstCandidateContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def candidate_id(self) -> str:
        return str(self.payload["candidate"]["id"])

    @property
    def control_id(self) -> str:
        return str(self.payload["control"]["id"])


def load_first_candidate_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> FirstCandidateContract:
    """Load and fail closed on experiment, data, budget, and gate drift."""

    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    observed = _contract_sha256(payload)
    if payload.get("contract_sha256") != observed:
        raise ValueError("first-candidate contract hash mismatch")

    data = payload.get("data", {})
    required_hashes = (
        "dataset_manifest_sha256",
        "acceptance_report_sha256",
        "training_statistics_json_sha256",
        "training_statistics_npz_sha256",
        "markov_contract_sha256",
        "typed_contract_sha256",
    )
    for name in required_hashes:
        value = str(data.get(name, ""))
        if len(value) != 64 or value.lower() != value or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"first-candidate data identity requires lowercase SHA256 {name}")
    if data.get("historical_v5_allowed") is not False or data.get("test_split_sealed") is not True:
        raise ValueError("v5 must be forbidden and the test split must remain sealed")

    candidate = payload.get("candidate", {})
    if candidate.get("family") != "hierarchical_axial_process_transformer":
        raise ValueError("the frozen candidate family has drifted")
    model_width = int(candidate.get("d_model", 0))
    heads = int(candidate.get("attention_heads", 0))
    if model_width < 1 or heads < 1 or model_width % heads:
        raise ValueError("candidate d_model must be divisible by its attention heads")
    minimum = int(candidate.get("parameter_count_minimum", 0))
    maximum = int(candidate.get("parameter_count_maximum", 0))
    if minimum < 1 or maximum < minimum:
        raise ValueError("candidate parameter interval is invalid")
    if candidate.get("global_all_token_attention") is not False:
        raise ValueError("global all-token attention is forbidden")

    control = payload.get("control", {})
    if control.get("same_forcing_transformer") is not True:
        raise ValueError("matched control must share the forcing Transformer")
    tolerance = float(control.get("maximum_parameter_count_relative_difference", -1.0))
    if not 0.0 <= tolerance <= 0.02:
        raise ValueError("matched-control parameter tolerance exceeds two percent")

    training = payload.get("training", {})
    if int(training.get("maximum_updates", 0)) != 10000:
        raise ValueError("the first candidate has a fixed 10,000-update ceiling")
    if tuple(training.get("seeds", ())) != (20260807, 20260808, 20260809):
        raise ValueError("the first-candidate seeds have drifted")
    if training.get("rollout_loss") is not False:
        raise ValueError("the first candidate may not use rollout loss")
    if training.get("bounded_target_loss_scale") != "unit_range":
        raise ValueError("bounded process targets must use their declared unit range")

    tiny_fit = payload.get("tiny_fit", {})
    if int(tiny_fit.get("batch_size", 0)) != 16 or int(tiny_fit.get("updates", 0)) != 100:
        raise ValueError("the tiny-fit batch and update budget have drifted")
    if float(tiny_fit.get("minimum_relative_loss_reduction", 0.0)) != 0.20:
        raise ValueError("the tiny-fit reduction gate has drifted")
    if tiny_fit.get("checkpoint_written") is not False:
        raise ValueError("the tiny fit may not retain a checkpoint")

    gates = payload.get("gates", {})
    if float(gates.get("budget_absolute_tolerance", 0.0)) != 2.0e-12:
        raise ValueError("the local budget gate has drifted")
    if float(gates.get("pft_permutation_atol", 0.0)) != 1.0e-10:
        raise ValueError("the PFT permutation absolute tolerance has drifted")
    if float(gates.get("pft_permutation_rtol", 0.0)) != 1.0e-12:
        raise ValueError("the PFT permutation relative tolerance has drifted")
    if float(gates.get("inactive_slot_isolation_atol", 0.0)) != 1.0e-10:
        raise ValueError("the inactive-slot isolation tolerance has drifted")
    if int(gates.get("hard_failure_count", -1)) != 0:
        raise ValueError("hard physical failures cannot be admitted")
    return FirstCandidateContract(payload=payload, source_path=source_path, sha256=observed)
