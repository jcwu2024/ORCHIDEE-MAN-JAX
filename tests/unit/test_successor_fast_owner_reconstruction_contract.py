import copy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "manifests" / "coarse_graining" / "successor_fast_owner_reconstruction_v1.json"
DOCUMENT_PATH = (
    ROOT / "docs" / "research" / "daily_coarse_graining" / "successor_fast_owner_reconstruction_contract_v1.md"
)


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))












def test_only_rveget_has_a_dynamic_defined_head_and_pft_axis():
    payload = _contract()
    dynamic = [
        record for record in payload["leaves"] if record["defined_policy"] == "neural_dynamic_value_and_defined_logit"
    ]

    assert [record["key"] for record in dynamic] == ["diffuco_previous_step_state.rveget"]
    assert dynamic[0]["shape"] == [1, 2]
    assert dynamic[0]["pft_axis"] == 1
    assert dynamic[0]["pft_axis_policy"] == "compact_active_pft_axis"
    assert dynamic[0]["loss_mask_policy"].endswith("eligible_slot_mask_bce")
