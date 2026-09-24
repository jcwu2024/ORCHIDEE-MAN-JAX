"""Bounded checks for same-expert 448-coordinate decoder grouping."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import shared_decoder_carbon_group
from scripts.hpc.audit_shared_decoder_cost import _load_static_layout
from scripts.hpc.audit_shared_decoder_packing import static_grouping_audit
from scripts.hpc.probe_shared_decoder_carbon_group import (
    AUDITED_FIELD_INDICES,
    AUDITED_FIELD_NAMES,
    AUDITED_SIGNATURE,
    DEFAULT_VARIANTS,
    audited_dynamic_indices,
    export_layout_metadata,
    fixture,
    run,
)


def test_audited_group_and_actual_projection_plan():
    group = next(
        group for group in static_grouping_audit(width=384)["groups"]
        if group["signature"] == AUDITED_SIGNATURE
    )
    members = tuple(member for member in group["members"] if member["expert"] == "litter_poc_doc")
    assert tuple(member["field_index"] for member in members) == AUDITED_FIELD_INDICES
    assert tuple(member["field"] for member in members) == AUDITED_FIELD_NAMES
    assert len(_load_static_layout()["layout"].source_family_names) == 20

    _shared, _payload, metadata = fixture(16, 448)
    assert all(item["axis_indices"].shape == (448, 4) for item in metadata)
    assert all(not np.asarray(item["dynamic_mask"])[..., -1].any() for item in metadata)
    assert not np.array_equal(metadata[0]["axis_indices"], metadata[1]["axis_indices"])


def test_exported_layout_does_not_call_static_builder(tmp_path, monkeypatch):
    path = tmp_path / "carbon-group-layout.json"
    export = export_layout_metadata(path)
    assert export["bytes"] > 0

    import scripts.hpc.audit_shared_decoder_cost as static_audit

    def unexpected_static_load():
        raise AssertionError("static builder must not run when layout metadata is supplied")

    monkeypatch.setattr(static_audit, "_load_static_layout", unexpected_static_load)
    audited_dynamic_indices.cache_clear()
    _shared, _payload, metadata = fixture(16, 448, layout_metadata_file=path)
    assert all(item["axis_indices"].shape == (448, 4) for item in metadata)
    assert all(not np.asarray(item["dynamic_mask"])[..., -1].any() for item in metadata)


@pytest.mark.parametrize("microbatch", [1, 4])
def test_padded_group_preserves_outputs_and_full_native_gradients(tmp_path, microbatch):
    args = SimpleNamespace(
        width=16,
        rows=73,
        microbatch=microbatch,
        variants=DEFAULT_VARIANTS,
        repeats=1,
        max_compiler_temp_gib=1.0,
        require_gpu=False,
        output=tmp_path / f"carbon-group-micro{microbatch}.json",
    )
    report = run(args, shared_decoder_carbon_group)
    assert report["status"] == "passed"
    assert report["numerical_parity_passed"]
    assert tuple(report["comparison"]) == DEFAULT_VARIANTS
    assert report["comparison"]["grouped256"]["loss_output"]["coordinates"] == 5 * 73 * microbatch + 1
    assert args.output.is_file()
