"""Check the independent FP64 coordinate oracle against tiny FP32 execution."""

from __future__ import annotations

import json

import numpy as np

from research.daily_coarse_graining.daily_operator.shared_decoder_macro_packed import (
    reference_decode_fields,
)
from research.daily_coarse_graining.daily_operator.shared_transformer import SharedTransformerConfig
from scripts.hpc.probe_shared_decoder_macro_packed import FIELD_SPECS, fixture, prepare_fields
from scripts.hpc.shared_decoder_roundoff_oracle import coordinate_oracle


def test_oracle_matches_all_tiny_coordinates_and_reports_finite_stages():
    width = 8
    shared, payload, masks = fixture(width)
    fields, _ = prepare_fields(shared, payload, masks)
    config = SharedTransformerConfig(
        width=width, ffn_width=4 * width, coordinate_blocks=2, readout_width=width,
    )
    production = reference_decode_fields(shared, fields, config)
    saw_active = saw_masked = saw_local_zero = False

    for field_index, ((expert, rows, *_), field_mask, output) in enumerate(
        zip(FIELD_SPECS, masks, production, strict=True)
    ):
        assert int(output.shape[0]) == rows
        for row_index in range(rows):
            result = coordinate_oracle(
                shared, payload, field_mask, field_index, row_index, expert,
                heads=8, epsilon=config.normalization_epsilon,
            )
            expected = float(np.asarray(output[row_index]))
            np.testing.assert_allclose(result["output"], expected, rtol=2e-5, atol=3e-6)
            saw_active |= bool(field_mask["active"][row_index])
            saw_masked |= not bool(field_mask["active"][row_index])
            saw_local_zero |= (
                bool(field_mask["active"][row_index])
                and not bool(field_mask["dynamic"][row_index, 0, -1])
            )
            assert result["field_index"] == field_index
            assert result["row_index"] == row_index
            assert result["expert_index"] == expert
            assert result["output_active"] == bool(field_mask["active"][row_index])
            for stage in result["stages"].values():
                if isinstance(stage, dict):
                    assert stage["finite"]
                    assert all(np.isfinite(stage[key]) for key in ("min", "max", "l2"))
                    assert all(np.isfinite(stage["preview"]))
                elif isinstance(stage, list):
                    assert np.all(np.isfinite(np.asarray(stage, dtype=np.float64)))
                elif isinstance(stage, float):
                    assert np.isfinite(stage)
            assert "two-block decoder" in result["scientific_scope"]
            json.dumps(result)

    assert saw_active and saw_masked and saw_local_zero

