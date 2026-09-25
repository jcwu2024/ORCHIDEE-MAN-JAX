"""Field mapping for a bounded failed-coordinate report."""

import numpy as np
import pytest

from scripts.hpc.probe_shared_decoder_macro_packed import output_discrepancies


@pytest.mark.parametrize("batch", [1, 4])
def test_failed_coordinate_at_field_boundary_maps_to_next_field(batch):
    specs = ((0, 2), (5, 3), (1, 1))
    reference = np.zeros((batch, 6), dtype=np.float32)
    actual = reference.copy()
    actual[-1, 2] = 0.125
    result = output_discrepancies(
        actual[0] if batch == 1 else actual,
        reference[0] if batch == 1 else reference, specs,
    )
    assert result["failed"] == 1
    item = result["coordinates"][0]
    assert (item["batch_index"], item["field_index"], item["field_row"], item["expert_index"]) == (
        batch - 1, 1, 0, 5,
    )


def test_failure_report_is_bounded_but_counts_all_failures():
    result = output_discrepancies(np.ones(12), np.zeros(12), ((0, 12),), limit=3)
    assert result["failed"] == 12
    assert result["reported"] == 3
