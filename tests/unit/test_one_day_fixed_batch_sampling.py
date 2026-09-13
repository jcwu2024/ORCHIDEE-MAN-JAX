from collections import Counter

import pytest

from research.daily_coarse_graining.daily_operator.one_day_sampling import (
    FIXED_REPLAY_PAD_PACKING,
    OneDayArtifact,
    OneDaySampler,
    PreparedRowKey,
)


def _artifact(artifact_id, row_count):
    return OneDayArtifact.from_schedule_entry(
        artifact_id,
        {
            "components": {
                "one_day": tuple(
                    {
                        "landpoint_id": f"point-{artifact_id}",
                        "year": 2001,
                        "start_rows": (row + 2,),
                    }
                    for row in range(row_count)
                )
            }
        },
    )


def _fixed_sampler(*, seed=17):
    # 43 source rows gives the production-relevant 11-row terminal remainder
    # at batch size 32.  The 18-row artifacts exercise carry across boundaries.
    return OneDaySampler(
        (_artifact("a", 18), _artifact("b", 18), _artifact("c", 7)),
        seed=seed,
        batch_size=32,
        microbatch_size=4,
        max_artifacts_per_batch=3,
        batch_packing=FIXED_REPLAY_PAD_PACKING,
    )


def test_fixed_replay_pad_covers_every_source_row_once_and_reports_terminal_padding():
    sampler = _fixed_sampler()
    batches = tuple(sampler.iter_epoch(0))
    source_rows = tuple(
        row
        for batch in batches
        for row in batch.rows[: batch.source_draw_count]
    )
    expected = {
        PreparedRowKey(artifact.artifact_id, row.batch_row)
        for artifact in (_artifact("a", 18), _artifact("b", 18), _artifact("c", 7))
        for row in artifact.rows
    }

    assert tuple(batch.draw_count for batch in batches) == (32, 32)
    assert tuple(batch.source_draw_count for batch in batches) == (32, 11)
    assert tuple(batch.padding_draw_count for batch in batches) == (0, 21)
    assert len(source_rows) == sampler.source_draw_count == 43
    assert {row.prepared_row_key for row in source_rows} == expected
    assert set(Counter(row.prepared_row_key for row in source_rows).values()) == {1}
    assert sampler.draw_count == 64
    assert sampler.padding_draw_count == 21
    assert batches[-1].rows[11:] == tuple(
        batches[-1].rows[index % 11] for index in range(21)
    )

    pool = sampler.pool_metadata()
    assert pool["batch_packing"] == FIXED_REPLAY_PAD_PACKING
    assert pool["source_draw_count"] == 43
    assert pool["draw_count"] == 64
    assert pool["padding_draw_count"] == 21
    metadata = batches[-1].as_metadata()
    assert metadata["source_draw_count"] == 11
    assert metadata["padding_draw_count"] == 21
    assert metadata["epoch_source_draw_count"] == 43
    assert metadata["epoch_padding_draw_count"] == 21
    assert [item["draw_count"] for item in metadata["microbatches"]] == [4, 4, 4, 4, 4, 4, 4, 4]
    assert [item["source_draw_count"] for item in metadata["microbatches"]] == [4, 4, 3, 0, 0, 0, 0, 0]
    assert [item["padding_draw_count"] for item in metadata["microbatches"]] == [0, 0, 1, 4, 4, 4, 4, 4]


def test_fixed_packing_is_epoch_deterministic_and_cursor_resume_rederives_exact_suffix():
    original = _fixed_sampler(seed=31)
    reconstructed = _fixed_sampler(seed=31)
    first_epoch = tuple(original.iter_epoch(4))

    assert first_epoch == tuple(reconstructed.iter_epoch(4))
    assert first_epoch != tuple(original.iter_epoch(5))
    assert tuple(original.resume(first_epoch[0].next_cursor)) == first_epoch[1:]
    assert tuple(original.resume(first_epoch[-1].next_cursor)) == ()


def test_fixed_packing_preserves_package_io_bound_or_fails_closed():
    sampler = _fixed_sampler()
    for batch in sampler.iter_epoch(2):
        assert len(batch.artifact_ids) <= 3
        assert all(microbatch.draw_count == 4 for microbatch in batch.microbatches)

    impossible = OneDaySampler(
        tuple(_artifact(identifier, 1) for identifier in range(5)),
        seed=3,
        batch_size=4,
        microbatch_size=2,
        max_artifacts_per_batch=2,
        batch_packing=FIXED_REPLAY_PAD_PACKING,
    )
    with pytest.raises(ValueError, match="max_artifacts_per_batch"):
        tuple(impossible.iter_epoch(0))

    with pytest.raises(ValueError, match="divide batch_size"):
        OneDaySampler(
            (_artifact("only", 32),),
            seed=3,
            batch_size=32,
            microbatch_size=6,
            max_artifacts_per_batch=1,
            batch_packing=FIXED_REPLAY_PAD_PACKING,
        )
