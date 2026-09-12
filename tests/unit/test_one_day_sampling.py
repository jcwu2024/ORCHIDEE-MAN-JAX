from collections import Counter

from research.daily_coarse_graining.daily_operator.one_day_sampling import (
    DayKey,
    OneDayArtifact,
    OneDaySampler,
    PreparedRowKey,
)


def _artifact(
    artifact_id: int | str,
    days: tuple[tuple[str, int, int], ...],
) -> OneDayArtifact:
    records = tuple(
        {
            "landpoint_id": landpoint_id,
            "year": year,
            "start_rows": (start_row,),
        }
        for landpoint_id, year, start_row in days
    )
    return OneDayArtifact.from_schedule_entry(
        artifact_id,
        {"components": {"one_day": records}},
    )


def _ordered_keys(sampler: OneDaySampler, epoch: int) -> tuple[PreparedRowKey, ...]:
    return tuple(
        row.prepared_row_key
        for batch in sampler.iter_epoch(epoch)
        for row in batch.rows
    )


def test_epoch_covers_every_exact_prepared_row_once_without_duplication():
    artifacts = tuple(
        _artifact(
            artifact_id,
            tuple((f"point-{artifact_id}-{row}", 1979, row + 2) for row in range(size)),
        )
        for artifact_id, size in (("a", 5), ("b", 3), ("c", 4))
    )
    sampler = OneDaySampler(
        artifacts,
        seed=20260912,
        batch_size=4,
        microbatch_size=2,
        max_artifacts_per_batch=2,
    )

    batches = tuple(sampler.iter_epoch(3))
    keys = tuple(row.prepared_row_key for batch in batches for row in batch.rows)
    expected = {
        row.prepared_row_key
        for artifact in artifacts
        for row in artifact.rows
    }

    assert len(keys) == sampler.draw_count == 12
    assert set(keys) == expected
    assert set(Counter(keys).values()) == {1}
    assert tuple(batch.batch_index for batch in batches) == tuple(range(len(batches)))
    expected_offset = 0
    for batch in batches:
        assert batch.draw_offset == expected_offset
        expected_offset += batch.draw_count
    assert expected_offset == sampler.draw_count
    assert batches[-1].epoch_unique_day_count_through_batch == sampler.unique_day_count


def test_seed_and_epoch_rederive_order_and_different_epochs_change_it():
    artifacts = tuple(
        _artifact(
            artifact_id,
            tuple((f"point-{artifact_id}-{row}", 1980, row + 2) for row in range(5)),
        )
        for artifact_id in (10, 20, 30, 40)
    )
    original = OneDaySampler(
        artifacts,
        seed=17,
        batch_size=7,
        microbatch_size=3,
        max_artifacts_per_batch=3,
    )
    reconstructed = OneDaySampler(
        tuple(reversed(artifacts)),
        seed=17,
        batch_size=7,
        microbatch_size=3,
        max_artifacts_per_batch=3,
    )

    epoch_four = _ordered_keys(original, 4)

    assert epoch_four == _ordered_keys(original, 4)
    assert epoch_four == _ordered_keys(reconstructed, 4)
    assert epoch_four != _ordered_keys(original, 5)


def test_artifact_mixing_is_bounded_and_remainders_are_not_padded():
    artifacts = tuple(
        _artifact(
            artifact_id,
            tuple((f"point-{artifact_id}-{row}", 1981, row + 2) for row in range(5)),
        )
        for artifact_id in ("a", "b", "c")
    )
    sampler = OneDaySampler(
        artifacts,
        seed=29,
        batch_size=6,
        microbatch_size=4,
        max_artifacts_per_batch=2,
    )

    batches = tuple(sampler.iter_epoch(0))

    assert tuple(batch.draw_count for batch in batches) == (6, 4, 5)
    assert all(len(batch.artifact_ids) <= 2 for batch in batches)
    first_ids = tuple(row.artifact_id for row in batches[0].rows)
    assert len(set(first_ids)) == 2
    assert first_ids[0::2] == (first_ids[0],) * 3
    assert first_ids[1::2] == (first_ids[1],) * 3
    assert tuple(
        tuple(microbatch.draw_count for microbatch in batch.microbatches)
        for batch in batches
    ) == ((4, 2), (4,), (4, 1))
    assert sum(batch.draw_count for batch in batches) == 15


def test_exact_row_and_semantic_day_exclusions_have_distinct_scopes():
    artifacts = (
        _artifact(
            "a",
            (
                ("shared-exact", 1982, 10),
                ("shared-day", 1982, 20),
                ("a-only", 1982, 30),
            ),
        ),
        _artifact(
            "b",
            (
                ("shared-exact", 1982, 10),
                ("shared-day", 1982, 20),
                ("b-only", 1982, 40),
            ),
        ),
    )
    sampler = OneDaySampler(
        artifacts,
        seed=31,
        batch_size=8,
        microbatch_size=2,
        max_artifacts_per_batch=2,
        excluded_rows=(PreparedRowKey("a", 0),),
        excluded_days=(DayKey("shared-day", 1982, 20),),
    )

    rows = tuple(row for batch in sampler.iter_epoch(0) for row in batch.rows)

    assert {row.prepared_row_key for row in rows} == {
        PreparedRowKey("a", 2),
        PreparedRowKey("b", 0),
        PreparedRowKey("b", 2),
    }
    assert PreparedRowKey("b", 0) in {row.prepared_row_key for row in rows}
    assert DayKey("shared-day", 1982, 20) not in {row.day_key for row in rows}
    assert sampler.draw_count == 3
    assert sampler.excluded_draw_count == 3


def test_schedule_flattening_preserves_record_and_start_row_order():
    entry = {
        "runtime_year": 1983,
        "components": {
            "one_day": (
                {
                    "landpoint_id": "point-a",
                    "year": 1983,
                    "start_rows": (11, 17),
                    "parent_shard_sha256": "parent-a",
                    "typed_shard_sha256": "typed-a",
                },
                {
                    "landpoint_id": "point-b",
                    "year": 1983,
                    "start_rows": (23,),
                    "parent_shard_sha256": "parent-b",
                    "typed_shard_sha256": "typed-b",
                },
            )
        },
    }

    artifact = OneDayArtifact.from_schedule_entry(
        1001,
        entry,
        prepared_day_indices=(101, 107, 203),
    )

    assert tuple(
        (
            row.batch_row,
            row.record_index,
            row.row_index_within_record,
            row.landpoint_id,
            row.year,
            row.start_row,
            row.prepared_day_index,
            row.parent_shard_sha256,
            row.typed_shard_sha256,
        )
        for row in artifact.rows
    ) == (
        (0, 0, 0, "point-a", 1983, 11, 101, "parent-a", "typed-a"),
        (1, 0, 1, "point-a", 1983, 17, 107, "parent-a", "typed-a"),
        (2, 1, 0, "point-b", 1983, 23, 203, "parent-b", "typed-b"),
    )
    without_indices = OneDayArtifact.from_schedule_entry(1001, entry)
    assert all(row.prepared_day_index is None for row in without_indices.rows)


def test_resume_rederives_the_exact_suffix_from_a_compact_cursor():
    artifacts = tuple(
        _artifact(
            artifact_id,
            tuple((f"point-{artifact_id}-{row}", 1984, row + 2) for row in range(3)),
        )
        for artifact_id in ("a", "b", "c", "d")
    )
    sampler = OneDaySampler(
        artifacts,
        seed=37,
        batch_size=4,
        microbatch_size=3,
        max_artifacts_per_batch=2,
    )
    full_epoch = tuple(sampler.iter_epoch(8))

    cursor = full_epoch[1].next_cursor

    assert tuple(sampler.resume(cursor)) == full_epoch[2:]
    assert tuple(sampler.resume(full_epoch[-1].next_cursor)) == ()


def test_metadata_reports_row_draws_separately_from_unique_days():
    artifacts = (
        _artifact("a", (("same-day", 1985, 42),)),
        _artifact("b", (("same-day", 1985, 42),)),
    )
    sampler = OneDaySampler(
        artifacts,
        seed=41,
        batch_size=8,
        microbatch_size=4,
        max_artifacts_per_batch=2,
    )

    pool = sampler.pool_metadata()
    batch = tuple(sampler.iter_epoch(2))[0]
    metadata = batch.as_metadata()

    assert pool["epoch_scope"] == "prepared_one_day_pool"
    assert pool["draw_count"] == 2
    assert pool["unique_day_count"] == 1
    assert metadata["actual_draw_count"] == 2
    assert metadata["unique_day_count"] == 1
    assert metadata["epoch_draw_count"] == 2
    assert metadata["epoch_unique_day_count"] == 1
    assert metadata["epoch_unique_day_count_through_batch"] == 1
    assert metadata["microbatches"] == [
        {
            "index": 0,
            "start": 0,
            "stop": 2,
            "draw_count": 2,
            "unique_day_count": 1,
            "artifact_ids": list(batch.artifact_ids),
        }
    ]
