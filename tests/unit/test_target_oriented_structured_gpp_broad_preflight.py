from types import SimpleNamespace

from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_preflight import (
    FROZEN_CONTRACT_SHA256,
    build_broad_training_schedule,
    load_structured_gpp_broad_preflight_contract,
    reference_identity,
)


def _reference(landpoint: str, year: int, *, days: int = 30):
    parent = SimpleNamespace(
        landpoint_id=landpoint,
        year=year,
        sha256=f"parent-{landpoint}-{year}",
    )
    return SimpleNamespace(
        parent=parent,
        sha256=f"typed-{landpoint}-{year}",
        day_count=days,
    )


def test_structured_gpp_broad_preflight_contract_freezes_scope():
    contract = load_structured_gpp_broad_preflight_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.payload["data"]["unique_training_days"] == 1_040_000
    assert contract.payload["data"]["effective_day_batch_size"] == 208
    assert contract.payload["execution"]["paid_training_authorized"] is False


def test_structured_gpp_broad_schedule_is_unique_and_monitor_disjoint():
    references = tuple(
        _reference(landpoint, year)
        for landpoint in ("a", "b", "c", "d")
        for year in (1961, 1962)
    )
    monitor = {reference_identity(references[0])}
    exposed = {(*reference_identity(references[1]), 3)}
    first = build_broad_training_schedule(
        references,
        prior_exposed_days=exposed,
        monitor_point_years=monitor,
        seed=42,
        optimizer_updates=3,
        point_years_per_update=3,
        days_per_point_year=2,
    )
    second = build_broad_training_schedule(
        references,
        prior_exposed_days=exposed,
        monitor_point_years=monitor,
        seed=42,
        optimizer_updates=3,
        point_years_per_update=3,
        days_per_point_year=2,
    )
    assert first.sha256 == second.sha256
    assert first.unique_days == 18
    used = []
    for batch in first.batches:
        identities = [reference_identity(value.reference) for value in batch.selections]
        assert len({value[0] for value in identities}) == 3
        for selection, identity in zip(batch.selections, identities, strict=True):
            assert identity not in monitor
            used.extend((*identity, row) for row in selection.rows)
    assert len(used) == len(set(used)) == 18
    assert not set(used) & exposed


def test_structured_gpp_broad_schedule_launcher_is_free_cpu_zero_training():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "scripts/hpc/audit_target_oriented_structured_gpp_broad_schedule.sh"
    ).read_text(encoding="utf-8")
    assert "sbatch" not in launcher
    assert "test01|test02|test03|test04" in launcher
    assert "--broad-preflight-contract" in launcher
    assert "--coverage-report" in launcher
