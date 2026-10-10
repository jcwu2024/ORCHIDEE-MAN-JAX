import copy

import pytest

from scripts.hpc.shared_gpu_budget import (
    authorize_limit, check_funding, original_funding_snapshot,
    ORIGINAL_SCOPE, ORIGINAL_ROUTE, ORIGINAL_STAGE, ORIGINAL_ENTRY,
    SCHEDULE_EXTRA_SCOPE, SCHEDULE_EXTRA_ROUTE, SCHEDULE_EXTRA_STAGE,
)


def test_approved_increment_preserves_charges_and_sessions():
    data = {"limit_gpu_seconds": 18000, "sessions": [{"occupied_gpu_seconds": 16616.99}],
            "charged_gpu_seconds": 16616.99}
    original_sessions = copy.deepcopy(data["sessions"])
    assert authorize_limit(data, 25200, "User-approved additional free session") == 25200
    assert data["sessions"] == original_sessions
    assert data["charged_gpu_seconds"] == 16616.99
    assert data["budget_authorizations"][0]["increment_gpu_seconds"] == 7200
    authorize_limit(data, 25200, "Same approved ceiling")
    assert len(data["budget_authorizations"]) == 1


def test_missing_approval_does_not_mutate_ledger():
    data = {"limit_gpu_seconds": 18000, "sessions": []}
    before = copy.deepcopy(data)
    with pytest.raises(ValueError, match="approval note"):
        authorize_limit(data, 25200)
    assert data == before


def test_default_cannot_reset_previously_extended_ledger():
    data = {"limit_gpu_seconds": 25200, "sessions": []}
    with pytest.raises(ValueError, match="reset or reduced"):
        authorize_limit(data)


def original_ledger():
    sessions = [
        {"id": "old-a", "stage": "preflight", "status": "finished", "ended_epoch": 10,
         "reserved_gpu_seconds": 10000, "occupied_gpu_seconds": 10000},
        {"id": "old-b", "stage": "T7", "status": "finished", "ended_epoch": 20,
         "reserved_gpu_seconds": 7000, "occupied_gpu_seconds": 6643.442122494802},
        # The wrapper began fractionally before authorize_limit recorded the increment.
        {"id": "new-daily", "stage": "daily", "status": "finished", "started_epoch": 99.99,
         "ended_epoch": 200, "reserved_gpu_seconds": 7200, "occupied_gpu_seconds": 4094.435518713668},
        {"id": "new-replay", "stage": "daily", "status": "finished", "started_epoch": 220,
         "ended_epoch": 230, "reserved_gpu_seconds": 900, "occupied_gpu_seconds": 133.9524115435779},
    ]
    return {"limit_gpu_seconds": 25200, "session_limit_seconds": 7200, "sessions": sessions,
            "budget_authorizations": [{"previous_limit_gpu_seconds": 18000,
                "limit_gpu_seconds": 25200, "increment_gpu_seconds": 7200,
                "approval_note": "Actual user approval", "recorded_epoch": 100}]}


def funding_check(data, seconds=500, scope=ORIGINAL_SCOPE, stage=ORIGINAL_STAGE, cards=2):
    return check_funding(data, scope=scope, stage=stage, seconds=seconds, cards=cards,
                         command=["python", "/overlay/" + ORIGINAL_ENTRY])


def retry_row(identifier="retry-1", seconds=400):
    return {"id": identifier, "stage": ORIGINAL_STAGE, "funding_scope": ORIGINAL_SCOPE,
            "funding_route": ORIGINAL_ROUTE, "status": "reserved", "started_epoch": 300,
            "cards": 2, "session_cap_seconds": seconds, "reserved_gpu_seconds": 2 * seconds,
            "command": ["python", "/overlay/" + ORIGINAL_ENTRY]}


def test_original_boundary_preserves_daily_extension_and_approval_records():
    data = original_ledger()
    before = copy.deepcopy(data)
    snapshot = funding_check(data)
    assert snapshot["baseline_session_ids"] == ["old-a", "old-b"]
    assert snapshot["baseline_charged_gpu_seconds"] == pytest.approx(16643.442122494802)
    assert snapshot["original_remaining_gpu_seconds"] == pytest.approx(1356.557877505198)
    assert data == before


def test_retries_reserve_conservatively_then_settle_without_resetting_route_allowance():
    data = original_ledger()
    baseline_history = copy.deepcopy(data["sessions"])
    authorization = copy.deepcopy(data["budget_authorizations"])
    data["funding_scopes"] = {ORIGINAL_SCOPE: funding_check(data, seconds=400)}
    first = retry_row()
    data["sessions"].append(first)
    assert original_funding_snapshot(data)["route_charged_or_reserved_gpu_seconds"] == 800
    funding_check(data, seconds=100)  # Entire1000 allowance, while first is still unresolved.
    with pytest.raises(ValueError, match="route1000"):
        funding_check(data, seconds=101)
    first.update(status="finished", ended_epoch=350, occupied_gpu_seconds=180)
    funding_check(data, seconds=410)  # Settlement releases620, without resetting180.
    second = retry_row("retry-2", seconds=410)
    data["sessions"].append(second)
    assert original_funding_snapshot(data)["route_charged_or_reserved_gpu_seconds"] == 1000
    with pytest.raises(ValueError, match="route1000"):
        funding_check(data, seconds=20)
    second.update(status="finished", ended_epoch=390, occupied_gpu_seconds=60)
    assert original_funding_snapshot(data)["route_charged_or_reserved_gpu_seconds"] == 240
    assert data["sessions"][:4] == baseline_history
    assert data["budget_authorizations"] == authorization


@pytest.mark.parametrize("scope,stage", [(None, ORIGINAL_STAGE), (None, "daily"),
                                        (ORIGINAL_SCOPE, "changed-stage"), ("daily_extension", ORIGINAL_STAGE)])
def test_original_entry_cannot_bypass_route_guard_with_changed_stage_or_scope(scope, stage):
    with pytest.raises(ValueError, match="fixed original funding"):
        funding_check(original_ledger(), scope=scope, stage=stage)


@pytest.mark.parametrize("cards,seconds", [(1, 500), (2, 501)])
def test_original_route_rejects_extra_wall_time_or_changed_card_count(cards, seconds):
    with pytest.raises(ValueError, match="two cards"):
        funding_check(original_ledger(), cards=cards, seconds=seconds)


def test_original_ceiling_cannot_be_funded_by_existing_daily_extension():
    data = original_ledger()
    data["sessions"][1]["occupied_gpu_seconds"] = 7900  # Only100 original seconds remain.
    with pytest.raises(ValueError, match="original18000"):
        funding_check(data, seconds=60)


def test_unknown_source_or_changed_historical_baseline_is_rejected():
    data = original_ledger()
    data["funding_scopes"] = {ORIGINAL_SCOPE: funding_check(data)}
    data["sessions"][0]["occupied_gpu_seconds"] += 1
    with pytest.raises(ValueError, match="baseline identity changed"):
        funding_check(data)
    data = original_ledger()
    data["budget_authorizations"] = []
    with pytest.raises(ValueError, match="known increment"):
        funding_check(data)
    data = original_ledger()
    row = retry_row()
    row.update(stage="changed-stage", funding_scope="daily_extension")
    data["sessions"].append(row)
    with pytest.raises(ValueError, match="altered stage/scope"):
        funding_check(data)


def extra_check(data, seconds=900, scope=SCHEDULE_EXTRA_SCOPE, stage=SCHEDULE_EXTRA_STAGE, cards=2):
    return check_funding(data, scope=scope, stage=stage, seconds=seconds, cards=cards,
                         command=["python", "/overlay/" + ORIGINAL_ENTRY])


def authorized_extra_ledger():
    data = original_ledger()
    prior = retry_row()
    prior.update(status="finished", ended_epoch=350, occupied_gpu_seconds=800.8189226984978)
    data["sessions"].append(prior)
    data["funding_scopes"] = {ORIGINAL_SCOPE: original_funding_snapshot(data)}
    authorize_limit(data, 27000, "User explicitly approves the separate two-card900s schedule window")
    return data


def test_additional_schedule_needs_its_specific_new_authorization_even_with_global_headroom():
    with pytest.raises(ValueError, match="own25200"):
        extra_check(original_ledger())
    data = original_ledger()
    authorize_limit(data, 27001, "Incorrect increment")
    with pytest.raises(ValueError, match="own25200"):
        extra_check(data)


def test_extra_source_preserves_settled_original_and_daily_records_without_relabeling():
    data = authorized_extra_ledger()
    before = copy.deepcopy(data)
    snapshot = extra_check(data)
    assert snapshot["limit_gpu_seconds"] == 1800
    assert snapshot["authorization_source"]["increment_gpu_seconds"] == 1800
    assert snapshot["route_charged_or_reserved_gpu_seconds"] == 0
    assert original_funding_snapshot(data)["route_charged_or_reserved_gpu_seconds"] == pytest.approx(800.8189226984978)
    assert data == before


def test_extra_retry_reservation_settlement_never_draws_original_or_daily_balances():
    data = authorized_extra_ledger()
    original = original_funding_snapshot(data)
    prior_rows = copy.deepcopy(data["sessions"])
    data["funding_scopes"][SCHEDULE_EXTRA_SCOPE] = extra_check(data)
    row = {"id": "extra-1", "stage": SCHEDULE_EXTRA_STAGE, "funding_scope": SCHEDULE_EXTRA_SCOPE,
        "funding_route": SCHEDULE_EXTRA_ROUTE, "cards": 2, "session_cap_seconds": 900,
        "reserved_gpu_seconds": 1800, "status": "reserved",
        "command": ["python", "/overlay/" + ORIGINAL_ENTRY]}
    data["sessions"].append(row)
    with pytest.raises(ValueError, match="route1800"):
        extra_check(data, seconds=20)
    row.update(status="finished", occupied_gpu_seconds=400, ended_epoch=10**12)
    assert extra_check(data, seconds=700)["route_remaining_gpu_seconds"] == 1400
    with pytest.raises(ValueError, match="route1800"):
        extra_check(data, seconds=701)
    assert original_funding_snapshot(data) == original
    assert data["sessions"][:-1] == prior_rows


@pytest.mark.parametrize("scope,stage,cards,seconds", [
    (SCHEDULE_EXTRA_SCOPE, ORIGINAL_STAGE, 2, 900),
    (ORIGINAL_SCOPE, SCHEDULE_EXTRA_STAGE, 2, 500),
    (None, SCHEDULE_EXTRA_STAGE, 2, 900),
    (SCHEDULE_EXTRA_SCOPE, SCHEDULE_EXTRA_STAGE, 1, 900),
    (SCHEDULE_EXTRA_SCOPE, SCHEDULE_EXTRA_STAGE, 2, 901),
])
def test_additional_original_entry_cannot_bypass_source_stage_or_card_wall_caps(scope, stage, cards, seconds):
    with pytest.raises(ValueError):
        extra_check(authorized_extra_ledger(), scope=scope, stage=stage, cards=cards, seconds=seconds)
