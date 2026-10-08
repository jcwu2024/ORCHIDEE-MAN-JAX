"""Local contract/readiness checks with synthetic metadata only."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).with_name("collect_fixed1024_result.py")
SPEC = importlib.util.spec_from_file_location("fixed1024_collector_test", SOURCE)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)

CODE_SHA = "a" * 40
LANES = [{"landpoint_id": f"point-{index:02d}", "inputs_sha256": f"{index:064x}"}
         for index in range(64)]
CERTIFICATE = {
    "active": True,
    "contract": "exact-DIFFUCO-host-mask-cache-repair-v1",
    "source_path": "jax_orchidee/sechiba/diffuco.py",
    "original_file_sha256": "04fc1193e85d2deca0f97cc06008989053ccaf956db3dc9552dd6b9cc2549d22",
    "actual_file_sha256": "42073dd14f427db8ed78803c90cd5a64b677bf822d1edf6b6fbed7aa97166a14",
    "baseline_full_source_sha256": collector.SCIENCE_SHA,
    "baseline_checkout_bytes_sha256": collector.SCIENCE_SHA,
    "baseline_checkout_representation": "canonical Linux LF source bytes",
    "actual_full_source_sha256": collector.ACTUAL_SCIENCE_SHA,
    "compatibility_scope": "exact helper-only immutable host Boolean mask; all other source bytes unchanged",
    "claim": "audited execution-only equivalence; source bytes differ",
}


def make_identity():
    return {
        "contract": collector.CONTRACT,
        "candidate_shapes": [512, 1024],
        "total_lanes": 1024,
        "unique_real_points": 64,
        "lane_days": 14336,
        "warm_repeats": 5,
        "science_sha256": collector.SCIENCE_SHA,
        "actual_science_source_sha256": collector.ACTUAL_SCIENCE_SHA,
        "baseline_ref": "base-ref",
        "lanes": LANES,
        "precision_policy": "native dtypes",
        "historical_B512_evidence_sha256": collector.LEGACY_B512_SHA,
        "conditional_policy": "fixed1024_resource_only_2700s_2GiB_GPU_8GiB_host_slack",
        "lane_origins": [{"lane_index": index, "source_index": index % 64,
            "landpoint_id": LANES[index % 64]["landpoint_id"], "repeat_index": index // 64}
            for index in range(1024)],
        "numerical_reference_sha256": "r" * 64,
        "gate": dict(collector.GATE),
        "execution_source_compatibility": dict(CERTIFICATE),
    }


def make_admission():
    return {
        "source_identity": {"baseline_ref": "base-ref", "science_sha256": collector.SCIENCE_SHA,
            "lanes": LANES, "precision_policy": "native dtypes"},
        "FULL64_reference_origin": {"sha256": "r" * 64},
    }


def make_approval(**overrides):
    values = {"job_id": "999999", "job_name": "orchjax-gpu-cap1024", "node": "test-node",
        "gpu_uuid": "GPU-synthetic", "slurm_global_gpu_id": "2", "driver_version": "525.85.12"}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_identity_binds_contract_code_source_gate_and_full_repeated_origins():
    identity = make_identity()
    collector.identity_checks(identity, make_admission(), CODE_SHA)
    identity["lane_origins"][1023]["source_index"] = 1
    with pytest.raises(ValueError, match="cyclic 1024-lane"):
        collector.identity_checks(identity, make_admission(), CODE_SHA)


@pytest.mark.parametrize("field,value,message", [
    ("actual_science_source_sha256", "c" * 64, "Linux source identity"),
    ("historical_B512_evidence_sha256", "d" * 64, "sealed historical B512"),
    ("conditional_policy", "speedup_required", "resource-only policy"),
])
def test_identity_mutations_are_rejected(field, value, message):
    identity = make_identity()
    identity[field] = value
    with pytest.raises(ValueError, match=message):
        collector.identity_checks(identity, make_admission(), CODE_SHA)


def test_stage_names_encode_two_ordered_sizes_and_1024_distinct_reference_gates():
    for size in (512, 1024):
        references, internal = collector.stage_gate_names(size)
        assert len(references) == 1024
        assert len(internal) == 3
        assert len(references | internal) == 1027
        assert f"capacity_B{size}_lane_1023_source_63_GPU_reference" in references


def test_complete_workload_sample_requires_five_real_full_lane_day_observations():
    sample_values = [20.0, 21.0, 19.0, 22.0, 20.5]
    metric = {"samples_seconds": sample_values, "median_seconds": 20.5,
        "range_seconds": [19.0, 22.0], "lane_days": 14336,
        "unique_real_points": 64, "unique_point_days": 896,
        "unique_point_days_is_throughput_denominator": False,
        "lane_days_per_second": 14336 / 20.5}
    result = collector.validate_sample(metric, "synthetic")
    assert result["lane_days"] == 14336 and result["lane_days_per_second"] == 14336 / 20.5
    metric["lane_days"] = 7168
    with pytest.raises(ValueError, match="14336 lane-day"):
        collector.validate_sample(metric, "synthetic")


def test_checkpoint_io_receipt_must_cover_actual_sealed_bytes():
    io = {"available": True, "bytes": 1024, "serialization_seconds": 0.2,
        "atomic_write_seconds": 0.1}
    collector.validate_checkpoint_io(io, 1024)
    with pytest.raises(ValueError, match="byte count"):
        collector.validate_checkpoint_io(io, 1025)


def test_raw9_counts_only_one_terminal_parent_and_reports_failed_job_cost():
    approval = make_approval()
    raw = "|".join((approval.job_id, approval.job_name, "gnall", "FAILED", "3412", "4",
        "cpu=4,node=1,billing=4,gres/gpu=1", approval.node, "1:0"))
    child = "|".join((approval.job_id + ".batch", approval.job_name, "gnall", "FAILED", "3400", "4",
        "cpu=4,node=1,billing=4", approval.node, "1:0"))
    accounting = collector.parse_raw9(raw + "\n" + child, approval)
    assert accounting["state"] == "FAILED"
    assert accounting["parent_counted_once"] is True
    assert accounting["whole_job_cost"]["primary"]["GPU_card_hours"] == 3412 / 3600
    assert accounting["raw_parent_row"][0] == approval.job_id


def test_unfinished_raw9_cannot_promote_stages():
    approval = make_approval()
    raw = "|".join((approval.job_id, approval.job_name, "gnall", "RUNNING", "120", "4",
        "cpu=4,node=1,billing=4,gres/gpu=1", approval.node, "0:0"))
    with pytest.raises(ValueError):
        collector.parse_raw9(raw, approval)


def test_terminal_wall_overrun_preserves_actual_parent_cost_and_withholds_stages():
    approval = make_approval()
    raw = "|".join((approval.job_id, approval.job_name, "gnall", "TIMEOUT", "5401", "4",
        "cpu=4,node=1,billing=4,gres/gpu=1", approval.node, "0:0"))
    accounting = collector.parse_raw9(raw, approval)
    assert accounting["approved_wall_exceeded"] is True
    assert accounting["observed_wall_seconds"] == 5401
    assert accounting["whole_job_cost"]["primary"]["GPU_card_hours"] == 5401 / 3600
    outcome = {"valid_exploratory_stages": [{"stage": collector.STAGE_NAMES[0]}],
        "withheld_stages": [], "capacity_curve_available": True}
    collector.apply_parent_accounting(outcome, accounting)
    assert outcome["valid_exploratory_stages"] == []
    assert outcome["parent_successfully_completed"] is False
    assert outcome["accounting"]["parent_counted_once"] is True
    assert outcome["accounting"]["observed_wall_seconds"] == 5401


def test_raw9_requires_exactly_one_matching_parent():
    approval = make_approval()
    raw = "|".join((approval.job_id, approval.job_name, "gnall", "COMPLETED", "100", "4",
        "cpu=4,node=1,gres/gpu=1", approval.node, "0:0"))
    with pytest.raises(ValueError, match="exactly one"):
        collector.parse_raw9(raw + "\n" + raw, approval)


def test_B1024_resource_guard_has_no_performance_gate_and_uses_new_B512_memory_origin():
    historical = {"provenance": {"executable_memory_analysis": {"temp_size_in_bytes": 100}}}
    current = {"memory": {"executable_memory_analysis": {"temp_size_in_bytes": 200}}}
    decision = {"shape": 1024, "permitted": True, "minimum_remaining_seconds": 2700,
        "remaining_seconds": 2710, "requires_performance_improvement": False,
        "policy": "fixed1024_resource_only_2700s_2GiB_GPU_8GiB_host_slack",
        "estimated_device_bytes": 100, "free_device_bytes": 2 * 1024**3 + 100,
        "compiler_memory_origin": {"historical_B512_sha256": collector.LEGACY_B512_SHA,
            "historical_actual_analysis": historical["provenance"]["executable_memory_analysis"],
            "current_memory_records": {"512": {"temp_size_in_bytes": 200}}},
        "host_memory": {"MemAvailable_bytes": 1000, "conservative_required_bytes": 900}}
    report = {"fixed1024_B1024_decision": decision}
    collector.validate_stage_decision(report, 1024, historical, current)
    decision["requires_performance_improvement"] = True
    with pytest.raises(ValueError, match="time/memory-only"):
        collector.validate_stage_decision(report, 1024, historical, current)


def test_two_stage_comparison_requires_one_actual_process_and_environment():
    def stage(pid, hostname="test-node"):
        env = {"git_head": CODE_SHA, "hostname": hostname, "jax": "0.4.38", "jaxlib": "0.4.38",
            "numpy": "2.4.6", "x64_enabled": True, "device_details": [{"platform": "gpu"}],
            "physical_gpu": {"uuid": "GPU-synthetic", "driver_version": "525.85.12"}}
        return {"execution_origin": {"pid": pid, "environment": env},
            "allocation": {"Slurm": {"SLURM_JOB_ID": "999999", "SLURM_JOB_NODELIST": "test-node"}}}
    collector.validate_shared_origin([stage(10), stage(10)])
    accepted, withheld = collector.retain_shared_stages([stage(10), stage(11)], [])
    assert accepted == [stage(10)]
    assert withheld[0]["stage"] == collector.STAGE_NAMES[1]
    accepted, withheld = collector.retain_shared_stages([stage(10), stage(10, "other-node")], [])
    assert accepted == [stage(10)]
    assert "environment" in withheld[0]["reason"]


def test_timeout_after_strict_B512_seal_keeps_stage_and_counts_parent_cost_once():
    approval = make_approval()
    raw = "|".join((approval.job_id, approval.job_name, "gnall", "TIMEOUT", "5000", "4",
        "cpu=4,node=1,billing=4,gres/gpu=1", approval.node, "124:0"))
    accounting = collector.parse_raw9(raw, approval)
    sealed = {"stage": collector.STAGE_NAMES[0], "strict_numerical_passed": True,
        "actual_saved_output_reference_comparison_count": 1024, "lane_days": 14336}
    result = {"valid_exploratory_stages": [sealed], "withheld_stages": [],
        "incomplete_stages": [{"stage": collector.STAGE_NAMES[1], "status": "stopped"}],
        "capacity_curve_available": False,
        "capacity_curve_scope": "B1024 unmeasured or withheld; no B512-versus-B1024 curve"}
    collector.apply_parent_accounting(result, accounting)
    assert result["valid_exploratory_stages"] == [sealed]
    assert result["incomplete_stages"][0]["status"] == "stopped"
    assert result["capacity_curve_available"] is False
    assert result["accounting"]["state"] == "TIMEOUT"
    assert result["accounting"]["whole_job_cost"]["primary"]["GPU_card_hours"] == 5000 / 3600
    assert result["accounting"]["parent_counted_once"] is True


def test_denied_empty_manifest_summary_withholds_both_stages_and_makes_no_claims():
    args = SimpleNamespace(job_id="999999", expected_code_sha=CODE_SHA)
    result = collector.empty_result(args, "B512 admission denied")
    assert result["valid_exploratory_stages"] == []
    assert [item["stage"] for item in result["withheld_stages"]] == list(collector.STAGE_NAMES)
    assert result["scientific_acceptance"] is False
    assert result["scientific_claims"]["cpu_gpu_ratio"] is None
    assert result["K_amortization"].startswith("not reported")
