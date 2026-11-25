from __future__ import annotations

import copy
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.coupled import stomate_active_layer_ok_leak_inputs  # noqa: E402
from jax_orchidee.stomate.soilcarbon_kernels import (  # noqa: E402
    AltCalcDocResult,
    altcalc_doc,
)
from scripts.dev.audit_restart_state_lifecycle import (  # noqa: E402
    DEFAULT_CONTRACT,
    build_audit,
    build_production_state_audit,
    lifecycle_contract_fields,
    reader_contract_fields,
)


def _document() -> dict:
    return yaml.safe_load(DEFAULT_CONTRACT.read_text(encoding="utf-8"))


def _write_contract(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "restart_state_lifecycle.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_contract_declares_axes_for_every_current_restart_reader_field():
    document = _document()
    declared = [
        field for group in document["axis_groups"].values() for field in group["fields"]
    ]
    assert len(declared) == len(set(declared))
    assert set(declared) == set(lifecycle_contract_fields())


def test_state_transition_and_stage_two_restart_file_lifecycle_are_closed():
    audit = build_audit()
    assert audit["state_transition_closed"] is True
    assert audit["restart_lifecycle_closed"] is True
    assert audit["closed"] is True
    assert (
        audit["summary"]["contract_fields"] == len(lifecycle_contract_fields()) == 181
    )
    assert (
        audit["summary"]["restart_reader_fields"]
        == len(reader_contract_fields())
        == 169
    )
    assert audit["summary"]["packet_only_fields"] == 12
    assert audit["summary"]["closed_fields"] == 181
    assert audit["summary"]["structured_state_contracts"] == 1
    assert audit["summary"]["production_value_contracts"] == 2
    assert all(record["lifecycle"]["closed"] for record in audit["records"])


def test_restart_variable_requires_ast_overwrite_or_exact_guarded_carry():
    records = {record["field"]: record for record in build_audit()["records"]}
    biomass = records["biomass"]
    assert biomass["lifecycle"]["required_stages"]["update_or_guarded_carry"] is True
    assert (
        biomass["freshness_contract"]["resolved_update_mode"] == "production_overwrite"
    )
    assert biomass["freshness_contract"]["update_group"] == "daily_packet_literal"
    assert biomass["mask_contract"]["npts"]["mode"] == "compressed_land_domain"
    assert biomass["mask_contract"]["nvm"]["mode"] == "PFTpresent"

    carried = records["O2_soil"]
    assert carried["freshness_contract"]["resolved_update_mode"] == "guarded_carry"
    assert carried["freshness_contract"]["update_group"] == "fixed_path_carry_fields"
    assert carried["freshness_contract"]["guard"] == "OK_PC=false"


def test_non_restart_dispositions_have_their_own_strict_contracts():
    records = {record["field"]: record for record in build_audit()["records"]}
    assert all(records["soilc_total"]["lifecycle"]["required_stages"].values())
    assert records["soilc_total"]["lifecycle"]["jax_restart_write_exists"] is False
    assert all(records["snowfall_daily"]["lifecycle"]["required_stages"].values())
    assert (
        records["snowfall_daily"]["lifecycle"]["fortran_restart_write_exists"] is False
    )
    assert all(
        records["read_input_thawed_humidity"]["lifecycle"]["required_stages"].values()
    )
    assert (
        records["read_input_thawed_humidity"]["lifecycle"]["jax_restart_write_exists"]
        is False
    )


def test_packet_only_has_transition_owners_without_fabricated_restart_io():
    records = {record["field"]: record for record in build_audit()["records"]}
    for field in _document()["packet_only_fields"]:
        record = records[field]
        assert all(record["lifecycle"]["required_stages"].values())
        assert record["lifecycle"]["fortran_restart_read"] is False
        assert record["lifecycle"]["fortran_restart_write_exists"] is False
        assert record["lifecycle"]["jax_restart_write_exists"] is False


def test_slowproc_height_and_frac_age_have_field_specific_transition_contracts():
    records = {record["field"]: record for record in build_audit()["records"]}
    height = records["height"]
    frac_age = records["frac_age"]

    assert height["field_disposition"] == "packet_only"
    assert height["axes"] == ["npts", "nvm"]
    assert height["mask_contract"]["nvm"]["mode"] == "PFTpresent"
    assert height["owners"]["producer"].endswith("carbon_kernels.crown_step")
    assert frac_age["field_disposition"] == "packet_only"
    assert frac_age["axes"] == ["npts", "nvm", "nleafages"]
    assert frac_age["mask_contract"]["nvm"]["mode"] == "veget_mask_2d"
    assert "LAI_MAP=false" in frac_age["freshness_contract"]["cross_step"] or (
        "slowproc" in frac_age["freshness_contract"]["cross_step"]
    )


def test_every_land_or_pft_field_has_group_resolved_mask_and_freshness():
    for record in build_audit()["records"]:
        axes = set(record["axes"])
        if "npts" in axes:
            assert record["mask_contract"]["npts"]["mode"] == "compressed_land_domain"
            assert record["mask_contract"]["npts"]["inactive_write_behavior"]
            assert record["mask_contract"]["npts"]["fortran_provenance"]
        if "nvm" in axes:
            assert record["mask_contract"]["nvm"]["mode"] in {
                "all_slots_state",
                "veget_mask_2d",
                "PFTpresent",
                "is_peat_or_permafrost",
            }
            assert record["mask_contract"]["nvm"]["inactive_write_behavior"]
            assert record["mask_contract"]["nvm"]["owner"]
        assert {"step_behavior", "cross_step", "cross_year"} <= set(
            record["freshness_contract"]
        )


def test_resp_hetero_targets_daily_save_state_not_instantaneous_output():
    record = {record["field"]: record for record in build_audit()["records"]}[
        "resp_hetero"
    ]
    assert record["restart_field_contract"]["fortran_state_symbol"] == "resp_hetero_d"
    assert record["restart_field_contract"]["restart_variable_name"] == "resp_hetero"
    assert record["freshness_contract"]["update_group"] == "active_restart_producers"
    assert record["lifecycle"]["closed"] is True


def test_lm_thisyearmax_is_ast_proven_season_annual_overwrite():
    record = {record["field"]: record for record in build_audit()["records"]}[
        "lm_thisyearmax"
    ]
    assert (
        record["freshness_contract"]["resolved_update_mode"] == "production_overwrite"
    )
    assert record["freshness_contract"]["update_group"] == "season_outputs"
    assert record["owners"]["update_owner"].endswith(
        "_paper_day_season_fields_from_bundle_source"
    )
    assert record["lifecycle"]["required_stages"]["update_or_guarded_carry"] is True


def test_paper_static_inactive_updates_have_exact_executable_guards():
    records = {record["field"]: record for record in build_audit()["records"]}
    assert (
        records["veget_lastlight"]["freshness_contract"]["guard"]
        == "STOMATE_OK_DGVM=false"
    )
    assert records["need_adjacent"]["freshness_contract"]["guard"] == (
        "STOMATE_OK_DGVM=false and LPJ_GAP_CONST_MORT=true"
    )
    assert records["prod10"]["freshness_contract"]["guard"] == (
        "LAND_COVER_CHANGE=false makes do_now_stomate_lcchange false"
    )
    assert records["deepC_a"]["freshness_contract"]["guard"] == (
        "OK_PC=false and LAND_COVER_CHANGE=false"
    )
    assert all(
        records[field]["owners"]["update_owner"].endswith(
            "stomate_fixed_path_carry_state"
        )
        for field in ("veget_lastlight", "need_adjacent", "prod10", "deepC_a")
    )


def test_c98_active_layer_is_a_six_leaf_private_state_with_one_public_projection():
    audit = build_production_state_audit()
    assert "ok_leak_active_layer" not in lifecycle_contract_fields()

    contract = audit["structured_state_contracts"]["ok_leak_active_layer"]
    assert contract["state_type"].endswith("AltCalcDocResult")
    assert set(contract["leaves"]) == {
        "alt",
        "alt_ind",
        "altmax",
        "altmax_ind",
        "altmax_lastyear",
        "altmax_ind_lastyear",
    }
    assert all(item["axes"] == ["npts", "nvm"] for item in contract["leaves"].values())
    assert contract["leaves"]["altmax"]["dtype_family"] == "floating"
    assert contract["leaves"]["altmax_ind"]["dtype_family"] == "integer"
    assert contract["public_restart_field"] == "altmax"
    assert set(contract["private_leaves"]) == set(contract["leaves"]) - {"altmax"}
    assert contract["serialization"] == "private_not_fortran_restart"
    assert contract["owners"]["cold_and_file_warm_owner"].endswith(
        "stomate_active_layer_ok_leak_inputs"
    )
    assert contract["owners"]["compiled_fold_owner"].endswith(
        "_paper_compiled_ok_leak_fold"
    )
    assert contract["owners"]["compiled_projection_owner"].endswith(
        "_compiled_ok_leak_updates"
    )
    assert contract["closed"] is True


def test_c98_active_layer_production_shapes_cover_cold_warm_and_narrow_jit():
    npts, ndeep, nvm = 2, 3, 4
    tprof = jnp.full((npts, ndeep, nvm), 274.0)
    zi_soil = jnp.asarray([0.1, 0.5, 1.0])
    restart_altmax = jnp.asarray([[0.2, 0.4, 0.6, 0.8], [0.3, 0.5, 0.7, 0.9]])
    fixed_depth = jnp.zeros((npts, nvm))

    cold = stomate_active_layer_ok_leak_inputs(
        tprof=tprof,
        zi_soil=zi_soil,
        restart_altmax=restart_altmax,
        restart_fixed_cryoturbation_depth=fixed_depth,
        dayno=1,
        firstcall_soilcarbon=True,
    )
    cold_state = AltCalcDocResult(**cold.state_updates)
    carried = stomate_active_layer_ok_leak_inputs(
        tprof=tprof,
        zi_soil=zi_soil,
        restart_altmax=restart_altmax,
        restart_fixed_cryoturbation_depth=fixed_depth,
        dayno=2,
        firstcall_soilcarbon=False,
        previous_state=cold_state,
    )
    carried_state = AltCalcDocResult(**carried.state_updates)
    file_warm = stomate_active_layer_ok_leak_inputs(
        tprof=jnp.full_like(tprof, 272.0),
        zi_soil=zi_soil,
        restart_altmax=restart_altmax,
        restart_fixed_cryoturbation_depth=fixed_depth,
        dayno=3,
        firstcall_soilcarbon=True,
    )
    file_warm_state = AltCalcDocResult(**file_warm.state_updates)
    compiled = jax.jit(
        altcalc_doc,
        static_argnames=("firstcall", "newaltcalc", "dayno", "soilc_isspinup"),
    )(
        tprof,
        zi_soil,
        cold_state.altmax,
        cold_state.altmax_ind,
        cold_state.altmax_lastyear,
        cold_state.altmax_ind_lastyear,
        jnp.ones((npts, nvm), dtype=bool),
        firstcall=False,
        dayno=2,
    )

    for state in (cold_state, carried_state, file_warm_state, compiled):
        assert isinstance(state, AltCalcDocResult)
        assert all(jnp.shape(value) == (npts, nvm) for value in state)
        assert all(
            jnp.issubdtype(jnp.asarray(getattr(state, name)).dtype, jnp.floating)
            for name in ("alt", "altmax", "altmax_lastyear")
        )
        assert all(
            jnp.issubdtype(jnp.asarray(getattr(state, name)).dtype, jnp.integer)
            for name in ("alt_ind", "altmax_ind", "altmax_ind_lastyear")
        )
    assert jnp.array_equal(cold.output_inputs["altmax"], cold_state.altmax)
    assert jnp.array_equal(carried.output_inputs["altmax"], carried_state.altmax)
    assert jnp.array_equal(file_warm.output_inputs["altmax"], restart_altmax)
    assert jnp.array_equal(file_warm_state.altmax_lastyear, restart_altmax)
    assert jnp.array_equal(
        file_warm_state.altmax_ind, jnp.asarray([[1, 1, 2, 2], [1, 2, 2, 2]])
    )
    assert jnp.array_equal(file_warm_state.altmax_ind_lastyear, file_warm_state.altmax_ind)


def test_c103_and_wetness_accumulators_require_real_production_paths():
    contracts = build_production_state_audit()["production_value_contracts"]
    c103 = contracts["c103_day_end_state"]
    assert c103["fields"] == ["co2_fire", "litter_not_avail", "ni_acc"]
    assert c103["producer_keys"] == ["co2_fire", "ni_acc"]
    assert c103["packet_keys"] == ["litter_not_avail"]
    assert c103["writer_serialized"] is True
    assert c103["owners"]["packet_owner"].endswith(
        "_paper_day_first_stomate_state_packet"
    )

    wetness = contracts["peat_daily_accumulators"]
    assert wetness["fields"] == ["fwet_daily", "liqwt_daily"]
    assert wetness["producer_keys"] == ["fwet_daily", "liqwt_daily"]
    assert wetness["compiled_keys"] == ["fwet_daily", "liqwt_daily"]
    assert wetness["reset_constant"] == "PAPER_DAY_ZERO_DAILY_RESET_FIELDS"
    assert wetness["writer_serialized"] is True
    assert wetness["owners"]["compiled_owner"].endswith(
        "_stomate_daily_accumulation_fold_runtime_from_entries"
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda d: d["mask_contract_groups"].pop("nvm"),
            "mask contract coverage mismatch for nvm",
        ),
        (
            lambda d: d["derived_fields"]["soilc_total"].pop("jax_owner"),
            "missing YAML jax_owner",
        ),
        (
            lambda d: d["runtime_initialized_fields"]["snowfall_daily"].pop(
                "update_owner"
            ),
            "missing YAML update_owner",
        ),
        (
            lambda d: d["packet_only_fields"]["veget"].pop("cross_year"),
            "missing non-empty YAML contract key cross_year",
        ),
        (
            lambda d: d["disposition_contracts"]["restart_variable"][
                "freshness_contract"
            ].pop("cross_step"),
            "missing non-empty YAML contract key cross_step",
        ),
        (
            lambda d: d["fixed_path_carry_fields"]["carbon"].pop("guard"),
            "missing non-empty YAML contract key guard",
        ),
    ],
)
def test_missing_contracts_fail_audit(tmp_path: Path, mutation, message: str):
    document = copy.deepcopy(_document())
    mutation(document)
    with pytest.raises(ValueError, match=message):
        build_audit(contract_path=_write_contract(tmp_path, document))


def test_unknown_mask_field_cannot_inherit_a_default(tmp_path: Path):
    document = copy.deepcopy(_document())
    fields = document["mask_contract_groups"]["nvm"]["pftpresent_state"]["fields"]
    fields[fields.index("biomass")] = "unknown_state"
    with pytest.raises(ValueError, match="mask contract coverage mismatch for nvm"):
        build_audit(contract_path=_write_contract(tmp_path, document))


def test_production_owner_must_ast_write_the_claimed_field(tmp_path: Path):
    document = copy.deepcopy(_document())
    document["restart_update_groups"]["daily_packet_literal"]["fields"].append("carbon")
    with pytest.raises(ValueError, match="owner does not write fields: .*carbon"):
        build_audit(contract_path=_write_contract(tmp_path, document))


def test_restart_field_cannot_inherit_a_default_update_owner(tmp_path: Path):
    document = copy.deepcopy(_document())
    document["restart_update_groups"]["daily_packet_literal"]["fields"].remove(
        "biomass"
    )
    with pytest.raises(
        ValueError, match="restart update contract coverage mismatch: .*biomass"
    ):
        build_audit(contract_path=_write_contract(tmp_path, document))


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda d: d["structured_state_contracts"]["ok_leak_active_layer"]["leaves"].pop(
                "altmax_ind"
            ),
            "structural leaves must exactly match AltCalcDocResult",
        ),
        (
            lambda d: d["structured_state_contracts"]["ok_leak_active_layer"]["leaves"][
                "alt"
            ].update(axes=["npts"]),
            r"structural leaf axes must be \[npts, nvm\]",
        ),
        (
            lambda d: d["structured_state_contracts"]["ok_leak_active_layer"].update(
                public_restart_field="alt"
            ),
            "public restart projection must be altmax",
        ),
        (
            lambda d: d["structured_state_contracts"]["ok_leak_active_layer"]["leaves"][
                "altmax_ind"
            ].update(dtype_family="floating"),
            "structural leaf dtype_family must be integer",
        ),
        (
            lambda d: d["structured_state_contracts"]["ok_leak_active_layer"].update(
                serialization="fortran_restart"
            ),
            "structural state must not claim a Fortran restart variable",
        ),
        (
            lambda d: d["production_value_contracts"]["c103_day_end_state"].update(
                fields=["co2_fire", "ni_acc"]
            ),
            "c103_day_end_state: fields must be co2_fire, ni_acc, litter_not_avail",
        ),
        (
            lambda d: d["production_value_contracts"]["peat_daily_accumulators"].update(
                compiled_owner="jax_orchidee.stomate.daily.stomate_daily_accumulation_fold_from_entries"
            ),
            "compiled owner lacks wetness fields",
        ),
        (
            lambda d: d["production_value_contracts"]["peat_daily_accumulators"].update(
                writer_owner="jax_orchidee.stomate.restart_io._write_remainder_dataset"
            ),
            "writer_owner must be _write_daily_dataset",
        ),
        (
            lambda d: d["production_value_contracts"]["c103_day_end_state"].update(
                next_reader_owner="jax_orchidee.stomate.restart_io._write_remainder_dataset"
            ),
            "next_reader_owner must be read_stomate_readstart_states_from_template",
        ),
    ],
)
def test_structured_and_production_contract_mutations_fail(
    tmp_path: Path, mutation, message: str
):
    document = copy.deepcopy(_document())
    mutation(document)
    with pytest.raises(ValueError, match=message):
        build_production_state_audit(contract_path=_write_contract(tmp_path, document))
