from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from outputs.server_trace_patch.bridge_trace_patch_plan import (
    BRIDGE_TRACE_TAGS,
    fixed_format_registry_draft,
    schema_by_tag,
    tag_names,
)


def test_bridge_trace_patch_plan_exposes_minimum_record_groups():
    names = set(tag_names())

    assert {
        "after_diffuco_main",
        "after_diffuco_main_active_pft14",
        "after_enerbil_main",
        "after_hydrol_main_pft",
        "after_hydrol_main_layer",
        "after_hydrol_main_tile_layer",
        "after_hydrol_main_tile",
        "after_thermosoil_before_slowproc_pft",
        "after_thermosoil_before_slowproc_layer",
        "before_slowproc_main",
        "before_stomate_main",
    } <= names


def test_bridge_trace_patch_plan_covers_required_bridge_fields():
    schemas = schema_by_tag()

    assert {"gpp", "gsmean", "rveget", "rstruct", "cimean"} <= set(schemas["after_diffuco_main"])
    assert {
        "valpha",
        "vbeta",
        "vbeta_pft",
        "vbeta1",
        "vbeta2",
        "vbeta3",
        "vbeta3pot",
        "vbeta4",
        "vbeta4_pft",
        "evap_bare_lim",
        "tot_bare_soil",
        "qair",
        "temp_air",
        "pb",
    } <= set(schemas["after_diffuco_main_active_pft14"])
    assert {
        "transpir",
        "transpot",
        "vevapwet",
        "vevapnu",
        "vevapnu_pft",
        "vevapsno",
        "vevapflo",
        "evapot_corr",
        "temp_sol",
        "temp_sol_new",
        "qsurf",
        "t2mdiag",
    } <= set(schemas["after_enerbil_main"])
    assert {"humrel", "vegstress", "litterhumdiag"} <= set(schemas["after_hydrol_main_pft"])
    assert {"shumdiag", "shumdiag_perma", "soilmoist"} <= set(schemas["after_hydrol_main_layer"])
    assert {"soil_mc", "wat_flux"} <= set(schemas["after_hydrol_main_tile_layer"])
    assert {"runoff_per_soil", "runoff2peat", "drainage_per_soil"} <= set(schemas["after_hydrol_main_tile"])
    assert {"stempdiag", "mc_layh", "mcl_layh"} <= set(schemas["after_thermosoil_before_slowproc_layer"])
    assert {"do_slow", "dt_sechiba", "dt_stomate", "dt_days"} <= set(schemas["before_stomate_main"])


def test_bridge_trace_patch_registry_draft_has_stable_unique_fields():
    registry = fixed_format_registry_draft()

    assert set(registry) == set(tag_names())
    for tag in BRIDGE_TRACE_TAGS:
        fields = registry[tag.name]["fields"]
        assert isinstance(registry[tag.name]["trace_file"], str)
        assert tuple(fields) == tag.fields
        assert len(fields) == len(set(fields)), tag.name
        assert tag.fields[0] == "kjit"
