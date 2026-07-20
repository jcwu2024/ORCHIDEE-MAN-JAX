from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from outputs.server_trace_patch import (
    apply_bridge_trace_patch,
    apply_diffuco_active_after_main_patch,
    apply_enerbil_active_trace_patch,
    apply_enerbil_pottemp_active_trace_patch,
)
from outputs.server_trace_patch.bridge_trace_patch_plan import tag_names


def test_bridge_patch_script_targets_private_server_copy_only():
    assert apply_bridge_trace_patch.SRC.parts[-3:] == (
        "modipsl_250919_dev",
        "modeles",
        "ORCHIDEE",
    )
    assert apply_diffuco_active_after_main_patch.SRC.parts[-3:] == (
        "modipsl_250919_dev",
        "modeles",
        "ORCHIDEE",
    )
    assert apply_enerbil_active_trace_patch.SRC.parts[-3:] == (
        "modipsl_250919_dev",
        "modeles",
        "ORCHIDEE",
    )
    assert apply_enerbil_pottemp_active_trace_patch.SRC.parts[-3:] == (
        "modipsl_250919_dev",
        "modeles",
        "ORCHIDEE",
    )
    assert "fortran_source" not in apply_bridge_trace_patch.SRC.parts
    assert "fortran_source" not in apply_diffuco_active_after_main_patch.SRC.parts
    assert "fortran_source" not in apply_enerbil_active_trace_patch.SRC.parts
    assert "fortran_source" not in apply_enerbil_pottemp_active_trace_patch.SRC.parts


def test_bridge_patch_script_anchors_match_local_fortran_truth_without_editing():
    sechiba = ROOT / "fortran_source" / "ORCHIDEE" / "src_sechiba" / "sechiba.f90"
    slowproc = ROOT / "fortran_source" / "ORCHIDEE" / "src_sechiba" / "slowproc.f90"
    sechiba_text = sechiba.read_text(encoding="latin-1")
    slowproc_text = slowproc.read_text(encoding="latin-1")

    assert "CALL diffuco_main" in sechiba_text
    assert "CALL enerbil_main" in sechiba_text
    assert "CALL hydrol_main" in sechiba_text
    assert "CALL thermosoil_main" in sechiba_text
    assert "CALL slowproc_main" in sechiba_text
    assert "CALL stomate_main" in slowproc_text

    # Exercise the exact replacement helper against representative anchors in
    # memory only. The local Fortran source truth must not be patched.
    patched = apply_bridge_trace_patch.replace_once(
        sechiba_text,
        "    CALL slowproc_main (kjit, kjpij, kjpindex, date0, &\n",
        "    ! ORCHJAX_TEST_SENTINEL\n    CALL slowproc_main (kjit, kjpij, kjpindex, date0, &\n",
        "test sentinel",
    )
    assert "ORCHJAX_TEST_SENTINEL" in patched
    assert "ORCHJAX_TEST_SENTINEL" not in sechiba.read_text(encoding="latin-1")

    active_patched = apply_diffuco_active_after_main_patch.replace_once(
        sechiba_text,
        "         & hist_id, hist2_id)\n\n!    WRITE(numout,*) 'xuhui: before enerbil_main:'\n",
        "         & hist_id, hist2_id)\n    ! ORCHJAX_ACTIVE_AFTER_MAIN_SENTINEL\n\n!    WRITE(numout,*) 'xuhui: before enerbil_main:'\n",
        "active after-main sentinel",
    )
    assert "ORCHJAX_ACTIVE_AFTER_MAIN_SENTINEL" in active_patched
    assert "ORCHJAX_ACTIVE_AFTER_MAIN_SENTINEL" not in sechiba.read_text(encoding="latin-1")

    bridge_patched_text = (
        "       CLOSE(961)\n"
        "    ENDIF\n\n"
        "!    WRITE(numout,*) 'xuhui: before enerbil_main:'\n"
    )
    active_bridge_patched = apply_diffuco_active_after_main_patch.replace_once(
        bridge_patched_text,
        "       CLOSE(961)\n    ENDIF\n\n!    WRITE(numout,*) 'xuhui: before enerbil_main:'\n",
        "       CLOSE(961)\n    ENDIF\n\n    ! ORCHJAX_ACTIVE_BRIDGE_SENTINEL\n\n!    WRITE(numout,*) 'xuhui: before enerbil_main:'\n",
        "active bridge sentinel",
    )
    assert "ORCHJAX_ACTIVE_BRIDGE_SENTINEL" in active_bridge_patched


def test_bridge_patch_script_emits_all_required_tag_names():
    script = (ROOT / "outputs" / "server_trace_patch" / "apply_bridge_trace_patch.py").read_text()

    required = set(tag_names())
    # The optional modelout tag is not part of the first bridge patch script.
    for tag in required:
        assert tag in script

    assert "orchjax_sechiba_bridge_diffuco_trace.txt" in script
    assert "orchjax_diffuco_trans_co2_trace.txt" in script
    assert "after_diffuco_trans_co2_pft14" in script
    assert "orchjax_sechiba_bridge_enerbil_trace.txt" in script
    assert "orchjax_sechiba_bridge_hydrol_trace.txt" in script
    assert "orchjax_sechiba_bridge_thermosoil_trace.txt" in script
    assert "orchjax_sechiba_bridge_slowproc_trace.txt" in script


def test_active_after_main_patch_script_is_incremental_and_uses_new_trace_file():
    script = (ROOT / "outputs" / "server_trace_patch" / "apply_diffuco_active_after_main_patch.py").read_text()

    assert "after_diffuco_main_active_pft14" in script
    assert "orchjax_sechiba_bridge_diffuco_active_trace.txt" in script
    assert "CALL diffuco_main" not in script


def test_enerbil_active_patch_script_is_incremental_and_uses_new_trace_file():
    script_path = ROOT / "outputs" / "server_trace_patch" / "apply_enerbil_active_trace_patch.py"
    script = script_path.read_text()

    assert "before_enerbil_main_active_pft14" in script
    assert "after_enerbil_main_active_pft14" in script
    assert "orchjax_sechiba_bridge_enerbil_active_trace.txt" in script
    assert "swnet(ji)" in script
    assert "soilflx(ji)" in script
    assert "soilflx_pft(ji,jv)" in script
    assert "pgflux(ji)" in script
    assert "temp_sol_add(ji)" in script
    assert "lai(ji,jv), gpp(ji,jv), veget_max(ji,jv)" in script
    assert "CALL enerbil_main" not in script
    assert "active before_enerbil_main trace already present" in script
    assert "active after_enerbil_main trace already present" in script
    assert "active after_enerbil_main trace after bridge block" in script


def test_enerbil_pottemp_active_patch_script_is_incremental_and_uses_new_trace_file():
    script_path = ROOT / "outputs" / "server_trace_patch" / "apply_enerbil_pottemp_active_trace_patch.py"
    script = script_path.read_text()

    assert "before_enerbil_pottemp_active" in script
    assert "after_enerbil_pottemp_active" in script
    assert "orchjax_enerbil_pottemp_active_trace.txt" in script
    assert "q_sol_pot(ji)" in script
    assert "temp_sol_pot(ji)" in script
    assert "CALL enerbil_pottemp" in script
    assert "active before_enerbil_pottemp trace already present" in script
    assert "active after_enerbil_pottemp trace already present" in script


def test_enerbil_active_patch_anchors_match_local_fortran_truth_without_editing():
    sechiba = ROOT / "fortran_source" / "ORCHIDEE" / "src_sechiba" / "sechiba.f90"
    text = sechiba.read_text(encoding="latin-1")

    before_patched = apply_enerbil_active_trace_patch.replace_once(
        text,
        "!    WRITE(numout,*) 'xuhui: before enerbil_main:'\n",
        "    ! ORCHJAX_BEFORE_ENERBIL_SENTINEL\n!    WRITE(numout,*) 'xuhui: before enerbil_main:'\n",
        "active before enerbil sentinel",
    )
    assert "ORCHJAX_BEFORE_ENERBIL_SENTINEL" in before_patched

    after_patched = apply_enerbil_active_trace_patch.replace_once(
        text,
        "         & precip_rain,snowdz,pgflux,temp_sol_add)\n!    WRITE(numout,*) 'xuhui: after enerbil_main:'\n",
        "         & precip_rain,snowdz,pgflux,temp_sol_add)\n    ! ORCHJAX_AFTER_ENERBIL_SENTINEL\n!    WRITE(numout,*) 'xuhui: after enerbil_main:'\n",
        "active after enerbil sentinel",
    )
    assert "ORCHJAX_AFTER_ENERBIL_SENTINEL" in after_patched
    bridge_after_patched = apply_enerbil_active_trace_patch.replace_once(
        "       CLOSE(962)\n    ENDIF\n!    WRITE(numout,*) 'xuhui: after enerbil_main:'\n",
        "       CLOSE(962)\n    ENDIF\n!    WRITE(numout,*) 'xuhui: after enerbil_main:'\n",
        "       CLOSE(962)\n    ENDIF\n    ! ORCHJAX_AFTER_BRIDGE_ENERBIL_SENTINEL\n!    WRITE(numout,*) 'xuhui: after enerbil_main:'\n",
        "active after bridge sentinel",
    )
    assert "ORCHJAX_AFTER_BRIDGE_ENERBIL_SENTINEL" in bridge_after_patched
    assert "ORCHJAX_BEFORE_ENERBIL_SENTINEL" not in sechiba.read_text(encoding="latin-1")
    assert "ORCHJAX_AFTER_ENERBIL_SENTINEL" not in sechiba.read_text(encoding="latin-1")


def test_enerbil_pottemp_active_patch_anchors_match_local_fortran_truth_without_editing():
    enerbil = ROOT / "fortran_source" / "ORCHIDEE" / "src_sechiba" / "enerbil.f90"
    text = enerbil.read_text(encoding="latin-1")
    call_block = (
        "    CALL enerbil_pottemp (kjpindex, zlev, emis, &\n"
        "       & epot_air, petAcoef, petBcoef, qair, peqAcoef, peqBcoef, soilflx, rau, u, v, q_cdrag, vbeta,&\n"
        "       & valpha, vbeta1, vbeta5, soilcap, lwdown, swnet, q_sol_pot, temp_sol_pot)\n"
    )

    before_patched = apply_enerbil_pottemp_active_trace_patch.replace_once(
        text,
        call_block,
        "    ! ORCHJAX_BEFORE_POTTEMP_SENTINEL\n" + call_block,
        "active before pottemp sentinel",
    )
    assert "ORCHJAX_BEFORE_POTTEMP_SENTINEL" in before_patched

    after_patched = apply_enerbil_pottemp_active_trace_patch.replace_once(
        text,
        call_block,
        call_block + "\n    ! ORCHJAX_AFTER_POTTEMP_SENTINEL\n",
        "active after pottemp sentinel",
    )
    assert "ORCHJAX_AFTER_POTTEMP_SENTINEL" in after_patched
    assert "ORCHJAX_BEFORE_POTTEMP_SENTINEL" not in enerbil.read_text(encoding="latin-1")
    assert "ORCHJAX_AFTER_POTTEMP_SENTINEL" not in enerbil.read_text(encoding="latin-1")
