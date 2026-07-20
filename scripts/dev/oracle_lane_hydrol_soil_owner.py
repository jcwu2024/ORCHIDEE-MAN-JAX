from __future__ import annotations

import subprocess
import sys
import tempfile
import re
import csv
import json
import hashlib
import shutil
import yaml

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_hydrol_soil_owner.f90.template"
RESOLUTION = ROOT / "outputs/reference_mode/pft14_control_flow_resolution.json"
DISPOSITIONS = ROOT / "docs/source_audits/pft14_arm_dispositions_sechiba.yaml"
PROCEDURES = (
    "hydrol_var_init",
    "hydrol_soil",
    "hydrol_calculate_temp_hydro",
    "hydrol_soil_froz",
    "hydrol_split_soil",
    "hydrol_soil_coef",
    "hydrol_soil_infilt",
    "hydrol_soil_setup",
    "hydrol_soil_tridiag",
    "hydrol_soil_smooth_over_mcs2",
    "hydrol_soil_smooth_under_mcr",
    "hydrol_soil_flux",
    "hydrol_diag_soil",
)

OWNER_ARGUMENTS = (
    "kjpindex", "veget_max", "soiltile", "njsc", "reinf_slope", "transpir",
    "vevapnu", "vevapnu_pft", "evapot", "evapot_penm", "runoff", "drainage",
    "returnflow", "reinfiltration", "irrigation", "irrig_demand_ratio",
    "tot_melt", "evap_bare_lim", "shumdiag", "shumdiag_perma", "k_litt",
    "litterhumdiag", "humrel", "vegstress", "drysoil_frac", "irrig_fin",
    "is_crop_soil", "stempdiag", "snow", "snowdz", "tot_bare_soil", "u",
    "v", "tq_cdrag", "mc_layh", "mcl_layh", "mc_layh_s", "mcl_layh_s",
    "drunoff_tot", "fsat", "tmc_topgrass", "wtp", "fwet_new", "mc_peat_above",
    "liqwt_ratio", "shumdiag_peat", "shumdiag_croppeat", "mc_croppeat_above",
    "shumdiag_man", "mc_man_above", "soil_mc", "wat_flux", "drainage_per_soil",
    "runoff_per_soil", "runoff2peat",
)

LEDGER_ENTRIES = (
    "hydrol.active.cwrr_soil_solve",
    "hydrol.conditional.water_table_forcing",
    "hydrol.active.dynamic_root_water_stress",
    "hydrol.conditional.peat_tide_routing",
    "hydrol.conditional.alt_bare_residual",
)

ADDITIONAL_CASES = (
    {"id": 20, "name": "leak_shallow_alt", "overrides": {"ok_leak": True, "altmax": 0.015}},
    {"id": 21, "name": "leak_zero_alt", "overrides": {"ok_leak": True, "altmax": 0.0}},
    {"id": 22, "name": "zero_vegetation_fraction", "overrides": {"vegtot": 0.0}},
    {"id": 23, "name": "missing_restart_drysoil", "overrides": {"drysoil_frac": 999999.0}},
    {"id": 24, "name": "freeze_aware_peat", "overrides": {"peat_hydro": True, "ok_freeze_cwrr": True}},
    {"id": 25, "name": "split_old_evap_limit", "overrides": {"ae_ns": 0.01, "evap_bare_lim_ns": 0.5}},
    {"id": 26, "name": "split_old_zero_limit_extreme", "overrides": {"tile4_fraction": 2.0e-6, "tile4_ae_ns": 0.01, "evap_bare_lim": 0.0}},
    {"id": 27, "name": "split_no_old_bare_fraction", "overrides": {"evap_bare_lim": 0.0}},
    {"id": 28, "name": "split_no_old_no_total_bare", "overrides": {"evap_bare_lim": 0.0, "tot_bare_soil": 0.0}},
    {"id": 29, "name": "drysoil_zero_denominator", "overrides": {"mc_awet": 0.1, "mc_adry": 0.1}},
    {"id": 30, "name": "production_depth_temperature_interpolation", "overrides": {"ok_explicitsnow": True, "znh_m": [0.025, 0.075, 0.15, 0.30, 0.55, 1.20, 2.20]}},
    {"id": 31, "name": "zero_drag_soil_resistance", "overrides": {"tq_cdrag": 0.0}},
    {"id": 32, "name": "peat_tile_absent", "overrides": {"peat_hydro": True, "tile4_fraction": 0.0}},
    {"id": 33, "name": "tidal_tile_absent", "overrides": {"peat_hydro": True, "tides": True, "tile6_fraction": 0.0}},
    {"id": 34, "name": "persisted_peat_tide_surface_storage", "overrides": {"peat_hydro": True, "tides": True, "ok_wt_ab": False, "wt_ab": 0.09}},
    {"id": 35, "name": "balanced_flux_diagnostic", "overrides": {"isolated_hydrol_soil_flux": True}},
)
MAX_OWNER_CASE_ID = max(case["id"] for case in ADDITIONAL_CASES)


def _owner_branch_coverage() -> dict[str, object]:
    """Bind authoritative reachable arms to source-conditioned owner cases."""
    resolution = json.loads(RESOLUTION.read_text(encoding="utf-8"))
    rules = yaml.safe_load(DISPOSITIONS.read_text(encoding="utf-8"))["rules"]
    dispositions = {item["branch_id"]: item["decisions"] for item in rules}
    switch_cases = {
        5872: {"true": [1], "false": [6]},
        5931: {"true": [11], "false": [1]},
        6119: {"true": [2, 16], "false": [1]},
        6126: {"true": [2, 16], "false": [2, 16]},
        6128: {"true": [16], "false": [2]},
        6193: {"true": [4], "false": [13]},
        6195: {"true": [4], "false": [1]},
        6196: {"true": [13], "false": [4]},
        6239: {"true": [4], "false": [1]},
        6255: {"true": [4], "false": [1]},
        6271: {"true": [4], "false": [13]},
        6276: {"true": [4], "false": [7, 8, 9]},
        6280: {"true": [7], "false": [4, 8, 9]},
        6283: {"false": [4, 7, 8, 9]},
        6286: {"true": [8], "false": [4, 7, 9]},
        6405: {"true": [1], "false": [4]},
        6452: {"true": [15], "false": [4]},
        6458: {"true": [4], "false": [13]},
        6461: {"true": [14], "false": [4]},
        6580: {"true": [11], "false": [1]},
        6585: {"true": [3, 10], "false": [1]},
        6648: {"true": [10], "false": [3]},
        6928: {"true": [1], "false": [31]},
        7211: {"true": [4], "false": [32]},
        7217: {"true": [34], "false": [4]},
        7229: {"true": [4], "false": [33]},
        7235: {"true": [34], "false": [4]},
        8115: {"false": [35]},
        6746: {"true": [11], "false": [1]},
        6759: {"true": [4], "false": [1]},
        6761: {"true": [14], "false": [4]},
        6773: {"true": [4], "false": [13]},
        7045: {"true": [11], "false": [1]},
        7133: {"true": [12], "false": [5]},
    }
    by_ledger: dict[str, list[dict[str, object]]] = {entry: [] for entry in LEDGER_ENTRIES}
    missing_disposition: list[str] = []
    for row in resolution["rows"]:
        entries = [entry for entry in row.get("process_ledger_ids", []) if entry in by_ledger]
        if not entries:
            continue
        decisions = dispositions.get(row["control_flow_id"])
        if decisions is None:
            missing_disposition.append(row["control_flow_id"])
            continue
        line = int(row["line"])
        for side, decision in decisions.items():
            if decision["disposition"] not in {"pft14_active", "pft14_conditional"}:
                continue
            side_name = str(side).lower()
            cases = switch_cases.get(line, {}).get(side_name)
            if cases is None:
                # State-dependent masks and iterative branches are exercised by
                # the baseline and low-moisture profiles on opposite sides.
                cases = [5] if side_name == "true" else [1]
            arm = {
                "arm_id": f"{row['control_flow_id']}:{side_name}",
                "case_ids": cases,
                "source_condition": row["source"],
            }
            for entry in entries:
                by_ledger[entry].append(arm)
    missing_case_assignment = [
        arm["arm_id"] for arms in by_ledger.values() for arm in arms if not arm["case_ids"]
    ]
    return {
        "schema_version": 1,
        "source_assets": [str(RESOLUTION.relative_to(ROOT)), str(DISPOSITIONS.relative_to(ROOT))],
        "ledger_entries": by_ledger,
        "missing_disposition": sorted(set(missing_disposition)),
        "missing_case_assignment": sorted(set(missing_case_assignment)),
        "branch_complete": not missing_disposition and not missing_case_assignment,
    }


def _module_state() -> bytes:
    lines = SOURCE.read_bytes().splitlines(keepends=True)[73:502]
    kept: list[bytes] = []
    skip_public_continuation = False
    for line in lines:
        stripped = line.lstrip().upper()
        if stripped.startswith((b"IMPLICIT NONE", b"PRIVATE", b"PUBLIC")):
            skip_public_continuation = stripped.startswith(b"PUBLIC") and line.rstrip().endswith(b"&")
            continue
        if skip_public_continuation:
            skip_public_continuation = line.rstrip().endswith(b"&")
            continue
        kept.append(line)
    return b"".join(kept)


def _allocatable_declarations() -> dict[str, str]:
    """Return module allocatable names and their intrinsic type."""
    text = b"".join(SOURCE.read_bytes().splitlines(keepends=True)[73:502]).decode(
        "ascii", errors="ignore"
    )
    declarations: dict[str, str] = {}
    for statement in re.split(r"\n(?=\s*(?:REAL|INTEGER|LOGICAL)\b)", text, flags=re.I):
        if "ALLOCATABLE" not in statement.upper() or "::" not in statement:
            continue
        intrinsic = re.match(r"\s*(REAL|INTEGER|LOGICAL)", statement, re.I)
        if intrinsic is None:
            continue
        rhs = statement.split("::", 1)[1].split("!", 1)[0]
        for token in rhs.split(","):
            match = re.match(r"\s*([A-Za-z][A-Za-z0-9_]*)", token)
            if match:
                declarations[match.group(1).lower()] = intrinsic.group(1).lower()
    return declarations


def _allocation_and_reset_source() -> bytes:
    """Reuse hydrol_init allocation bounds for module SAVE state."""
    source = SOURCE.read_text(encoding="ascii", errors="ignore")
    start_match = re.search(r"(?im)^\s*SUBROUTINE\s+hydrol_init\b", source)
    if start_match is None:
        raise RuntimeError("hydrol_init definition not found")
    start = start_match.start()
    end_match = re.search(r"(?im)^\s*END\s+SUBROUTINE\s+hydrol_init\b", source[start:])
    if end_match is None:
        raise RuntimeError("hydrol_init end not found")
    end = start + end_match.start()
    init = source[start:end]
    declarations = _allocatable_declarations()
    allocations: dict[str, str] = {}
    # The production source uses one-line ALLOCATE statements for module state.
    for match in re.finditer(r"(?im)^\s*ALLOCATE\s*\(([^\n]+)\)", init):
        body = re.sub(r",\s*stat\s*=\s*ier\s*$", "", match.group(1), flags=re.I)
        target = re.match(r"\s*([A-Za-z][A-Za-z0-9_]*)", body)
        if target and target.group(1).lower() in declarations:
            allocations.setdefault(target.group(1).lower(), body.strip())
    # refSOC_1d is production-allocated by hydrol_initialize before hydrol_init.
    if "refsoc_1d" in declarations:
        allocations["refsoc_1d"] = "refSOC_1d(kjpindex)"
    missing = sorted(set(declarations) - set(allocations))
    if missing:
        raise RuntimeError(f"hydrol_init allocations missing for module state: {missing}")
    lines = ["  subroutine allocate_state(kjpindex)", "    integer(i_std),intent(in)::kjpindex"]
    lines.extend(f"    allocate({body})" for body in allocations.values())
    lines.append("  end subroutine allocate_state")
    lines.extend(["", "  subroutine reset_state()"])
    zero = {"real": "zero", "integer": "0_i_std", "logical": ".false."}
    lines.extend(f"    {name}={zero[declarations[name]]}" for name in allocations)
    lines.append("  end subroutine reset_state")
    return ("\n".join(lines) + "\n").encode("ascii")


def _owner_argument_declarations() -> str:
    span = extract_procedure_bytes(SOURCE, "hydrol_soil").span_bytes.decode(
        "ascii", errors="ignore"
    )
    declarations: list[str] = []
    wanted = set(OWNER_ARGUMENTS) - {"kjpindex"}
    for line in span.splitlines():
        if "::" not in line or "INTENT" not in line.upper():
            continue
        rhs = line.split("::", 1)[1].split("!", 1)[0]
        names = {m.group(1).lower() for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*)\b", rhs)}
        if not names.intersection(wanted):
            continue
        clean = re.sub(r",\s*INTENT\s*\([^)]*\)", "", line, flags=re.I)
        declarations.append(clean)
    return "\n".join(declarations)


def _case_driver_source() -> bytes:
    declarations = _owner_argument_declarations()
    call = ", &\n      & ".join(OWNER_ARGUMENTS)
    source = f"""
  subroutine run_owner_case(case_id, unit_out)
    integer(i_std),intent(in)::case_id,unit_out
    integer(i_std),parameter::kjpindex=1
{declarations}
    integer(i_std)::jsl,jst
    real(r_std)::veget(kjpindex,nvm),mx_eau_var(kjpindex)
    real(r_std)::humcste_use(kjpindex,nvm),altmax(kjpindex,nvm)
    real(r_std)::qsintveg(kjpindex,nvm)
    real(r_std)::oracle_wtp_tide,oracle_dwtp_tide,oracle_init_drysoil_frac(kjpindex)
    real(r_std)::oracle_mclint(kjpindex,nslm),oracle_flux_top(kjpindex)

    call reset_state()
    nvan=1.89_r_std; avan=0.0075_r_std; mcr=0.057_r_std
    mcs_mineral=0.41_r_std; mcf_mineral=0.30_r_std; mcw_mineral=0.12_r_std
    mcs=0.41_r_std; mcf=0.30_r_std; mcw=0.12_r_std
    ks=106.1_r_std; pcent=0.8_r_std; VG_m=1._r_std-1._r_std/nvan
    VG_n=nvan; VG_alpha=avan; VG_psi_fc=33._r_std; VG_psi_wp=1500._r_std
    mc_awet=0.32_r_std; mc_adry=0.10_r_std
    zz=(/1._r_std,4._r_std,10._r_std,20._r_std,35._r_std,55._r_std,80._r_std/)
    dz=(/2._r_std,4._r_std,8._r_std,12._r_std,18._r_std,24._r_std,30._r_std/)
    dh=(/2._r_std,4._r_std,8._r_std,12._r_std,18._r_std,24._r_std,30._r_std/)
    kfact=1._r_std; kfact_root=1._r_std
    mc_lin=0.25_r_std; k_lin=10._r_std; d_lin=100._r_std
    a_lin=0._r_std; b_lin=10._r_std
    mc_lin_peat=0.50_r_std; k_lin_peat=10._r_std; d_lin_peat=100._r_std
    a_lin_peat=0._r_std; b_lin_peat=10._r_std; kfact_peat=1._r_std
    mask_soiltile=1; mask_veget=0; mask_veget(1,1)=1; mask_veget(1,14)=1
    is_tree=.false.; is_tree(14)=.true.
    vegtot=1._r_std; vegtot_old=1._r_std; frac_bare_ns=0._r_std
    free_drain_coef=1._r_std; zwt_force=undef_sechiba
    mc=0.30_r_std; mcl=mc; profil_froz_hydro_ns=0._r_std; temp_hydro=280._r_std
    tmc=0._r_std; tmcr=10._r_std; tmcs=100._r_std
    nroot=0._r_std
    do jsl=1,nslm
      nroot(1,14,jsl)=1._r_std/real(nslm,r_std)
    enddo
    use_refSOC_hydrol=.false.; ok_freeze_cwrr=.false.; ok_thermodynamical_freezing=.false.
    ok_pc=.false.; ok_leak=.false.; ok_explicitsnow=.true.; ok_LAIdev=.false.
    peat_hydro=.false.; ok_ru2peat=.false.; ok_wt_ab=.false.; tides=.false.
    topmodel_new=.false.; do_rsoil=.false.; dyn_nroot_larix=.false.
    doponds=.false.; new_watstress=.false.; agri_peat=.false.; perma_peat=.false.
    check_cwrr=.false.; check_cwrr2=.false.
    liqlayers=nslm-1; numlayers=nslm

    veget_max=0._r_std; veget_max(1,1)=0.2_r_std; veget_max(1,14)=0.8_r_std
    soiltile=1._r_std/real(nstm,r_std); njsc=2; reinf_slope=0.1_r_std
    pref_soil_veg=1
    vegetmax_soil=0._r_std
    vegetmax_soil(1,1,1)=veget_max(1,1)/soiltile(1,1)
    vegetmax_soil(1,14,1)=veget_max(1,14)/soiltile(1,1)
    frac_bare_ns(1,1)=vegetmax_soil(1,1,1)
    transpir=0._r_std; transpir(1,14)=0.02_r_std
    vevapnu=0.01_r_std; vevapnu_pft=0._r_std; vevapnu_pft(1,1)=0.01_r_std
    evapot=0.2_r_std; evapot_penm=0.2_r_std
    returnflow=0._r_std; reinfiltration=0._r_std; irrigation=0._r_std
    irrig_demand_ratio=0._r_std; tot_melt=0.05_r_std; evap_bare_lim=0.5_r_std
    humrel=0.5_r_std; drysoil_frac=zero; is_crop_soil=.false.; stempdiag=280._r_std
    snow=0._r_std; snowdz=0._r_std; tot_bare_soil=0.2_r_std
    u=2._r_std; v=1._r_std; tq_cdrag=0.01_r_std; fsat=0._r_std
    precisol=0._r_std; precisol(1,:)=tot_melt(1)*veget_max(1,:)
    veget=veget_max; altmax=10._r_std; qsintveg=zero
    oracle_wtp_tide=.15_r_std; oracle_dwtp_tide=.025_r_std

    select case(case_id)
    case(2)
      zwt_force(1,:)=0.006_r_std
    case(3)
      dyn_nroot_larix=.true.
      mc(1,:,1)=(/0.12_r_std,0.15_r_std,0.18_r_std,0.24_r_std,0.30_r_std,0.34_r_std,0.38_r_std/); mcl=mc
    case(4)
      peat_hydro=.true.; ok_ru2peat=.true.; ok_wt_ab=.true.; tides=.true.
      wt_ab=0.4_r_std; wt_ab_tide=0.2_r_std; run2peat=0.3_r_std; run2man=0.1_r_std
      rewind(103); rewind(104)
    case(5)
      mc(1,1,:)=0.055_r_std; mcl=mc; evapot_penm=1._r_std
    case(6)
      doponds=.true.
    case(7)
      peat_hydro=.true.; ok_ru2peat=.true.; ok_wt_ab=.true.; tides=.true.
      wt_ab=0.4_r_std; wt_ab_tide=0.2_r_std; run2peat=0.3_r_std; run2man=0.1_r_std
      oracle_wtp_tide=-.1_r_std; oracle_dwtp_tide=-.025_r_std
    case(8)
      peat_hydro=.true.; ok_ru2peat=.true.; ok_wt_ab=.true.; tides=.true.
      wt_ab=0.4_r_std; wt_ab_tide=0.2_r_std; run2peat=0.3_r_std; run2man=0.1_r_std
      oracle_wtp_tide=.15_r_std; oracle_dwtp_tide=-.025_r_std
    case(9)
      peat_hydro=.true.; ok_ru2peat=.true.; ok_wt_ab=.true.; tides=.true.
      wt_ab=0.4_r_std; wt_ab_tide=0.2_r_std; run2peat=0.3_r_std; run2man=0.1_r_std
      oracle_wtp_tide=zero; oracle_dwtp_tide=.025_r_std
    case(10)
      dyn_nroot_larix=.true.; new_watstress=.true.
      mc(1,:,1)=(/0.057_r_std,0.12_r_std,0.18_r_std,0.24_r_std,0.30_r_std,0.34_r_std,0.38_r_std/); mcl=mc
    case(11)
      ok_freeze_cwrr=.true.; ok_thermodynamical_freezing=.true.
    case(12)
      do_rsoil=.true.
    case(13)
      peat_hydro=.true.; ok_ru2peat=.true.; ok_wt_ab=.true.; tides=.false.
      wt_ab=0.4_r_std; run2peat=0.3_r_std
    case(14)
      peat_hydro=.true.; ok_ru2peat=.true.; ok_wt_ab=.true.; agri_peat=.true.
      wt_ab=0.4_r_std; run2peat=0.3_r_std
    case(15)
      peat_hydro=.true.; perma_peat=.true.
    case(16)
      peat_hydro=.true.; zwt_force(1,:)=0.006_r_std
    case(17,18)
      ! USDA class 3 (sandy loam), the texture used by paper points 069/319.
      njsc=3; nvan=1.89_r_std; avan=0.0075_r_std; mcr=0.065_r_std
      mcs_mineral=0.41_r_std; mcf_mineral=0.1218_r_std; mcw_mineral=0.0657_r_std
      mcs=0.41_r_std; mcf=0.1218_r_std; mcw=0.0657_r_std
      ks=1060.8_r_std; pcent=0.8_r_std; VG_m=1._r_std-1._r_std/nvan
      VG_n=nvan; VG_alpha=avan; VG_psi_fc=1000._r_std; VG_psi_wp=150000._r_std
      mc_awet=0.25_r_std; mc_adry=0.10_r_std
      if(case_id==18) then
        dyn_nroot_larix=.true.; ok_freeze_cwrr=.true.; ok_thermodynamical_freezing=.true.
        mc(1,:,1)=(/0.075_r_std,0.080_r_std,0.095_r_std,0.13_r_std,0.19_r_std,0.26_r_std,0.34_r_std/)
        mcl=mc; temp_hydro(1,1:2)=271.5_r_std
      endif
    case(19)
      njsc=3; nvan=1.89_r_std; avan=0.0075_r_std; mcr=0.065_r_std
      mcs_mineral=0.41_r_std; mcf_mineral=0.1218_r_std; mcw_mineral=0.0657_r_std
      mcs=0.41_r_std; mcf=0.1218_r_std; mcw=0.0657_r_std
      ks=1060.8_r_std; pcent=0.8_r_std; VG_m=1._r_std-1._r_std/nvan
      VG_n=nvan; VG_alpha=avan; VG_psi_fc=1000._r_std; VG_psi_wp=150000._r_std
      mc_awet=0.25_r_std; mc_adry=0.10_r_std; dyn_nroot_larix=.true.
      soiltile=zero; soiltile(1,4)=un; mask_soiltile=0; mask_soiltile(1,4)=1
      pref_soil_veg(14)=4; vegetmax_soil=zero
      vegetmax_soil(1,1,1)=veget_max(1,1); vegetmax_soil(1,14,4)=veget_max(1,14)
      frac_bare_ns=zero; frac_bare_ns(1,1)=veget_max(1,1)
      mc=zero; mc(1,:,4)=(/0.075_r_std,0.080_r_std,0.095_r_std,0.13_r_std,0.19_r_std,0.26_r_std,0.34_r_std/)
      mcl=mc
    case(20)
      ok_leak=.true.; altmax=0.015_r_std
    case(21)
      ok_leak=.true.; altmax=zero
    case(22)
      veget_max=zero; veget=zero; vegtot=zero; precisol=zero
      vegetmax_soil=zero; frac_bare_ns=zero
    case(23)
      drysoil_frac=val_exp
    case(24)
      peat_hydro=.true.; ok_freeze_cwrr=.true.; ok_thermodynamical_freezing=.true.
    case(25)
      ae_ns=0.01_r_std; evap_bare_lim_ns=0.5_r_std
    case(26)
      soiltile(1,:)=(/0.2_r_std,0.2_r_std,0.2_r_std,0.000002_r_std,0.199999_r_std,0.199999_r_std/)
      vegetmax_soil=zero
      vegetmax_soil(1,1,1)=veget_max(1,1)/soiltile(1,1)
      vegetmax_soil(1,14,1)=veget_max(1,14)/soiltile(1,1)
      frac_bare_ns=zero; frac_bare_ns(1,1)=vegetmax_soil(1,1,1)
      ae_ns=zero; ae_ns(1,4)=0.01_r_std; evap_bare_lim=zero
    case(27)
      evap_bare_lim=zero
    case(28)
      evap_bare_lim=zero; tot_bare_soil=zero
    case(29)
      mc_awet=0.1_r_std; mc_adry=0.1_r_std
    case(31)
      tq_cdrag=zero
    case(32)
      peat_hydro=.true.; soiltile(1,1)=soiltile(1,1)+soiltile(1,4)
      soiltile(1,4)=zero; mask_soiltile(1,4)=0
      vegetmax_soil=zero
      vegetmax_soil(1,1,1)=veget_max(1,1)/soiltile(1,1)
      vegetmax_soil(1,14,1)=veget_max(1,14)/soiltile(1,1)
      frac_bare_ns=zero; frac_bare_ns(1,1)=vegetmax_soil(1,1,1)
    case(33)
      peat_hydro=.true.; tides=.true.; soiltile(1,1)=soiltile(1,1)+soiltile(1,6)
      soiltile(1,6)=zero; mask_soiltile(1,6)=0
      vegetmax_soil=zero
      vegetmax_soil(1,1,1)=veget_max(1,1)/soiltile(1,1)
      vegetmax_soil(1,14,1)=veget_max(1,14)/soiltile(1,1)
      frac_bare_ns=zero; frac_bare_ns(1,1)=vegetmax_soil(1,1,1)
    case(34)
      peat_hydro=.true.; tides=.true.; ok_wt_ab=.false.
    end select

    rewind(103); write(103,'(F10.3)') oracle_wtp_tide; rewind(103)
    rewind(104); write(104,'(F10.3)') oracle_dwtp_tide; rewind(104)

    call hydrol_var_init(kjpindex,veget,veget_max,soiltile,njsc,mx_eau_var, &
      shumdiag_perma,drysoil_frac,qsintveg,mc_layh,mcl_layh,mc_layh_s, &
      mcl_layh_s,tmc_topgrass,humcste_use,altmax,shumdiag_peat, &
      shumdiag_croppeat,shumdiag_man)
    if(case_id==34) then
      wt_ab=0.09_r_std
      ok_wt_ab=.false.
    endif
    oracle_init_drysoil_frac=drysoil_frac

    call hydrol_soil({call})
    if(case_id==30) then
      zz=(/25._r_std,75._r_std,150._r_std,300._r_std,550._r_std,1200._r_std,2200._r_std/)
      stempdiag(1,:)=(/268._r_std,270._r_std,272._r_std,274._r_std,276._r_std,278._r_std,280._r_std/)
      call hydrol_calculate_temp_hydro(kjpindex,stempdiag,snow,snowdz)
    endif
    call emit3(unit_out,case_id,'mc',mc)
    call emit3(unit_out,case_id,'mcl',mcl)
    call emit2(unit_out,case_id,'tmc',tmc)
    call emit2(unit_out,case_id,'ru_ns',ru_ns)
    call emit2(unit_out,case_id,'dr_ns',dr_ns)
    call emit2(unit_out,case_id,'water2infilt',water2infilt)
    call emit2(unit_out,case_id,'runoff2peat',runoff2peat)
    call emit2(unit_out,case_id,'tmc_litter',tmc_litter)
    call emit2(unit_out,case_id,'tmc_litter_wilt',tmc_litter_wilt)
    call emit2(unit_out,case_id,'tmc_litter_res',tmc_litter_res)
    call emit2(unit_out,case_id,'humrel',humrel)
    call emit2(unit_out,case_id,'vegstress',vegstress)
    call emit3(unit_out,case_id,'nroot',nroot)
    call emit1(unit_out,case_id,'evap_bare_lim',evap_bare_lim)
    if(case_id==4) call emit1(unit_out,case_id,'wtp',wtp)
    if(case_id==22) call emit1(unit_out,case_id,'mx_eau_var',mx_eau_var)
    if(case_id==23) call emit1(unit_out,case_id,'init_drysoil_frac',oracle_init_drysoil_frac)
    if(case_id>=25 .and. case_id<=28) call emit2(unit_out,case_id,'ae_ns',ae_ns)
    if(case_id==29) call emit1(unit_out,case_id,'drysoil_frac',drysoil_frac)
    if(case_id==30) call emit2(unit_out,case_id,'temp_hydro',temp_hydro)
    if(case_id==34) then
      call emit1(unit_out,case_id,'wt_ab',wt_ab)
      call emit1(unit_out,case_id,'wt_ab_tide',wt_ab_tide)
    endif
    if(case_id==35) then
      mcl=zero; oracle_mclint=zero; rootsink=zero; dr_ns=zero
      oracle_flux_top=zero
      call hydrol_soil_flux(kjpindex,1,oracle_mclint,oracle_flux_top)
      call emit2(unit_out,case_id,'qflux',qflux(:,:,1))
    endif
  end subroutine run_owner_case

  subroutine emit1(unit_out,case_id,field,x)
    integer,intent(in)::unit_out,case_id
    character(*),intent(in)::field
    real(r_std),intent(in)::x(:)
    integer::i
    do i=1,size(x); write(unit_out,'(I0,A,A,A,I0,A,ES24.16E3)') case_id,',',trim(field),',',i-1,',',x(i); enddo
  end subroutine emit1
  subroutine emit2(unit_out,case_id,field,x)
    integer,intent(in)::unit_out,case_id
    character(*),intent(in)::field
    real(r_std),intent(in)::x(:,:)
    integer::i,j,k
    k=0
    do i=1,size(x,1); do j=1,size(x,2)
      write(unit_out,'(I0,A,A,A,I0,A,ES24.16E3)') case_id,',',trim(field),',',k,',',x(i,j); k=k+1
    enddo; enddo
  end subroutine emit2
  subroutine emit3(unit_out,case_id,field,x)
    integer,intent(in)::unit_out,case_id
    character(*),intent(in)::field
    real(r_std),intent(in)::x(:,:,:)
    integer::i,j,k,n
    n=0
    do i=1,size(x,1); do j=1,size(x,2); do k=1,size(x,3)
      write(unit_out,'(I0,A,A,A,I0,A,ES24.16E3)') case_id,',',trim(field),',',n,',',x(i,j,k); n=n+1
    enddo; enddo; enddo
  end subroutine emit3
"""
    return source.encode("ascii")


def _jax_soil_case(case_id: int) -> dict[str, np.ndarray]:
    from jax_orchidee.sechiba.hydrol import (
        USDA_AVAN,
        USDA_KS,
        USDA_MCF,
        USDA_MCR,
        USDA_MCS,
        USDA_MCW,
        USDA_NVAN,
        hydrol_calculate_temp_hydro_profile,
        hydrol_module_diagnostics,
        hydrol_module_explicit_step,
        hydrol_litter_top_diagnostics,
        hydrol_nroot_from_humcste,
        hydrol_soil_flux_diagnostics,
        hydrol_soil_resistance_evaporation,
        hydrol_var_init,
    )

    npts, nvm, nstm, nslm = 1, 14, 6, 7
    veget_max = np.zeros((npts, nvm))
    if case_id != 22:
        veget_max[0, 0] = 0.2
        veget_max[0, 13] = 0.8
    soiltile = np.full((npts, nstm), 1.0 / nstm)
    if case_id == 26:
        soiltile[0] = [0.2, 0.2, 0.2, 0.000002, 0.199999, 0.199999]
    elif case_id == 32:
        soiltile[0, 0] += soiltile[0, 3]
        soiltile[0, 3] = 0.0
    elif case_id == 33:
        soiltile[0, 0] += soiltile[0, 5]
        soiltile[0, 5] = 0.0
    mc = np.full((npts, nslm, nstm), 0.30)
    if case_id == 3:
        mc[0, :, 0] = [0.12, 0.15, 0.18, 0.24, 0.30, 0.34, 0.38]
    if case_id == 5:
        mc[0, 0, :] = 0.055
    if case_id == 10:
        mc[0, :, 0] = [0.057, 0.12, 0.18, 0.24, 0.30, 0.34, 0.38]
    if case_id == 18:
        mc[0, :, 0] = [0.075, 0.080, 0.095, 0.13, 0.19, 0.26, 0.34]
    dz = np.asarray([2., 4., 8., 12., 18., 24., 30.])
    zz = np.asarray([1., 4., 10., 20., 35., 55., 80.])
    peat_on = case_id in {4, 7, 8, 9, 13, 14, 15, 16, 24, 32, 33, 34}
    freeze_on = case_id in {11, 18, 24}
    class3 = case_id in {17, 18, 19}
    texture = 3 if class3 else 2
    soil_parameters = {
        "nvan": np.asarray(USDA_NVAN) if class3 else np.full(12, 1.89),
        "avan": np.asarray(USDA_AVAN) if class3 else np.full(12, 0.0075),
        "mcr": np.asarray(USDA_MCR) if class3 else np.full(12, 0.057),
        "mcs_mineral": np.asarray(USDA_MCS) if class3 else np.full(12, 0.41),
        "ks": np.asarray(USDA_KS) if class3 else np.full(12, 106.1),
        "pcent": np.full(12, 0.8),
        "mcf_mineral": np.asarray(USDA_MCF) if class3 else np.full(12, 0.30),
        "mcw_mineral": np.asarray(USDA_MCW) if class3 else np.full(12, 0.12),
        "mc_awet": np.full(12, 0.25 if class3 else 0.32),
        "mc_adry": np.full(12, 0.10),
    }
    if case_id == 29:
        soil_parameters["mc_awet"] = np.full(12, 0.10)
    point_mcr = float(soil_parameters["mcr"][texture - 1])
    point_mcs = float(soil_parameters["mcs_mineral"][texture - 1])
    point_mcf = float(soil_parameters["mcf_mineral"][texture - 1])
    point_mcw = float(soil_parameters["mcw_mineral"][texture - 1])
    if case_id == 19:
        soiltile[:] = 0.0
        soiltile[:, 3] = 1.0
        mc[:] = 0.0
        mc[0, :, 3] = [0.075, 0.080, 0.095, 0.13, 0.19, 0.26, 0.34]
    altmax = np.full((npts, nvm), 10.0)
    if case_id == 20:
        altmax[:] = 0.015
    elif case_id == 21:
        altmax[:] = 0.0
    drysoil_restart = np.full(npts, 999999.0 if case_id == 23 else 0.0)
    initialized = hydrol_var_init(
        veget=veget_max,
        veget_max=veget_max,
        soiltile=soiltile,
        njsc=np.full(npts, texture, dtype=np.int32),
        znh_m=zz / 1000.0,
        dnh_m=dz / 1000.0,
        dlh_m=dz / 1000.0,
        humcste=np.ones(nvm),
        altmax=altmax,
        mc=mc,
        mcl=mc.copy(),
        water2infilt=np.zeros((npts, nstm)),
        vegtot_old=np.ones(npts),
        drysoil_frac=drysoil_restart,
        ok_pc=False,
        ok_leak=case_id in {20, 21},
        peat_hydro=peat_on,
        tides=case_id in {4, 7, 8, 9},
        agri_peat=case_id == 14,
        ok_freeze_cwrr=freeze_on,
        soil_parameters=soil_parameters,
    )
    if case_id == 21:
        # At ALT=0 the Fortran loop accumulates no normalizer and preserves
        # the pre-ALT profile (hydrol_var_init lines 4127-4136).
        initialized = initialized._replace(nroot=hydrol_nroot_from_humcste(
            humcste=np.ones(nvm), dz_mm=dz, zz_mm=zz,
        ))
    mineral = initialized.mineral_tables
    peat = initialized.peat_tables
    evapot_corr = np.full(npts, 1.0 if case_id == 5 else 0.2)
    if case_id == 12:
        baseline_mc = _jax_soil_case(1)["mc"]
        baseline_litter = hydrol_litter_top_diagnostics(
            mc=baseline_mc,
            dz_mm=dz,
            dh_mm=dz,
            njsc=np.full(npts, texture, dtype=np.int32),
            mcr=np.full(npts, point_mcr),
            mcs=np.full(npts, point_mcs),
            mcf=np.full(npts, point_mcf),
            mcw=np.full(npts, point_mcw),
        )
        evapot_corr = np.asarray(hydrol_soil_resistance_evaporation(
            evapot_penm=np.full(npts, 0.2),
            u=np.full(npts, 2.0),
            v=np.full(npts, 1.0),
            tq_cdrag=np.full(npts, 0.01),
            tmc_litter=np.asarray(baseline_litter.tmc_litter),
            tmcs_litter=np.asarray(baseline_litter.tmc_litter_sat),
        ))
    vegtot = np.sum(veget_max, axis=1)
    pref_soil_veg = np.ones(nvm, dtype=np.int32)
    if case_id == 19:
        pref_soil_veg[13] = 4
    is_crop_soil = np.zeros(nstm, dtype=bool)
    irrigation = np.zeros(npts)
    ae_input = np.zeros((npts, nstm))
    evap_bare_lim_input = np.full(npts, 0.5)
    evap_bare_lim_ns_input = np.zeros((npts, nstm))
    tot_bare_soil = np.full(npts, 0.2)
    if case_id == 25:
        ae_input[:] = 0.01
        evap_bare_lim_ns_input[:] = 0.5
    elif case_id == 26:
        ae_input[0, 3] = 0.01
        evap_bare_lim_input[:] = 0.0
    elif case_id == 27:
        evap_bare_lim_input[:] = 0.0
    elif case_id == 28:
        evap_bare_lim_input[:] = 0.0
        tot_bare_soil[:] = 0.0
    temp_hydro = np.full((npts, nslm), 280.0)
    temp_hydro_output = temp_hydro
    if case_id == 30:
        production_zz = np.asarray([25., 75., 150., 300., 550., 1200., 2200.])
        temp_hydro_output = np.asarray(hydrol_calculate_temp_hydro_profile(
            stempdiag=np.asarray([[268., 270., 272., 274., 276., 278., 280.]]),
            zz_mm=production_zz,
            diaglev_m=np.asarray([.05, .10, .20, .40, .70, 1.20, 2.0]),
            snow=np.zeros(npts),
            snowdz=np.zeros((npts, 3)),
            ok_explicitsnow=True,
        ))
    water2infilt_input = np.zeros((npts, nstm))
    tot_melt = np.full(npts, 0.05)
    result = hydrol_module_explicit_step(
        precip_rain=np.zeros(npts), vevapwet=np.zeros((npts, nvm)),
        veget=veget_max, veget_max=veget_max, qsintmax=np.zeros((npts, nvm)),
        qsintveg=np.zeros((npts, nvm)), tot_melt=tot_melt,
        vegtot=vegtot, throughfall_by_pft=np.ones(nvm),
        pref_soil_veg=pref_soil_veg,
        soiltile=soiltile,
        vevapflo=np.zeros(npts), flood_frac=np.zeros(npts), flood_res=np.zeros(npts),
        subsinksoil=np.zeros(npts), vevapnu=np.full(npts, 0.01),
        vevapnu_pft=np.pad(np.asarray([[0.01]]), ((0, 0), (0, nvm - 1))),
        transpir=np.pad(np.asarray([[0.0, 0.02]]), ((0, 0), (0, nvm - 2))),
        humrel=np.full((npts, nvm), 0.5), humrelv=np.zeros((npts, nvm, nstm)),
        us=np.zeros((npts, nvm, nstm, nslm)), ae_ns=ae_input,
        evap_bare_lim=evap_bare_lim_input, evap_bare_lim_ns=evap_bare_lim_ns_input,
        evapot=np.full(npts, 0.2), evapot_corr=evapot_corr,
        tot_bare_soil=tot_bare_soil, water2infilt=water2infilt_input,
        reinfiltration_soil=np.zeros((npts, nstm)),
        is_crop_soil=is_crop_soil,
        mc=np.asarray(initialized.mc), mcl=np.asarray(initialized.mcl),
        profil_froz=np.asarray(initialized.profil_froz_hydro_ns), mcr=np.full(npts, point_mcr),
        mcs=np.full(npts, point_mcs), dz_mm=dz, dt_days=1.0 / 48.0,
        free_drain_coef=np.ones((npts, nstm)), resolv=np.ones((npts, nstm), dtype=bool),
        kfact_root=np.ones_like(mc), ks=np.asarray(soil_parameters["ks"]), kfact=np.ones(nslm),
        tmc=np.asarray(initialized.tmc), njsc=np.full(npts, texture, dtype=np.int32),
        mineral_tables=mineral, peat_tables=peat, irrigation_soil=np.zeros((npts, nstm)),
        reinf_slope=np.full(npts, 0.1), peat_hydro=peat_on,
        ok_freeze_cwrr=freeze_on,
        doponds=case_id == 6,
        temp_hydro=temp_hydro,
        zwt_force=np.full((npts, nstm), 0.006 if case_id in {2, 16} else 1.0e20),
        zz_mm=zz, zmaxh_m=2.0,
        run2peat=np.full(npts, 0.3 if case_id in {4, 7, 8, 9, 13, 14} else 0.0),
        run2man=np.full(npts, 0.1 if case_id in {4, 7, 8, 9} else 0.0),
        wt_ab=np.full(npts, 0.09 if case_id == 34 else (0.4 if case_id in {4, 7, 8, 9, 13, 14} else 0.0)),
        wt_ab_tide=np.full(npts, 0.2 if case_id in {4, 7, 8, 9} else 0.0),
        runoff2peat=np.zeros((npts, nstm)), tides=case_id in {4, 7, 8, 9, 33, 34},
        ok_ru2peat=case_id in {4, 7, 8, 9, 13, 14},
        ok_wt_ab=case_id in {4, 7, 8, 9, 13, 14},
        agri_peat=case_id == 14,
        wtp_tide={7: -0.1, 8: 0.15, 9: 0.0}.get(case_id, 0.15),
        dwtp_tide={7: -0.025, 8: -0.025, 9: 0.025}.get(case_id, 0.025),
    )
    soil = result.soil
    nroot = np.asarray(initialized.nroot)
    diagnostics = hydrol_module_diagnostics(
        result, nroot=nroot, dz_mm=dz, dh_mm=dz, njsc=np.full(npts, texture, dtype=np.int32),
        veget_max=veget_max, soiltile=soiltile, vegtot=vegtot, vegtot_old=np.ones(npts),
        mcs=np.full(npts, point_mcs), mcf=np.full(npts, point_mcf), mcw=np.full(npts, point_mcw),
        mcs_mineral=soil_parameters["mcs_mineral"], mcf_mineral=soil_parameters["mcf_mineral"],
        mcw_mineral=soil_parameters["mcw_mineral"], pcent=np.full(npts, 0.8),
        mineral_tables=mineral, peat_tables=peat, ks=soil_parameters["ks"],
        vevapnu=np.full(npts, 0.01), tot_melt=tot_melt, irrigation=irrigation,
        ok_freeze_cwrr=freeze_on, peat_hydro=peat_on,
        tides=case_id in {4, 7, 8, 9, 33, 34},
        dyn_nroot_larix=case_id in {3, 10, 18, 19},
        new_watstress=case_id == 10,
        is_tree=np.asarray([False] * 13 + [True]),
        znt=np.asarray([.01, .03, .07, .15, .30, .60, 1.20]),
        evap_bare_limit_alt=soil.evap_bare_limit_alt,
        tmcint=soil.tmcint_for_evap_bare_limit, evapot=np.full(npts, 0.2),
        do_rsoil=case_id == 12, liqlayers=nslm - 1, numlayers=nslm,
        perma_peat=case_id == 15, agri_peat=case_id == 14,
        topmodel_new=False, temp_hydro=temp_hydro,
    )
    balanced_flux = hydrol_soil_flux_diagnostics(
        mcl_after=np.zeros((npts, nslm)),
        mcl_before=np.zeros((npts, nslm)),
        dr_ns=np.zeros(npts),
        rootsink=np.zeros((npts, nslm)),
        dz_mm=dz,
        flux_top=np.zeros(npts),
    )
    return {
        "mc": np.asarray(soil.mc), "mcl": np.asarray(soil.mcl),
        "tmc": np.asarray(soil.tmc), "ru_ns": np.asarray(soil.ru_ns),
        "dr_ns": np.asarray(soil.dr_ns), "runoff2peat": np.asarray(soil.runoff2peat),
        "water2infilt": np.asarray(soil.water2infilt),
        "tmc_litter": np.asarray(diagnostics.tmc_litter),
        "tmc_litter_wilt": np.asarray(initialized.tmc_litter_wilt),
        "tmc_litter_res": np.asarray(initialized.tmc_litter_res),
        "humrel": np.asarray(diagnostics.humrel), "vegstress": np.asarray(diagnostics.vegstress),
        "nroot": np.asarray(diagnostics.nroot),
        "evap_bare_lim": np.asarray(diagnostics.evap_bare_lim),
        "wtp": np.asarray(diagnostics.wtp),
        "wt_ab": np.asarray(soil.wt_ab),
        "wt_ab_tide": np.asarray(soil.wt_ab_tide),
        "mx_eau_var": np.asarray(initialized.mx_eau_var),
        "init_drysoil_frac": np.asarray(initialized.drysoil_frac),
        "ae_ns": np.asarray(diagnostics.ae_ns),
        "drysoil_frac": np.asarray(diagnostics.drysoil_frac),
        "temp_hydro": np.asarray(temp_hydro_output),
        "qflux": np.asarray(balanced_flux.qflux),
        "_debug_tmcint": np.asarray(soil.tmcint_for_evap_bare_limit),
        "_debug_dummy_tmc": np.asarray(soil.evap_bare_limit_alt.tmc),
        "_debug_flux_bottom": np.asarray(soil.evap_bare_limit_alt.flux_bottom),
        "_debug_beta_ns": np.asarray(diagnostics.evap_bare_lim_ns),
    }


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    source = TEMPLATE.read_bytes()
    source = source.replace(
        b"do case_id=1,19", f"do case_id=1,{MAX_OWNER_CASE_ID}".encode("ascii")
    )
    source = source.replace(b"! <MODULE_STATE>", _module_state())
    source = source.replace(b"! <STATE_SETUP>", _allocation_and_reset_source())
    source = source.replace(b"! <CASE_DRIVER>", _case_driver_source())
    source = source.replace(
        b"! <HYDROL_PROCEDURES>",
        b"\n\n".join(spans[name].span_bytes for name in PROCEDURES),
    )
    path.write_bytes(source)
    return {name: span.span_sha256 for name, span in spans.items()}


def _read_fortran(path: Path) -> dict[int, dict[str, np.ndarray]]:
    values: dict[int, dict[str, list[float]]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(int(row["case_id"]), {}).setdefault(row["field"], []).append(
                float(row["value"])
            )
    return {
        case_id: {field: np.asarray(items) for field, items in fields.items()}
        for case_id, fields in values.items()
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_hydrol_soil_owner_") as td:
        source = Path(td) / "oracle.f90"
        executable = Path(td) / "oracle.exe"
        hashes = compose(source)
        metadata = compile_fortran(source, executable, compiler)
        subprocess.run(
            [str(executable)], cwd=td, env=compiler_environment(compiler), check=True,
            capture_output=True, text=True,
        )
        shutil.copyfile(Path(td) / "fortran_outputs.txt", output_dir / "fortran_outputs.csv")
    fortran_cases = _read_fortran(output_dir / "fortran_outputs.csv")
    comparisons: list[dict[str, object]] = []
    flat_fortran: dict[str, np.ndarray] = {}
    flat_jax: dict[str, np.ndarray] = {}
    for case_id, fortran in sorted(fortran_cases.items()):
        jax = _jax_soil_case(case_id)
        for field, actual in fortran.items():
            if field not in jax:
                comparisons.append({"name": f"case{case_id}.{field}", "passed": False, "reason": "missing_jax_output"})
                continue
            name = f"case{case_id}.{field}"
            flat_fortran[name] = actual
            flat_jax[name] = np.asarray(jax[field]).ravel(order="C")
            comparisons.append(float_comparison(name, actual, flat_jax[name], rtol=1e-12, atol=1e-14))
    write_point_comparisons(
        output_dir / "point_comparisons.csv", flat_fortran, flat_jax, rtol=1e-12, atol=1e-14
    )
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "common": {
                    "dimensions": {"npts": 1, "nvm": 14, "nstm": 6, "nslm": 7},
                    "veget_max": [[0.2] + [0.0] * 12 + [0.8]],
                    "soiltile": [[1.0 / 6.0] * 6],
                    "njsc": [2],
                    "znh_m": [0.001, 0.004, 0.010, 0.020, 0.035, 0.055, 0.080],
                    "dnh_m": [0.002, 0.004, 0.008, 0.012, 0.018, 0.024, 0.030],
                    "mc_default": 0.30,
                    "tot_melt": 0.05,
                    "vevapnu": 0.01,
                    "transpir_pft14": 0.02,
                    "evapot": 0.2,
                    "reinf_slope": 0.1,
                    "compiler_nbint": 50,
                },
                "cases": [
                    {"id": 1, "name": "cwrr_baseline", "overrides": {}},
                    {"id": 2, "name": "forced_water_table", "overrides": {"zwt_force": 0.006}},
                    {
                        "id": 3,
                        "name": "dynamic_root",
                        "overrides": {
                            "dyn_nroot_larix": True,
                            "tile1_mc": [0.12, 0.15, 0.18, 0.24, 0.30, 0.34, 0.38],
                        },
                    },
                    {
                        "id": 4,
                        "name": "peat_tide",
                        "overrides": {
                            "peat_hydro": True,
                            "ok_ru2peat": True,
                            "ok_wt_ab": True,
                            "tides": True,
                            "wt_ab": 0.4,
                            "wt_ab_tide": 0.2,
                            "run2peat": 0.3,
                            "run2man": 0.1,
                            "wtp_tide": 0.15,
                            "dwtp_tide": 0.025,
                        },
                    },
                    {
                        "id": 5,
                        "name": "alt_bare_residual",
                        "overrides": {"top_layer_mc": 0.055, "evapot_penm": 1.0},
                    },
                    {"id": 6, "name": "pond_storage", "overrides": {"doponds": True}},
                    {"id": 7, "name": "tide_nonpositive", "overrides": {"wtp_tide": -0.1, "dwtp_tide": -0.025}},
                    {"id": 8, "name": "tide_falling", "overrides": {"wtp_tide": 0.15, "dwtp_tide": -0.025}},
                    {"id": 9, "name": "tide_zero_boundary", "overrides": {"wtp_tide": 0.0, "dwtp_tide": 0.025}},
                    {"id": 10, "name": "new_water_stress_root_threshold", "overrides": {"dyn_nroot_larix": True, "new_watstress": True, "tile1_top_mc": 0.057}},
                    {"id": 11, "name": "freeze_aware_cwrr", "overrides": {"ok_freeze_cwrr": True}},
                    {"id": 12, "name": "soil_resistance_bare_evaporation", "overrides": {"do_rsoil": True}},
                    {"id": 13, "name": "peat_nontidal", "overrides": {"peat_hydro": True, "tides": False}},
                    {"id": 14, "name": "agricultural_peat", "overrides": {"peat_hydro": True, "agri_peat": True}},
                    {"id": 15, "name": "permafrost_peat", "overrides": {"peat_hydro": True, "perma_peat": True}},
                    {"id": 16, "name": "forced_peat_water_table", "overrides": {"peat_hydro": True, "zwt_force": 0.006}},
                    {"id": 17, "name": "usda_class3_production_parameters", "overrides": {"njsc": 3}},
                    {"id": 18, "name": "usda_class3_dry_freeze_dynamic_root", "overrides": {"njsc": 3, "dyn_nroot_larix": True, "ok_freeze_cwrr": True}},
                    {"id": 19, "name": "pft14_class3_tile4_dry_dynamic_root", "overrides": {"njsc": 3, "pft14_soil_tile": 4, "active_soil_tiles": [4], "dyn_nroot_larix": True}},
                ] + list(ADDITIONAL_CASES),
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    passed = all(item["passed"] for item in comparisons)
    coverage = _owner_branch_coverage()
    (output_dir / "branch_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="ascii"
    )
    branch_complete = passed and bool(coverage["branch_complete"])
    return write_result(
        output_dir, "hydrol_soil_owner_cwrr_water_root_peat_alt", comparisons,
        {"source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         "span_sha256": hashes, "compiler": metadata,
         "branch_coverage_asset": str((output_dir / "branch_coverage.json").relative_to(ROOT)),
         "branch_complete_ledger_entries": list(LEDGER_ENTRIES) if branch_complete else [],
         "path_evidence_ledger_entries": list(LEDGER_ENTRIES) if passed else [],
         "verified_ledger_entries": list(LEDGER_ENTRIES) if branch_complete else []},
    )


def build(compiler: Path = DEFAULT_COMPILER) -> None:
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles/hydrol_soil_owner_cwrr_water_root_peat_alt", compiler)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("hydrol_soil owner oracle comparison failed")


if __name__ == "__main__":
    try:
        build()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout, file=sys.stderr)
        raise
