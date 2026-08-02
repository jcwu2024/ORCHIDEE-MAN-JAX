"""PFT14 STOMATE parameter loading for the ORCHIDEE-MAN paper case.

This module intentionally reads only run-definition files from this repository.
Reference-run values override base-run values, matching the paper script flow;
no parameter is inferred from model output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from jax_orchidee.driver.domain import load_case_config
from jax_orchidee.driver.init import parse_run_def_bool, read_run_scalars
from jax_orchidee.parameters.pft_catalog import PFTRunLayout
from jax_orchidee.stomate.carbon_kernels import (
    NPARTS,
    NLEAFAGES,
    SENESCENCE_COLD,
    SENESCENCE_CROP,
    SENESCENCE_DRY,
    SENESCENCE_MIXED,
    SENESCENCE_NONE,
)


BASE_RUN_DEF = Path("fortran_run_scripts/paper_250919/run.def.vn")
REFERENCE_RUN_DEF = Path(
    "fortran_run_scripts/paper_250919/sen_reference_arg2_1.0_001.0-071.0/"
    "I10/S2_63.206_0.0876_0.2019_50.658/run.def_63.206_0.0876_0.2019_50.658"
)
USED_RUN_DEF = Path("outputs/server_1961_trace_full_20260623/run/used_run.def")
FORTRAN_UNDEF_INT = 999_999_999

PFT14_PARAMETER_KEYS = (
    "VCMAX25",
    "ARJV",
    "SLA",
    "MAINT_RESP_SLOPE_C",
    "RESIDENCE_TIME",
    "ALLOC_MIN",
    "LAI_MAX",
    "LEAFLIFE_TAB",
    "ALLOC_AGR_ST",
    "ALLOC_AGR_PN",
)

PAPER_CASE_INDEXED_FLOAT_KEYS = (
    "VCMAX25",
    "EXT_COEFF",
    "R0",
    "S0",
    "FRAC_GROWTHRESP",
    "AVAILABILITY_FACT",
    "MAINT_RESP_SLOPE_C",
    "MAINT_RESP_SLOPE_B",
    "MAINT_RESP_SLOPE_A",
    "SLA_MAX",
    "SLA_MIN",
    "LAI_MAX_TO_HAPPY",
    "PHENO_TYPE",
    "PHENO_GDD_CRIT_C",
    "PHENO_GDD_CRIT_B",
    "PHENO_GDD_CRIT_A",
    "PHENO_MOIGDD_T_CRIT",
    "NCDGDD_TEMP",
    "HUM_MIN_TIME",
    "TAU_SAP",
    "TAU_LEAFINIT",
    "TAU_FRUIT",
    "ALLOC_MAX",
    "DEMI_ALLOC",
    "LEAFFALL",
    "LEAFAGECRIT",
    "HUM_FRAC",
    "SENESCENCE_HUM",
    "NOSENESCENCE_HUM",
    "MAX_TURNOVER_TIME",
    "MIN_TURNOVER_TIME",
    "MIN_LEAF_AGE_FOR_SENESCENCE",
    "SENESCENCE_TEMP_C",
    "SENESCENCE_TEMP_B",
    "SENESCENCE_TEMP_A",
    "GDD_SENESCENCE",
    "TCM_CRIT",
    "CM_ZERO_LEAF",
    "CM_ZERO_SAPABOVE",
    "CM_ZERO_SAPBELOW",
    "CM_ZERO_HEARTABOVE",
    "CM_ZERO_HEARTBELOW",
    "CM_ZERO_ROOT",
    "CM_ZERO_FRUIT",
    "CM_ZERO_CARBRES",
    "CM_ZERO_AGRSAPST",
    "CM_ZERO_AGRSAPPN",
    "CM_ZERO_AGRHRTST",
    "CM_ZERO_AGRHRTPN",
)

PAPER_CASE_INDEXED_BOOL_KEYS = ("OK_LAIDEV",)
PAPER_CASE_INDEXED_STRING_KEYS = ("SENESCENCE_TYPE",)
CM_ZERO_KEYS = (
    "CM_ZERO_LEAF",
    "CM_ZERO_SAPABOVE",
    "CM_ZERO_SAPBELOW",
    "CM_ZERO_HEARTABOVE",
    "CM_ZERO_HEARTBELOW",
    "CM_ZERO_ROOT",
    "CM_ZERO_FRUIT",
    "CM_ZERO_CARBRES",
    "CM_ZERO_AGRSAPST",
    "CM_ZERO_AGRSAPPN",
    "CM_ZERO_AGRHRTST",
    "CM_ZERO_AGRHRTPN",
)

_PHENO_MODEL_MTC = (
    "none",
    "none",
    "moi",
    "none",
    "none",
    "ncdgdd",
    "none",
    "ncdgdd",
    "ngd",
    "moigdd",
    "moi_C4",
    "moigdd",
    "moigdd",
    "moigdd",
    "moigdd",
    "moigdd",
    "moigdd",
    "none",
)
_LEAF_TAB_MTC = np.asarray([4, 1, 1, 2, 1, 1, 2, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 1], dtype=np.int32)
_IS_GRASSLAND_MANAG_MTC = np.zeros(18, dtype=bool)
_PASTURE_MTC = np.zeros(18, dtype=bool)
_TMIN_CRIT_MTC = np.asarray(
    [-9999.0, 0.0, 0.0, -30.0, -14.0, -30.0, -45.0, -45.0, -60.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 0.0],
    dtype=np.float64,
)

SENESCENCE_TYPE_CODES = {
    "none": SENESCENCE_NONE,
    "crop": SENESCENCE_CROP,
    "cold": SENESCENCE_COLD,
    "dry": SENESCENCE_DRY,
    "mixed": SENESCENCE_MIXED,
}

MANGROVE_GLOBAL_KEYS = (
    "ITIMETIDE",
    "CONTROL_SALINITY_MIN",
    "CONTROL_INUDATE_MIN",
    "AGB_AGR_VEN_ALL_ST",
    "AGB_AGR_VEN_ALL_PN",
    "H_AGR_MAX_ST",
    "H_AGR_MAX_PN",
    "WEIGHT_VEN_A",
    "WEIGHT_VEN_B",
    "DEN_MAN",
)

PFT14_FORTRAN_PROVENANCE = {
    "VCMAX25": "src_parameters/pft_parameters.f90, config_stomate_pft_parameters, lines 3234-3248",
    "ARJV": "src_parameters/pft_parameters.f90, config_stomate_pft_parameters, lines 3234-3248",
    "SLA": "src_parameters/pft_parameters.f90, config_stomate_pft_parameters, lines 4147-4153",
    "RESIDENCE_TIME": "src_parameters/pft_parameters.f90, config_stomate_pft_parameters, lines 4163-4169",
    "ALLOC_AGR_ST": "src_parameters/pft_parameters.f90, config_stomate_pft_parameters, lines 4192-4206",
    "ALLOC_AGR_PN": "src_parameters/pft_parameters.f90, config_stomate_pft_parameters, lines 4192-4206",
    "MAINT_RESP_SLOPE_C": "src_parameters/pft_parameters.f90, config_stomate_pft_parameters, lines 4220-4226",
    "LAI_MAX": "src_parameters/pft_parameters.f90, config_stomate_pft_parameters, lines 4416-4422",
    "ALLOC_MIN": "src_parameters/pft_parameters.f90, config_stomate_pft_parameters, lines 4532-4538",
    "LEAFLIFE_TAB": "src_parameters/pft_parameters.f90, config_stomate_pft_parameters, lines 4556-4562",
}

MANGROVE_GLOBAL_FORTRAN_PROVENANCE = (
    "src_parameters/constantes.f90, config_stomate_parameters, lines 749-805"
)


@dataclass(frozen=True)
class PFT14Parameters:
    """Audited PFT14 parameter subset used by the Phase 0/1 STOMATE path.

    Fortran provenance:
    `pft_parameters.f90`, subroutine `config_stomate_pft_parameters`,
    lines 3234-3248, 4147-4153, 4163-4169, 4192-4206, 4220-4226,
    4416-4422, 4532-4538, and 4556-4562.
    Mangrove global provenance: `constantes.f90`,
    subroutine `config_stomate_parameters`, lines 749-805.
    Run protocol provenance: `run.def.vn`, lines 1-31 and 37-49; reference
    override `run.def_63.206_0.0876_0.2019_50.658`, lines 1-31 and 37-49.
    """

    pft_index_fortran: int
    values: dict[str, float]
    mangrove_globals: dict[str, float]
    source_paths: tuple[Path, Path]

    def __getitem__(self, key: str) -> float:
        return self.values[key]


@dataclass(frozen=True)
class PaperCaseStomateParameterBundle:
    """Source-backed STOMATE static/run parameter arrays for the paper case.

    Fortran provenance: ``src_parameters/pft_parameters.f90``,
    ``config_stomate_pft_parameters`` lines 3410-3459, 4155-4242,
    4394-4670, and ``src_stomate/stomate_data.f90`` lines 151-178 and
    585-587. MTC table provenance:
    ``src_parameters/constantes_mtc.f90`` lines 76-84, 91-96, 495-555, and
    790-797. Run protocol provenance: ``used_run.def`` materialized by the
    audited 1961 server trace plus the selected paper reference run.def
    sensitivity override.
    """

    nvm: int
    pft_layout: PFTRunLayout
    pft_ids: tuple[str, ...]
    fortran_pft_ids: np.ndarray
    pft_to_mtc: np.ndarray
    natural: np.ndarray
    pasture: np.ndarray
    is_tree: np.ndarray
    is_peat: np.ndarray
    pheno_model: tuple[str, ...]
    pheno_is_none: np.ndarray
    is_grassland_manag: np.ndarray
    ok_laidev: np.ndarray
    lpj_gap_const_mort: bool
    ok_dgvm: bool
    r0: np.ndarray
    s0: np.ndarray
    ext_coeff: np.ndarray
    lai_max: np.ndarray
    lai_max_to_happy: np.ndarray
    tau_leafinit: np.ndarray
    alloc_min: np.ndarray
    alloc_max: np.ndarray
    demi_alloc: np.ndarray
    alloc_agr_st: np.ndarray
    alloc_agr_pn: np.ndarray
    frac_growthresp: np.ndarray
    availability_fact: np.ndarray
    residence_time: np.ndarray
    leaf_tab: np.ndarray
    pheno_type: np.ndarray
    tmin_crit: np.ndarray
    tcm_crit: np.ndarray
    pheno_gdd_crit: np.ndarray
    ncdgdd_temp: np.ndarray
    hum_min_time: np.ndarray
    min_leaf_age_for_senescence: np.ndarray
    gdd_senescence: np.ndarray
    senescence_temp: np.ndarray
    hum_frac: np.ndarray
    senescence_hum: np.ndarray
    nosenescence_hum: np.ndarray
    max_turnover_time: np.ndarray
    min_turnover_time: np.ndarray
    leaffall: np.ndarray
    leafagecrit: np.ndarray
    leaf_timecst: np.ndarray
    vcmax25: np.ndarray
    grm_n_limitation: bool
    lai_initmin: np.ndarray
    tau_fruit: np.ndarray
    tau_sap: np.ndarray
    sla_max: np.ndarray
    sla_min: np.ndarray
    maint_resp_slope: np.ndarray
    coeff_maint_zero: np.ndarray
    senescence_type: np.ndarray
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class PFTPhysiologyLabels:
    """Physiological labels derived from ``leaf_tab`` and ``pheno_model``.

    Fortran provenance: ``pft_parameters.f90``, ``pft_parameters_init``
    lines 340-363 and 491-513; ``config_pft_parameters`` lines 2962-2982.
    """

    is_tree: np.ndarray
    is_deciduous: np.ndarray
    is_evergreen: np.ndarray
    is_needleleaf: np.ndarray


@dataclass(frozen=True)
class PFTParameterSectionSwitches:
    """Branch selections for the parameter-family configuration sections.

    Fortran provenance: ``pft_parameters.f90``, ``pft_parameters_init``
    lines 370-405, 519-829, 833-874, and 878-976;
    ``config_sechiba_pft_parameters`` lines 3822-3892.
    """

    load_sechiba: bool
    zero_offline_cwrr_throughfall: bool
    load_stomate: bool
    load_bvoc: bool


@dataclass(frozen=True)
class AgeClassConfig:
    """Resolved age-class configuration arrays and config-read selections."""

    nvmap: int
    agec_group: np.ndarray
    read_age_class_bound: bool
    no_age_class_warning: bool


@dataclass(frozen=True)
class AgeClassLayout:
    """Fortran-style, zero-based start indices and counts for real PFTs."""

    start_index: np.ndarray
    nagec_pft: np.ndarray


def _strip_inline_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _parse_scalar(value: str) -> float:
    cleaned = value.strip().replace("D", "E").replace("d", "e")
    return float(cleaned)


def resolve_pft_parameter_main(
    *,
    first_call: bool,
    nvm: int,
    nvmc: int,
    pft_to_mtc: Sequence[int] | None = None,
) -> np.ndarray | None:
    """Apply the scientific input contract of ``pft_parameters_main``.

    The returned mapping keeps Fortran's one-based MTC indices. ``None`` is
    the exact no-op result of the non-first-call arm. Diagnostic printing is
    intentionally outside this scientific contract.

    Fortran provenance: ``pft_parameters.f90``, ``pft_parameters_main``,
    lines 97-169 (branch sites 97, 127, 129, 140, 149, and 158).
    """

    if not first_call:
        return None
    if nvm < 1 or nvmc < 1:
        raise ValueError("nvm and nvmc must be positive")
    if pft_to_mtc is None:
        mapping = np.arange(1, nvm + 1, dtype=np.int32)
    else:
        mapping = np.asarray(pft_to_mtc, dtype=np.int32)
        if mapping.shape != (nvm,):
            raise ValueError(f"pft_to_mtc must have shape ({nvm},)")
    if nvm != nvmc and mapping[0] == FORTRAN_UNDEF_INT:
        raise ValueError("PFT_TO_MTC is required when nvm differs from nvmc")
    if np.any((mapping > nvmc) | (mapping <= 0)):
        raise ValueError("PFT_TO_MTC contains a metaclass outside 1..nvmc")
    if mapping[0] != 1:
        raise ValueError("the first PFT must map to bare-soil MTC 1")
    if np.any(mapping[1:] == 1):
        raise ValueError("only the first PFT may map to bare-soil MTC 1")
    return mapping.copy()


def select_humcste_reference(
    zmaxh: float,
    pft_to_mtc: Sequence[int],
    humcste_ref2m: Sequence[float],
    humcste_ref4m: Sequence[float],
) -> np.ndarray:
    """Select and map the 2 m/4 m root profile table exactly as Fortran does.

    Any depth other than exactly 4 m uses the 2 m table, including the
    explicit 2 m arm and the fallback arm.

    Fortran provenance: ``pft_parameters.f90``, ``pft_parameters_init``,
    lines 265-273 and 424-433.
    """

    mapping = np.asarray(pft_to_mtc, dtype=np.int64)
    table = np.asarray(humcste_ref4m if zmaxh == 4.0 else humcste_ref2m, dtype=np.float64)
    if mapping.ndim != 1:
        raise ValueError("pft_to_mtc must be one-dimensional")
    if np.any((mapping < 1) | (mapping > table.size)):
        raise ValueError("pft_to_mtc index leaves the selected humcste table")
    return table[mapping - 1]


def derive_pft_physiology_labels(
    leaf_tab: Sequence[int], pheno_model: Sequence[str]
) -> PFTPhysiologyLabels:
    """Derive tree, phenology, and leaf-form flags for arbitrary ``nvm``."""

    leaves = np.asarray(leaf_tab, dtype=np.int32)
    phenology = np.asarray(tuple(pheno_model), dtype=str)
    if leaves.ndim != 1 or phenology.shape != leaves.shape:
        raise ValueError("leaf_tab and pheno_model must be same-length vectors")
    is_tree = leaves <= 2
    pheno_is_none = phenology == "none"
    return PFTPhysiologyLabels(
        is_tree=is_tree,
        is_deciduous=is_tree & ~pheno_is_none,
        is_evergreen=is_tree & pheno_is_none,
        is_needleleaf=leaves == 2,
    )


def resolve_pft_parameter_section_switches(
    *,
    ok_sechiba: bool,
    hydrol_cwrr: bool,
    offline_mode: bool,
    ok_stomate: bool,
    ok_bvoc: bool,
) -> PFTParameterSectionSwitches:
    """Resolve parameter-family branch selections without treating IO as science."""

    return PFTParameterSectionSwitches(
        load_sechiba=bool(ok_sechiba),
        zero_offline_cwrr_throughfall=bool(ok_sechiba and hydrol_cwrr and offline_mode),
        load_stomate=bool(ok_stomate),
        load_bvoc=bool(ok_bvoc),
    )


def apply_configured_pft_constraints(
    natural: Sequence[bool],
    is_grassland_manag: Sequence[bool],
    pref_soil_veg: Sequence[int],
    *,
    nstm: int,
    use_age_class: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the scientific writes in ``config_pft_parameters``.

    The invalid-``nstm`` branch only prints in Fortran and therefore does not
    replace ``nstm`` here. The soil-tile clamp is applied only for ``nstm < 6``
    exactly as lines 3424-3428 specify. ``pref_soil_veg`` is the pre-``getin``
    value: the subsequent line-3440 ``PREF_SOIL_VEG`` read may replace it.
    """

    natural_array = np.asarray(natural, dtype=bool)
    managed = np.asarray(is_grassland_manag, dtype=bool)
    soil_tiles = np.asarray(pref_soil_veg, dtype=np.int32)
    if (
        natural_array.ndim != 1
        or managed.shape != natural_array.shape
        or soil_tiles.shape != natural_array.shape
    ):
        raise ValueError("natural, is_grassland_manag, and pref_soil_veg must be same-length vectors")
    if use_age_class and not np.any(managed):
        raise ValueError("age classes require at least one managed grassland")
    configured_natural = natural_array.copy()
    configured_natural[managed] = False
    configured_soil_tiles = soil_tiles.copy()
    if nstm < 6:
        configured_soil_tiles = np.minimum(configured_soil_tiles, nstm)
    return configured_natural, configured_soil_tiles


def crop_rotation_sowing_keys(cyc_rot_max: int) -> tuple[str, ...]:
    """Return the exact conditional sowing-date keys read by Fortran.

    ``SP_IPLT0`` is unconditional, while ``SP_IPLT1`` and ``SP_IPLT2`` are
    selected at branch sites 3479 and 3482 respectively. This function models
    config selection only; it does not manufacture parameter values.

    Fortran provenance: ``pft_parameters.f90``, ``config_pft_parameters``,
    lines 3472-3484.
    """

    keys = ["SP_IPLT0"]
    if cyc_rot_max > 1:
        keys.append("SP_IPLT1")
    if cyc_rot_max > 2:
        keys.append("SP_IPLT2")
    return tuple(keys)


def resolve_age_class_config(
    nvm: int,
    *,
    use_age_class: bool,
    nvmap: int | None = None,
    agec_group: Sequence[int] | None = None,
    single_age_class: bool = False,
    use_bound_spa: bool = False,
) -> AgeClassConfig:
    """Resolve the config-selection arms at Fortran lines 3568-3589."""

    if nvm < 1:
        raise ValueError("nvm must be positive")
    if not use_age_class:
        return AgeClassConfig(
            nvmap=nvm,
            agec_group=np.arange(1, nvm + 1, dtype=np.int32),
            read_age_class_bound=False,
            no_age_class_warning=False,
        )
    resolved_nvmap = nvm if nvmap is None else int(nvmap)
    groups = (
        np.arange(1, nvm + 1, dtype=np.int32)
        if agec_group is None
        else np.asarray(agec_group, dtype=np.int32)
    )
    if not 1 <= resolved_nvmap <= nvm:
        raise ValueError("nvmap must be in 1..nvm")
    if groups.shape != (nvm,):
        raise ValueError(f"agec_group must have shape ({nvm},)")
    return AgeClassConfig(
        nvmap=resolved_nvmap,
        agec_group=groups.copy(),
        read_age_class_bound=not use_bound_spa,
        no_age_class_warning=resolved_nvmap == nvm and not single_age_class,
    )


def build_age_class_layout(
    agec_group: Sequence[int],
    *,
    nvmap: int,
    is_tree: Sequence[bool],
    nagec_tree: int,
    nagec_herb: int,
) -> AgeClassLayout:
    """Build and validate age-class starts/counts for arbitrary ``nvm``.

    Results use zero-based indices for JAX/Python. The loop reproduces the
    Fortran contiguous-group count, including its single-age-class early-exit
    condition, while checking bounds before indexing.

    Fortran provenance: ``pft_parameters.f90``,
    ``config_stomate_pft_parameters``, lines 4672-4725.
    """

    groups = np.asarray(agec_group, dtype=np.int32)
    tree = np.asarray(is_tree, dtype=bool)
    if groups.ndim != 1 or tree.shape != groups.shape:
        raise ValueError("agec_group and is_tree must be same-length vectors")
    nvm = groups.size
    if not 1 <= nvmap <= nvm:
        raise ValueError("nvmap must be in 1..nvm")
    starts = np.full(nvmap, -1, dtype=np.int32)
    counts = np.full(nvmap, -1, dtype=np.int32)
    single_age_mode = nagec_tree == 1 and nagec_herb == 1
    for group_fortran in range(1, nvmap + 1):
        matches = np.flatnonzero(groups == group_fortran)
        if matches.size == 0:
            raise ValueError(f"age class group {group_fortran} has no PFT")
        start = int(matches[0])
        count = 0
        while start + count < nvm and groups[start + count] == group_fortran:
            count += 1
            next_fortran = start + count + 1
            if (single_age_mode and next_fortran > nvmap) or next_fortran > nvm:
                break
        starts[group_fortran - 1] = start
        counts[group_fortran - 1] = count
    for index, (start, count) in enumerate(zip(starts, counts, strict=True), start=1):
        if count > 1:
            expected = nagec_tree if tree[start] else nagec_herb
            if count != expected:
                kind = "tree" if tree[start] else "herb"
                raise ValueError(f"age class group {index} has {count} {kind} classes; expected {expected}")
    return AgeClassLayout(start_index=starts, nagec_pft=counts)


def read_run_def_scalars(path: str | Path) -> dict[str, float]:
    """Read numeric scalar assignments from a run-definition file.

    The parser is deliberately narrow: it keeps only `KEY=value` lines whose
    values parse as floats. This covers audited PFT14 and mangrove constants
    while ignoring logical flags and path strings.
    """

    values: dict[str, float] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = _strip_inline_comment(raw_line)
            if not line or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip().upper()
            raw_value = raw_value.strip()
            if not key or not raw_value:
                continue
            try:
                values[key] = _parse_scalar(raw_value)
            except ValueError:
                continue
    return values


def read_run_def_values(path: str | Path) -> dict[str, str]:
    """Read literal run.def assignments with comments removed."""

    values: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = _strip_inline_comment(raw_line)
            if not line or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip().upper()
            raw_value = raw_value.strip()
            if key and raw_value:
                values[key] = raw_value
    return values


def _pft_key(base_key: str, pft_index_fortran: int) -> str:
    return f"{base_key}__{pft_index_fortran:05d}"


def load_pft14_parameters(
    root: str | Path = ".",
    *,
    base_run_def: str | Path = BASE_RUN_DEF,
    reference_run_def: str | Path = REFERENCE_RUN_DEF,
) -> PFT14Parameters:
    """Load audited PFT14 parameters with reference-run override semantics.

    For each PFT parameter, the Fortran-facing run.def key is
    `<PARAMETER>__00014`. The reference sensitivity run overrides base
    `run.def.vn`; values absent from the reference run fall back to base.
    Global mangrove constants are read by their plain run.def keys with the
    same override rule.
    """

    root = Path(root)
    base_path = root / base_run_def
    reference_path = root / reference_run_def
    base = read_run_def_scalars(base_path)
    reference = read_run_def_scalars(reference_path)
    merged = {**base, **reference}

    pft_values: dict[str, float] = {}
    missing: list[str] = []
    for key in PFT14_PARAMETER_KEYS:
        run_def_key = _pft_key(key, 14)
        if run_def_key not in merged:
            missing.append(run_def_key)
        else:
            pft_values[key] = merged[run_def_key]

    mangrove_globals: dict[str, float] = {}
    for key in MANGROVE_GLOBAL_KEYS:
        if key in merged:
            mangrove_globals[key] = merged[key]

    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"Missing required PFT14 run.def keys: {joined}")

    return PFT14Parameters(
        pft_index_fortran=14,
        values=pft_values,
        mangrove_globals=mangrove_globals,
        source_paths=(base_path, reference_path),
    )


def _merge_run_def_values(paths: tuple[Path, ...]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for path in paths:
        if path.exists():
            merged.update(read_run_def_values(path))
    return merged


def _indexed_float_vector(
    values: dict[str, str],
    name: str,
    fortran_pft_ids: Sequence[int],
) -> np.ndarray:
    result = []
    missing = []
    for index in fortran_pft_ids:
        key = _pft_key(name, index)
        if key not in values:
            missing.append(key)
        else:
            result.append(_parse_scalar(values[key]))
    if missing:
        raise KeyError("Missing required indexed run.def keys: " + ", ".join(missing))
    return np.asarray(result, dtype=np.float64)


def _indexed_bool_vector(
    values: dict[str, str],
    name: str,
    fortran_pft_ids: Sequence[int],
) -> np.ndarray:
    return np.asarray(
        [parse_run_def_bool(values[_pft_key(name, index)]) for index in fortran_pft_ids],
        dtype=bool,
    )


def _indexed_string_vector(
    values: dict[str, str],
    name: str,
    fortran_pft_ids: Sequence[int],
) -> tuple[str, ...]:
    return tuple(values[_pft_key(name, index)].strip() for index in fortran_pft_ids)


def _scalar_bool(values: dict[str, str], name: str) -> bool:
    if name not in values:
        raise KeyError(name)
    return parse_run_def_bool(values[name])


def _scalar_float(values: dict[str, str], name: str) -> float:
    if name not in values:
        raise KeyError(name)
    return _parse_scalar(values[name])


def _senescence_type_codes(labels: tuple[str, ...]) -> np.ndarray:
    codes = []
    for label in labels:
        key = label.strip().lower()
        if key not in SENESCENCE_TYPE_CODES:
            raise ValueError(f"unsupported SENESCENCE_TYPE label {label!r}")
        codes.append(SENESCENCE_TYPE_CODES[key])
    return np.asarray(codes, dtype=np.int32)


def load_paper_case_stomate_parameter_bundle(
    config_path: str | Path,
    *,
    used_run_def: str | Path | None = None,
    base_run_def: str | Path = BASE_RUN_DEF,
    reference_run_def: str | Path = REFERENCE_RUN_DEF,
) -> PaperCaseStomateParameterBundle:
    """Load source-backed STOMATE static parameter arrays for the paper case.

    ``used_run.def`` supplies materialized Fortran defaults and, when passed
    explicitly, is the trace-run truth. Without an explicit ``used_run.def``,
    the selected paper reference run.def is applied last so calibrated
    sensitivity values such as ``VCMAX25__00014`` and ``ALLOC_MIN__00014``
    override base/default values without hard-coded patches.
    """

    config_path = Path(config_path)
    root = Path(load_case_config(config_path)["paths"]["workspace_root"])
    used_path = root / (USED_RUN_DEF if used_run_def is None else Path(used_run_def))
    base_path = root / base_run_def
    reference_path = root / reference_run_def
    if used_run_def is None:
        values = _merge_run_def_values((used_path, base_path, reference_path))
        scalar_run_def_path = reference_path
    else:
        values = _merge_run_def_values((base_path, reference_path, used_path))
        scalar_run_def_path = used_path
    run_scalars = read_run_scalars(config_path, run_def_path=scalar_run_def_path)
    nvm = int(run_scalars.nvm)
    fortran_pft_ids = tuple(int(value) for value in run_scalars.fortran_pft_ids)

    float_vectors = {
        name: _indexed_float_vector(values, name, fortran_pft_ids)
        for name in PAPER_CASE_INDEXED_FLOAT_KEYS
    }
    ok_laidev = _indexed_bool_vector(values, "OK_LAIDEV", fortran_pft_ids)
    senescence_labels = _indexed_string_vector(values, "SENESCENCE_TYPE", fortran_pft_ids)
    mtc_index = np.asarray(run_scalars.pft_to_mtc, dtype=np.int32) - 1
    leaf_tab = _LEAF_TAB_MTC[mtc_index].astype(np.int32)
    pheno_model = tuple(_PHENO_MODEL_MTC[int(index)] for index in mtc_index)
    is_grassland_manag = _IS_GRASSLAND_MANAG_MTC[mtc_index].astype(bool)
    tmin_crit = _TMIN_CRIT_MTC[mtc_index].astype(np.float64)
    lai_initmin_tree = _scalar_float(values, "LAI_INITMIN_TREE")
    lai_initmin_grass = _scalar_float(values, "LAI_INITMIN_GRASS")
    lai_initmin = np.where(leaf_tab == 3, lai_initmin_grass, lai_initmin_tree).astype(np.float64)
    cm_zero = np.stack([float_vectors[name] for name in CM_ZERO_KEYS], axis=1)

    return PaperCaseStomateParameterBundle(
        nvm=nvm,
        pft_layout=run_scalars.pft_layout,
        pft_ids=run_scalars.pft_ids,
        fortran_pft_ids=np.asarray(run_scalars.fortran_pft_ids, dtype=np.int32),
        pft_to_mtc=np.asarray(run_scalars.pft_to_mtc, dtype=np.int32),
        natural=np.asarray(run_scalars.natural, dtype=bool),
        pasture=_PASTURE_MTC[mtc_index].astype(bool),
        is_tree=np.asarray(run_scalars.is_tree, dtype=bool),
        is_peat=np.asarray(run_scalars.is_peat, dtype=bool),
        pheno_model=pheno_model,
        pheno_is_none=np.asarray([model == "none" for model in pheno_model], dtype=bool),
        is_grassland_manag=is_grassland_manag,
        ok_laidev=ok_laidev,
        lpj_gap_const_mort=_scalar_bool(values, "LPJ_GAP_CONST_MORT"),
        ok_dgvm=_scalar_bool(values, "STOMATE_OK_DGVM"),
        r0=float_vectors["R0"],
        s0=float_vectors["S0"],
        ext_coeff=float_vectors["EXT_COEFF"],
        lai_max=_indexed_float_vector(values, "LAI_MAX", fortran_pft_ids),
        lai_max_to_happy=float_vectors["LAI_MAX_TO_HAPPY"],
        tau_leafinit=float_vectors["TAU_LEAFINIT"],
        alloc_min=_indexed_float_vector(values, "ALLOC_MIN", fortran_pft_ids),
        alloc_max=float_vectors["ALLOC_MAX"],
        demi_alloc=float_vectors["DEMI_ALLOC"],
        alloc_agr_st=_indexed_float_vector(values, "ALLOC_AGR_ST", fortran_pft_ids),
        alloc_agr_pn=_indexed_float_vector(values, "ALLOC_AGR_PN", fortran_pft_ids),
        frac_growthresp=float_vectors["FRAC_GROWTHRESP"],
        availability_fact=float_vectors["AVAILABILITY_FACT"],
        residence_time=_indexed_float_vector(values, "RESIDENCE_TIME", fortran_pft_ids),
        leaf_tab=leaf_tab,
        pheno_type=float_vectors["PHENO_TYPE"].astype(np.int32),
        tmin_crit=tmin_crit,
        tcm_crit=float_vectors["TCM_CRIT"],
        pheno_gdd_crit=np.stack(
            [float_vectors["PHENO_GDD_CRIT_C"], float_vectors["PHENO_GDD_CRIT_B"], float_vectors["PHENO_GDD_CRIT_A"]],
            axis=1,
        ),
        ncdgdd_temp=float_vectors["NCDGDD_TEMP"],
        hum_min_time=float_vectors["HUM_MIN_TIME"],
        min_leaf_age_for_senescence=float_vectors["MIN_LEAF_AGE_FOR_SENESCENCE"],
        gdd_senescence=float_vectors["GDD_SENESCENCE"],
        senescence_temp=np.stack(
            [float_vectors["SENESCENCE_TEMP_C"], float_vectors["SENESCENCE_TEMP_B"], float_vectors["SENESCENCE_TEMP_A"]],
            axis=1,
        ),
        hum_frac=float_vectors["HUM_FRAC"],
        senescence_hum=float_vectors["SENESCENCE_HUM"],
        nosenescence_hum=float_vectors["NOSENESCENCE_HUM"],
        max_turnover_time=float_vectors["MAX_TURNOVER_TIME"],
        min_turnover_time=float_vectors["MIN_TURNOVER_TIME"],
        leaffall=float_vectors["LEAFFALL"],
        leafagecrit=float_vectors["LEAFAGECRIT"],
        leaf_timecst=(float_vectors["LEAFAGECRIT"] / float(NLEAFAGES)).astype(np.float64),
        vcmax25=float_vectors["VCMAX25"],
        grm_n_limitation=parse_run_def_bool(values.get("GRM_N_LIMITATION", "FALSE")),
        lai_initmin=lai_initmin,
        tau_fruit=float_vectors["TAU_FRUIT"],
        tau_sap=float_vectors["TAU_SAP"],
        sla_max=float_vectors["SLA_MAX"],
        sla_min=float_vectors["SLA_MIN"],
        maint_resp_slope=np.stack(
            [float_vectors["MAINT_RESP_SLOPE_C"], float_vectors["MAINT_RESP_SLOPE_B"], float_vectors["MAINT_RESP_SLOPE_A"]],
            axis=1,
        ),
        coeff_maint_zero=cm_zero.reshape((nvm, NPARTS)),
        senescence_type=_senescence_type_codes(senescence_labels),
        source_paths=(used_path, base_path, reference_path),
    )
