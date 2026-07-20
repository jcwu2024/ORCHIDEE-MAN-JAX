"""Configuration control flow from ``src_parameters/constantes.f90``.

The routines in this module reproduce ``getin_p`` ordering and guards.  A
caller's ``defaults`` mapping represents the already initialized Fortran
module variables; ``overrides`` represents run.def.  Only defaults assigned
inside the audited procedures are supplied here.  This keeps compile-time or
table structure separate from values that may vary between runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_MILLE = 1000.0


@dataclass(frozen=True)
class ConfigWarning:
    severity: int
    procedure: str
    message: str


@dataclass(frozen=True)
class ConfigResult:
    values: dict[str, Any]
    read_order: tuple[str, ...]
    warnings: tuple[ConfigWarning, ...] = ()


def _normalized(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key).upper(): value for key, value in mapping.items()}


def _parse_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    token = str(value).strip().strip(".").upper()
    if token in {"T", "TRUE", "Y", "YES", "1"}:
        return True
    if token in {"F", "FALSE", "N", "NO", "0"}:
        return False
    raise ValueError(f"{key} is not a Fortran logical: {value!r}")


def _coerce_scalar(value: Any, current: Any, key: str) -> Any:
    if isinstance(current, bool):
        return _parse_bool(value, key)
    if isinstance(current, int):
        number = float(str(value).strip().replace("D", "E").replace("d", "e"))
        if not number.is_integer():
            raise ValueError(f"{key} is not an integer: {value!r}")
        return int(number)
    if isinstance(current, float):
        return float(str(value).strip().replace("D", "E").replace("d", "e"))
    if isinstance(current, str):
        return str(value).strip()
    return value


def _coerce(value: Any, current: Any, key: str) -> Any:
    if isinstance(current, tuple):
        raw = value if isinstance(value, Sequence) and not isinstance(value, str) else str(value).split(",")
        if len(raw) != len(current):
            raise ValueError(f"{key} expects {len(current)} values, got {len(raw)}")
        return tuple(_coerce_scalar(item, template, key) for item, template in zip(raw, current))
    if isinstance(current, list):
        raw = value if isinstance(value, Sequence) and not isinstance(value, str) else str(value).split(",")
        if len(raw) != len(current):
            raise ValueError(f"{key} expects {len(current)} values, got {len(raw)}")
        return [_coerce_scalar(item, template, key) for item, template in zip(raw, current)]
    return _coerce_scalar(value, current, key)


class _Reader:
    def __init__(self, defaults: Mapping[str, Any], overrides: Mapping[str, Any]):
        self.values = _normalized(defaults)
        self.overrides = _normalized(overrides)
        self.order: list[str] = []

    def get(self, key: str, target: str | None = None) -> Any:
        key = key.upper()
        target = (target or key).upper()
        self.order.append(key)
        if target not in self.values:
            raise KeyError(f"missing initialized Fortran value for {target} (config key {key})")
        if key in self.overrides:
            self.values[target] = _coerce(self.overrides[key], self.values[target], key)
        return self.values[target]

    def result(self, warnings: tuple[ConfigWarning, ...] = ()) -> ConfigResult:
        return ConfigResult(dict(self.values), tuple(self.order), warnings)


def _structural_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} is a structural logical and must be bool")
    return value


def activate_sub_models(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    ok_stomate: bool,
    ok_dgvm: bool,
) -> ConfigResult:
    """Implement ``activate_sub_models`` lines 62-248, including all 10 arms."""

    ok_stomate = _structural_bool(ok_stomate, "ok_stomate")
    ok_dgvm = _structural_bool(ok_dgvm, "ok_dgvm")
    reader = _Reader(defaults, overrides)
    warnings: list[ConfigWarning] = []
    if ok_stomate:
        reader.get("HERBIVORES", "OK_HERBIVORES")
        reader.get("TREAT_EXPANSION")

        # The default is assigned before getin_p and depends on OK_DGVM.
        reader.values["LPJ_GAP_CONST_MORT"] = not ok_dgvm
        lpj_gap_const_mort = reader.get("LPJ_GAP_CONST_MORT")
        if ok_dgvm and lpj_gap_const_mort:
            warnings.append(
                ConfigWarning(
                    1,
                    "activate_sub_models",
                    "Both OK_DGVM and LPJ_GAP_CONST_MORT are activated.",
                )
            )

        reader.get("HARVEST_AGRI")
        disable_fire = reader.get("FIRE_DISABLE", "DISABLE_FIRE")
        if not disable_fire:
            reader.get("ALLOW_DEFOREST_FIRE")
        reader.get("SPINUP_ANALYTIC")
        reader.get("MOIST_FUNC_MOYANO")
        reader.get("DANS_RESTART")
        reader.get("PRIMING")
        reader.get("TF_DOC", "OK_TF_DOC")

        reader.values["USE_AGE_CLASS"] = False
        use_age_class = reader.get("GLUC_USE_AGE_CLASS", "USE_AGE_CLASS")
        if use_age_class:
            for key, target, default in (
                ("GLUC_NAGEC_TREE", "NAGEC_TREE", 1),
                ("GLUC_NAGEC_HERB", "NAGEC_HERB", 1),
                ("GLUC_ALLOW_FORESTRY_HARVEST", "ALLOW_FORESTRY_HARVEST", True),
                ("GLUC_SINGLE_AGE_CLASS", "SINGLEAGECLASS", False),
                ("GLUC_USE_BOUND_SPA", "USE_BOUND_SPA", False),
            ):
                reader.values[target] = default
                reader.get(key, target)
        reader.get("OK_ROTATE")
    return reader.result(tuple(warnings))


_VEGET_UNCONDITIONAL_1 = (
    "AGRICULTURE", "IMPOSE_VEG", "DYN_PEAT", "DYNPEAT_PWT", "DYNPEAT_PC",
    "OK_PEAT", "PEAT_OCCUR", "PERMA_PEAT", "FRAC1", "FRAC2", "NUMLAYERS",
    "LIQLAYERS", "AGRI_PEAT", "AGRI_PEAT_PROP", "AGRI_PEAT_MINCROP",
    "AGRI_PEAT_MAXCROP",
)


def veget_config(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> ConfigResult:
    """Implement ``veget_config`` lines 268-390, including its 3 arms."""

    reader = _Reader(defaults, overrides)
    for key in _VEGET_UNCONDITIONAL_1:
        reader.get(key)
    if reader.values["IMPOSE_VEG"]:
        reader.get("IMPOSE_SOILT")
    reader.get("LAI_MAP", "READ_LAI")
    map_pft_format = reader.get("MAP_PFT_FORMAT")
    if map_pft_format:
        reader.get("VEGET_REINIT")
        reader.get("VEGET_YEAR", "VEGET_YEAR_ORIG")
    return reader.result()


_SECHIBA_BEFORE_IMPAZE = (
    "TESTPFT", "MAXMASS_SNOW", "SNOWCRI", "IRRIG_DOSMAX", "IRRIG_DRIP",
    "MIN_WIND", "MAX_SNOW_AGE", "SNOW_TRANS", "HEIGHT_DISPLACEMENT", "Z0_BARE",
    "Z0_ICE", "TCST_SNOWA", "SNOWCRI_ALB", "VIS_DRY", "NIR_DRY", "VIS_WET",
    "NIR_WET", "ALBSOIL_VIS", "ALBSOIL_NIR", "ALB_DEADLEAF", "ALB_ICE",
    "CONDVEG_SNOWA", "ALB_BARE_MODEL", "ALB_BG_MODIS", "IMPOSE_AZE",
)
_SECHIBA_IMPAZE = (
    ("CONDVEG_Z0", "Z0_SCAL"),
    ("ROUGHHEIGHT", "ROUGHHEIGHT_SCAL"),
    ("CONDVEG_ALBVIS", "ALBEDO_SCAL_IVIS"),
    ("CONDVEG_ALBNIR", "ALBEDO_SCAL_INIR"),
    ("CONDVEG_EMIS", "EMIS_SCAL"),
)
_SECHIBA_AFTER_WATSTRESS = (
    "ROUGH_DYN", "NLAI", "LAIMAX", "DEW_VEG_POLY_COEFF", "DOWNREGULATION_CO2",
    "DOWNREGULATION_CO2_BASELEVEL", "GB_REF", "ITIMETIDE", "CONTROL_SALINITY_MIN",
    "CONTROL_INUDATE_MIN", "AGB_AGR_VEN_ALL_ST", "AGB_AGR_VEN_ALL_PN", "H_AGR_MAX_ST",
    "H_AGR_MAX_PN", "CLAYFRACTION_DEFAULT", "MIN_VEGFRAC", "STEMPDIAG_BID",
    "SHIFT_FSAT_FWET", "WTD1_BORNE", "WTD2_BORNE", "WTD3_BORNE", "WTD4_BORNE",
    "RUNF_POWER", "STREAM_POWER", "BASAREA_POWER", "MAX_ERODEPDAILY", "COEF_STC",
    "COEF_REDET", "COEF_CHANERO", "COEF_SEDDEP", "COEF_SEDDEP_FLOOD",
    "STREAMFLD_SCALER", "A_MUSLE", "FOUT_POC_COEF", "FOUT_SED_COEF", "DOCFAST_DEC",
    "DOCSLOW_DEC", "POCA_DEC", "POCS_DEC", "POCP_DEC", "K_STREAM", "K_FLOOD",
    "FLOW_COEF",
)


def config_sechiba_parameters(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> ConfigResult:
    """Implement ``config_sechiba_parameters`` lines 410-1012 and its 3 arms."""

    reader = _Reader(defaults, overrides)
    for key in _SECHIBA_BEFORE_IMPAZE:
        reader.get(key)
        if key == "SNOWCRI":
            reader.values["SNEIGE"] = reader.values["SNOWCRI"] / _MILLE
    if reader.values["IMPOSE_AZE"]:
        for key, target in _SECHIBA_IMPAZE:
            reader.get(key, target)
    new_watstress = reader.get("NEW_WATSTRESS")
    if new_watstress:
        reader.get("ALPHA_WATSTRESS")
    for key in _SECHIBA_AFTER_WATSTRESS:
        reader.get(key)
    return reader.result()


_STOMATE_BEFORE_CH4 = (
    "TOO_LONG", "TAU_FIRE", "LITTER_CRIT", "FIRE_RESIST_STRUCT", "CO2FRAC",
    "BCFRAC_COEFF", "FIREFRAC_COEFF", "REF_GREFF", "OK_MINRES", "RESERVE_TIME_TREE",
    "RESERVE_TIME_GRASS", "F_FRUIT", "ALLOC_SAP_ABOVE_GRASS", "MIN_LTOLSR",
    "MAX_LTOLSR", "Z_NITROGEN", "NLIM_TREF", "PIPE_TUNE1", "PIPE_TUNE2",
    "PIPE_TUNE3", "PIPE_TUNE4", "PIPE_DENSITY", "PIPE_K1", "PIPE_TUNE_EXP_COEFF",
    "PRECIP_CRIT", "GDD_CRIT_ESTAB", "FPC_CRIT", "ALPHA_GRASS", "ALPHA_TREE",
    "MASS_RATIO_HEART_SAP", "TAU_HUM_MONTH", "TAU_HUM_WEEK", "TAU_T2M_MONTH",
    "TAU_T2M_WEEK", "TAU_TSOIL_MONTH", "TAU_SOILHUM_MONTH", "TAU_GPP_WEEK",
    "TAU_GDD", "TAU_NGD", "COEFF_TAU_LONGTERM", "BM_SAPL_CARBRES",
    "BM_SAPL_SAPABOVE", "BM_SAPL_HEARTABOVE", "BM_SAPL_HEARTBELOW",
    "INIT_SAPL_MASS_LEAF_NAT", "INIT_SAPL_MASS_LEAF_AGRI", "INIT_SAPL_MASS_CARBRES",
    "INIT_SAPL_MASS_ROOT", "INIT_SAPL_MASS_FRUIT", "CN_SAPL_INIT", "MIGRATE_TREE",
    "MIGRATE_GRASS", "LAI_INITMIN_TREE", "LAI_INITMIN_GRASS", "DIA_COEFF",
    "MAXDIA_COEFF", "BM_SAPL_LEAF", "METABOLIC_REF_FRAC", "Z_DECOMP", "CN", "LC",
    "FRAC_SOIL_STRUCT_AA", "FRAC_SOIL_STRUCT_AB", "FRAC_SOIL_STRUCT_SA",
    "FRAC_SOIL_STRUCT_SB", "FRAC_SOIL_METAB_AA", "FRAC_SOIL_METAB_AB",
    "METABOLIC_LN_RATIO", "TAU_METABOLIC", "TAU_STRUCT", "SOIL_Q10", "TSOIL_REF",
    "LITTER_STRUCT_COEF", "MOIST_COEFF", "MOISTCONT_MIN", "FRAC_TURNOVER_DAILY",
    "TAX_MAX", "ALWAYS_INIT", "MIN_GROWTHINIT_TIME", "MOIAVAIL_ALWAYS_TREE",
    "MOIAVAIL_ALWAYS_GRASS", "T_ALWAYS_ADD", "GDDNCD_REF", "GDDNCD_CURVE",
    "GDDNCD_OFFSET", "BM_SAPL_RESCALE", "MAINT_RESP_MIN_VMAX", "MAINT_RESP_COEFF",
    "CN_RATIO_MANURE", "FRAC_CARB_AP", "FRAC_CARB_SA", "FRAC_CARB_SP", "FRAC_CARB_PA",
    "FRAC_CARB_PS", "ACTIVE_TO_PASS_CLAY_FRAC", "CARBON_TAU_IACTIVE",
    "CARBON_TAU_ISLOW", "CARBON_TAU_IPASSIVE", "FLUX_TOT_COEFF", "P_A", "P_C",
    "CF_A", "CF_C", "V_RATIO", "KA_INI", "KP_INI", "KC_INI", "TAU_FWET_MONTH",
    "TAU_LIQWT_MONTH", "PRIMING_PARAM_IACTIVE", "PRIMING_PARAM_ISLOW",
    "PRIMING_PARAM_IPASSIVE", "DOC_TAU_LABILE", "DOC_TAU_STABLE", "D_DOC",
    "CONC_DOC_RAIN", "DOC_TF_MAX", "CUE", "KD_ADS", "GPPFRAC_DORMANCE",
    "TAU_CLIMATOLOGY", "HVC1", "HVC2", "LEAF_FRAC_HVC", "TLONG_REF_MAX",
    "TLONG_REF_MIN", "NCD_MAX_YEAR", "GDD_THRESHOLD", "GREEN_AGE_EVER", "GREEN_AGE_DEC",
)
_STOMATE_CH4 = (
    "NVERT", "NS", "NDAY", "H", "RK", "DIFFAIR", "POX", "DVEG", "RKM", "XVMAX",
    "OXQ10", "SCMAX", "SR0PL", "PWATER_WET1", "PWATER_WET2", "PWATER_WET3",
    "PWATER_WET4", "RPV", "IOTHER", "RQ10", "ALPHA_CH4",
)


def config_stomate_parameters(
    defaults: Mapping[str, Any], overrides: Mapping[str, Any], *, ch4_calcul: bool
) -> ConfigResult:
    """Implement ``config_stomate_parameters`` lines 1081-2309 and its CH4 arm."""

    ch4_calcul = _structural_bool(ch4_calcul, "ch4_calcul")
    reader = _Reader(defaults, overrides)
    for key in _STOMATE_BEFORE_CH4:
        reader.get(key)
    if ch4_calcul:
        for key in _STOMATE_CH4:
            reader.get(key)
    reader.get("CODESLA")
    reader.get("PERCENT_RESIDUAL", "PRC_RESIDUAL")
    return reader.result()


class PrintLevelReader:
    """Stateful implementation of ``get_printlev`` lines 2473-2520."""

    def __init__(self) -> None:
        self._first = True
        self.printlev: int | None = None

    def get_printlev(self, modname: str, overrides: Mapping[str, Any]) -> int:
        config = _normalized(overrides)
        if self._first:
            self.printlev = 1
            if "PRINTLEV" in config:
                self.printlev = _coerce_scalar(config["PRINTLEV"], self.printlev, "PRINTLEV")
            self._first = False
        assert self.printlev is not None
        result = self.printlev
        local_key = f"PRINTLEV_{modname}".upper()
        if local_key in config:
            result = _coerce_scalar(config[local_key], result, local_key)
        return result
