from __future__ import annotations

import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax_orchidee.driver.orchestration as orch  # noqa: E402


def main() -> int:
    cfg = ROOT / "configs" / "orchidee_man_250919.yaml"
    context = orch.prepare_paper_1961_driver_context(
        cfg,
        used_run_def_path=ROOT / "outputs" / "reference_mode" / "used_run.def",
    )
    with (ROOT / "outputs" / "cache" / "paper_driver_1962_year_end_state.pkl").open("rb") as handle:
        previous = pickle.load(handle)["state"]
    state = orch.rebase_driver_state_for_year_start(previous)

    original_assemble = orch.assemble_hydrol_first_step_precall_payload
    original_hydrol = orch.run_hydrol_first_step_module_from_precall
    original_condveg = orch.run_condveg_first_step_module
    original_therm = orch.run_thermosoil_first_step_module

    def assemble_wrapper(*args, **kwargs):
        diffuco_payload = dict(kwargs.get("diffuco_payload") or {})
        keys = (
            "evap_bare_lim",
            "qsintmax",
            "qsintveg",
            "flood_frac",
            "flood_res",
            "snow",
            "snowdz",
            "snowrho",
            "snowtemp",
            "snow_age",
            "snow_nobio",
            "snow_nobio_age",
        )
        print("HYDROL_ASSEMBLE_DIFFUCO_KEYS", {key: key in diffuco_payload for key in keys})
        result = original_assemble(*args, **kwargs)
        print("HYDROL_ASSEMBLE", {"ok": result.ok, "missing": result.missing_inputs})
        return result

    def hydrol_wrapper(*args, **kwargs):
        result = original_hydrol(*args, **kwargs)
        print("HYDROL_RUN", {"ok": result.ok, "missing": result.missing_inputs})
        return result

    def condveg_wrapper(*args, **kwargs):
        result = original_condveg(*args, **kwargs)
        print("CONDVEG_RUN", {"ok": result.ok, "missing": getattr(result, "missing_inputs", None)})
        return result

    def therm_wrapper(*args, **kwargs):
        result = original_therm(*args, **kwargs)
        print("THERMOSOIL_RUN", {"ok": result.ok, "missing": getattr(result, "missing_inputs", None)})
        return result

    orch.assemble_hydrol_first_step_precall_payload = assemble_wrapper
    orch.run_hydrol_first_step_module_from_precall = hydrol_wrapper
    orch.run_condveg_first_step_module = condveg_wrapper
    orch.run_thermosoil_first_step_module = therm_wrapper

    step = orch.paper_1961_next_step_runtime_result(
        cfg,
        previous_state=state,
        year=1963,
        tstep=0,
        used_run_def_path=context.run_def_path,
        prepared_context=context,
    )
    print(
        "STEP_RESULT",
        {
            "ok": step.ok,
            "missing": step.missing_components,
            "entry": step.entry_payload is not None,
            "next_state": step.next_state is not None,
            "metadata": step.metadata is not None,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
