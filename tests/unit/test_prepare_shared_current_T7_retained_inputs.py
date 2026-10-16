from dataclasses import fields
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.hpc import prepare_shared_current_T7_retained_inputs as admission
from jax_orchidee.driver.sechiba_boundary import IntersurfFirstStepPayload


def _verified_inputs(tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    forcing = root / "forcing" / "cruncep_twodeg_1991.nc"
    co2 = root / "INPUTDIR_ZZ" / "global_co2_ann_1700_2017.txt"
    forcing.parent.mkdir(parents=True)
    co2.parent.mkdir(parents=True)
    forcing.write_bytes(b"fixture forcing file")
    co2.write_bytes(b"fixture annual co2 file")
    monkeypatch.setattr(admission, "FORCING_SHA256", hashlib.sha256(forcing.read_bytes()).hexdigest())
    monkeypatch.setattr(admission, "CO2_SHA256", hashlib.sha256(co2.read_bytes()).hexdigest())
    return root, forcing.resolve(), co2.resolve()


def _assets(tmp_path, monkeypatch):
    config = tmp_path / "orchidee.yaml"
    config.write_text("fixture config", encoding="utf-8")
    monkeypatch.setattr(admission, "CONFIG_SHA256", admission.sha256(config))
    domain = SimpleNamespace(
        forcing_indices_zero_based=np.asarray([[145, 46]], dtype=np.int64),
        lon=np.asarray([111.0]), lat=np.asarray([-3.0]), nbindex=1,
        area=np.asarray([1.0]), contfrac_land=np.asarray([1.0]),
    )
    context = SimpleNamespace(
        config_path=config,
        first_step_bundle=SimpleNamespace(year=1961, domain=domain),
        dt_sechiba=1800.0,
    )
    owner = SimpleNamespace(runtime_year=1991, lifecycle_start_tstep=48,
        result_day_index=2, context=context)
    adapter = SimpleNamespace(static_owner=owner)
    selection = dict(artifact_id=299, window_index=0, horizon=7,
        spatial_split="train", temporal_split="train", landpoint_id="345.0-069.0",
        year=1967, day_indices=list(range(203, 210)), model_date_labels=["1967-07-22"],
        case_sha256="c" * 64)
    return dict(context={"template": adapter}, manifest_metadata={
        "saved_B1_mathematical_identity": {"parameter_count": 66123713},
        "sampling": selection,
        "case_identity": {"case_sha256": admission.CASE_SHA256,
            "initial_state_sha256": admission.STATE_SHA256},
    }), context


def test_data_root_requires_both_exact_hashed_regular_inputs(tmp_path, monkeypatch):
    root, forcing, co2 = _verified_inputs(tmp_path, monkeypatch)
    resolved, files = admission.validate_data_root(root)
    assert resolved == root.resolve()
    assert Path(files["forcing_1991"]["path"]) == forcing
    assert Path(files["co2_annual"]["path"]) == co2

    forcing.write_bytes(b"changed")
    with pytest.raises(ValueError, match="1991 forcing SHA-256 differs"):
        admission.validate_data_root(root)

    forcing.unlink()
    with pytest.raises(FileNotFoundError, match="1991 forcing is missing:.*cruncep_twodeg_1991.nc"):
        admission.validate_data_root(root)


def test_consumption_runs_dynamic_step_bundle_then_static_payload_and_restores_data_root(
        tmp_path, monkeypatch):
    root, forcing_path, co2_path = _verified_inputs(tmp_path, monkeypatch)
    assets, context = _assets(tmp_path, monkeypatch)
    monkeypatch.setenv("ORCHIDEE_DATA_ROOT", "old-root")
    from jax_orchidee.driver import domain, orchestration as teacher

    monkeypatch.setattr(domain, "load_case_config", lambda path: {"drivers": {
        "atmospheric_forcing": {"file_pattern": str(forcing_path)},
        "co2": {"file": str(co2_path)},
    }})
    monkeypatch.setattr(domain, "forcing_file_for_year", lambda config, year: forcing_path)
    forcing_values = {name: np.asarray([[index + 1.0]]) for index, name in enumerate(
        ("Tair", "PSurf", "Qair", "Wind_E", "Wind_N", "Rainf", "Snowf", "SWdown", "LWdown", "Areas", "contfrac"))}
    forcing_values.update(Height_Lev1=2.0, Height_Levuv=10.0)
    forcing = SimpleNamespace(**forcing_values)
    bundle = SimpleNamespace(year=1991, tstep=48, co2_ppm=350.0, ccanopy=np.asarray([350.0]),
        domain=context.first_step_bundle.domain, forcing=forcing)
    calls = []

    def read_bundle(received_context, *, year, tstep):
        calls.append(("bundle", received_context, year, tstep, __import__("os").environ["ORCHIDEE_DATA_ROOT"]))
        return bundle

    def build_payload(received_bundle, *, driver_z0_for_wind, dt_sechiba):
        calls.append(("payload", received_bundle, driver_z0_for_wind, dt_sechiba))
        values = {}
        optional = {"resolution", "neighbours", "corners", "seglength", "veget", "njsc",
            "clay_frac", "sand_frac", "silt_frac", "bulk_dens", "soil_ph", "poor_soils",
            "soilclass_sub_index", "soilclass_sub_area", "soilclass", "salinity", "tide_height"}
        for field in fields(IntersurfFirstStepPayload):
            name = field.name
            if name == "missing_fields":
                values[name] = ("soilclass",)
            elif name in optional:
                values[name] = None
            elif name in {"kjpindex", "nbindex"}:
                values[name] = 1
            elif name == "lalo":
                values[name] = np.asarray([[111.0, -3.0]])
            elif name in {"veget_max", "soiltile"}:
                values[name] = np.ones((1, 2))
            else:
                values[name] = np.ones((1,))
        return SimpleNamespace(**values)

    monkeypatch.setattr(teacher, "_paper_1961_step_bundle_from_context", read_bundle)
    monkeypatch.setattr(teacher, "build_intersurf_first_step_payload", build_payload)
    report = admission.consume_retained_inputs(assets, root)

    assert [row[0] for row in calls] == ["bundle", "payload"]
    assert calls[0][1:4] == (context, 1991, 48)
    assert calls[0][4] == str(root.resolve())
    assert calls[1][1:] == (bundle, 0.1, 1800.0)
    assert report["sampling_identity"]["year"] == 1967
    assert report["sampling_identity"]["case_sha256"] == admission.CASE_SHA256
    assert report["sampling_identity"]["initial_state_sha256"] == admission.STATE_SHA256
    assert report["retained_template_identity"]["runtime_year"] == 1991
    assert report["retained_template_identity"]["serialized_first_step_bundle_year"] == 1961
    assert report["resolved_annual_co2_ppm"] == 350.0
    assert set(report["static_intersurf_payload_fields"]) == {
        field.name for field in fields(IntersurfFirstStepPayload) if field.name != "missing_fields"
    }
    assert report["static_intersurf_payload_missing_fields"] == ["soilclass"]
    assert report["forcing_heights_m"] == {"Height_Lev1": 2.0, "Height_Levuv": 10.0}
    assert __import__("os").environ["ORCHIDEE_DATA_ROOT"] == "old-root"


def test_consumption_rejects_wrong_retained_landpoint_before_driver_calls(tmp_path, monkeypatch):
    root, _, _ = _verified_inputs(tmp_path, monkeypatch)
    assets, _ = _assets(tmp_path, monkeypatch)
    assets["context"]["template"].static_owner.context.first_step_bundle.domain.forcing_indices_zero_based = np.asarray([[1, 2]])
    with pytest.raises(ValueError, match="forcing landpoint differs"):
        admission.consume_retained_inputs(assets, root)


def test_supervisor_plan_preserves_explicit_retained_data_root(tmp_path):
    from types import SimpleNamespace
    from scripts.hpc import run_shared_current_T7_precision_admission as supervisor

    retained_root = tmp_path / "recovery-inputs"
    args = SimpleNamespace(output=tmp_path / "result.json", runtime_preflight=None)
    argv = ["--stage", "execute", "--retained-data-root", str(retained_root)]
    plan = supervisor.build_plan(args, argv)
    assert plan["commands"]["probe"][-len(argv):] == argv


def test_execute_validates_retained_data_root_before_gpu_bootstrap(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from scripts.hpc import probe_shared_current_T7_precision_admission as probe

    argv = ["entry", "--stage", "execute"]
    for name in ("parameters", "prepared-context", "training-prepared-manifest", "dataset-manifest",
                 "saved-identity-receipt", "state-input-metadata", "turnover-overlay", "output"):
        argv += ["--" + name, str(tmp_path / name)]
    argv += ["--retained-data-root", str(tmp_path / "missing-root")]
    monkeypatch.setattr(probe.sys, "argv", argv)
    monkeypatch.setattr(probe, "check_approval", lambda args: {"approved": True})
    monkeypatch.setenv("ORCJAX_CURRENT_T7_SUPERVISED", "1")
    monkeypatch.setitem(sys.modules, "scripts.hpc.orcjax_gpu_runtime",
                        SimpleNamespace(initialize_gpu_backend=lambda: pytest.fail("GPU bootstrap reached")))
    with pytest.raises(FileNotFoundError, match="ORCHIDEE_DATA_ROOT is missing"):
        probe.main()
