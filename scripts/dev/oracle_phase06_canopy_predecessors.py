"""Finite PB01 restart/held/source-order predecessor gates; no model step.

Original source fragments execute only the declared complete-present restart
and held-state contracts. Fortran IOIPSL/file interpolation is not simulated
as numerical truth. Actual maintained reader/init/transition/DIFFUCO adapters
run separately against these declared independent component tuples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dev import oracle_phase05_canopy_conditions as phase05  # noqa: E402
from scripts.dev import oracle_physics_branch_canopy as prior  # noqa: E402
from scripts.dev.oracle_canopy_component_fixture import make_canopy_fixture  # noqa: E402

OUTPUT = (
    ROOT
    / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/canopy-predecessors"
)
FIXTURE = ROOT / "tests/fixtures/phase06_canopy_predecessors_reference.json"
SLOW = ROOT / "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90"
RTOL, ATOL = phase05.RTOL, phase05.ATOL
CASES = (
    "lai_zero_independent_fc",
    "lai_exact_threshold_independent_fc",
    "lai_just_above",
    "lai_twice_threshold",
    "lai_1e7",
    "lai_1e6",
    "lai_1e4",
    "lai_point1",
)
SURFACE_FIELDS = ("lai", "height", "veget", "veget_max", "frac_nobio", "totfrac_nobio", "soiltile", "tot_bare_soil")


def make_inputs(*, maintenance=False):
    x = phase05.make_low_lai_inputs()
    x["veget_max"][:] = 0.0
    x["veget_max"][:, 13] = 0.9
    x["veget"][:] = 0.0
    # Independent post-restart coverage, intentionally different from the
    # daily exp(-LAI) relation. These finite values are not observed history.
    fc = np.array([0.3, 0.4, 1e-6, 2e-6, 2e-5, 2e-4, 0.02, 0.2])
    x["veget"][:, 13] = 0.9 * fc
    x["tot_bare_soil"] = 0.9 - x["veget"][:, 13]
    if maintenance:
        x["lai"][:, 13] = 1.4
    return x


def maintenance_inputs():
    """Independent component biomass/SLA constants, not inverted endpoints."""
    biomass = np.zeros((8, 14, 12, 1))
    biomass[:, 13, 0, 0] = [0.0, 5e-7, 5.05e-7, 1e-6, 5e-6, 5e-5, 0.005, 5.0]
    return biomass, np.full((8, 14), 0.02)


def restart_inputs(x=None):
    x = make_inputs() if x is None else x
    n = 8
    restart = {k: np.asarray(x[k]).copy() for k in ("lai", "height", "veget", "veget_max", "frac_nobio")}
    restart.update(
        frac_age=np.full((n, 14, 4), 0.25),
        njsc=np.full(n, 2, dtype=np.int32),
        clay_frac=np.full(n, 0.2),
        sand_frac=np.full(n, 0.4),
        bulk_dens=np.full(n, 1650.0),
        soil_ph=np.full(n, 7.0),
        poor_soils=np.zeros(n),
        reinf_slope=np.zeros(n),
        growth_day=np.full(n, 17.0),
    )
    default = np.zeros((n, 14))
    default[:, 13] = 1.0
    return dict(
        restart=restart,
        veget_max_default=default,
        frac_nobio_default=0.0,
        height_presc=np.full(14, 4.0),
        pref_soil_veg=phase05.PREF_SOIL_VEG,
        ext_coeff_vegetfrac=np.ones(14),
        nstm=6,
        diaglev=np.array([0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0]),
        active_pft_index=13,
        salinity_data=np.full(n, 31.25),
        tide_height_data=np.tile([-0.05, 0.0, 0.1], (n, 1)),
    )


def read_declared_restart(restart):
    """Real restart reader, temporary declared NetCDF; not a natural restart."""
    import xarray as xr

    from jax_orchidee.sechiba.slowproc import read_slowproc_restart_entry_state

    variables = {k: (("point", "pft"), restart[k]) for k in ("lai", "height", "veget", "veget_max")}
    variables["frac_nobio"] = (("point", "nobio"), restart["frac_nobio"])
    variables["frac_age"] = (("point", "leafage", "pft"), np.transpose(restart["frac_age"], (0, 2, 1)))
    with tempfile.TemporaryDirectory(prefix="orchidee_phase06_restart_") as directory:
        path = Path(directory) / "declared_restart.nc"
        xr.Dataset(variables).to_netcdf(path)
        return read_slowproc_restart_entry_state(path)


def run_predecessors(*, use_jit=False, high_fc=False, maintenance=False):
    import jax

    from jax_orchidee.sechiba.science_completion import slowproc_initialize_source_routed
    from jax_orchidee.sechiba.slowproc import slowproc_surface_transition_explicit

    x = make_inputs(maintenance=maintenance)
    if high_fc:
        x["lai"][:, 13] = 1.01e-8
        x["veget"][:, 13] = 0.6
    init_kwargs = restart_inputs(x)
    read = read_declared_restart(init_kwargs["restart"])
    # The actual reader's normalized axes reach the actual maintained init.
    init_kwargs["restart"].update({k: getattr(read, k) for k in read._fields})
    initialized, qsintmax, _ = slowproc_initialize_source_routed(
        init_kwargs=init_kwargs, dt_stomate=86400.0, dt_sechiba=1800.0, ok_stomate=True, qsintcst=0.1
    )

    def held_numeric(lai, frac_nobio, veget_max, veget, totfrac_nobio, soiltile):
        value = slowproc_surface_transition_explicit(
            do_slow=False,
            lai=lai,
            frac_nobio=frac_nobio,
            veget_max=veget_max,
            veget=veget,
            totfrac_nobio=totfrac_nobio,
            soiltile=soiltile,
        ).surface
        return {
            **{
                k: getattr(value.vegetation, k)
                for k in ("frac_nobio", "veget_max", "veget", "totfrac_nobio", "soiltile")
            },
            "tot_bare_soil": value.tot_bare_soil,
        }

    fn = jax.jit(held_numeric) if use_jit else held_numeric
    held = fn(
        **{k: getattr(initialized, k) for k in ("lai", "frac_nobio", "veget_max", "veget", "totfrac_nobio", "soiltile")}
    )
    held.update(lai=initialized.lai, height=initialized.height)
    return x, initialized, qsintmax, held


def run_surface(*, use_jit=False, maintenance=False):
    from jax_orchidee.sechiba.condveg import run_condveg_first_step_module

    x, initialized, qsintmax, held = run_predecessors(use_jit=use_jit, maintenance=maintenance)
    x.update({k: np.asarray(held[k]) for k in SURFACE_FIELDS if k in x})
    values = {"ALB_BG_MODIS": "y", "ALB_BARE_MODEL": "n", "ALB_ICE__00001": ".6", "ALB_ICE__00002": ".2"}
    for name in ("alb_leaf_vis", "alb_leaf_nir", "snowa_aged_vis", "snowa_aged_nir", "snowa_dec_vis", "snowa_dec_nir"):
        values.update({f"{name.upper()}__{i:05d}": str(v) for i, v in enumerate(x[name], 1)})
    closure = run_condveg_first_step_module(
        restart_payload={**x, "soilalbedo_bg": x["soilalb_bg"]},
        driver_payload=x,
        slowproc_payload=x,
        hydrol_diagnostics=NS(drysoil_frac=x["drysoil_frac"]),
        run_def_values=values,
        use_jit=use_jit,
    )
    assert closure.ok and not closure.missing_inputs
    if maintenance:
        import jax

        from jax_orchidee.driver import orchestration as driver
        from jax_orchidee.sechiba.slowproc import slowproc_surface_transition_explicit

        def maintenance_then_held(state, biomass, sla_calc):
            fields = {"slowproc_stomate_previous_step_state": {}, "diffuco_previous_step_state": {}}
            prior_state = {**state, "biomass": biomass, "sla_calc": sla_calc}
            driver._write_maintenance_lai_to_previous_fields(fields, prior_state, (False,) * 14)
            lai = fields["diffuco_previous_step_state"]["lai"]
            value = slowproc_surface_transition_explicit(
                do_slow=False,
                lai=lai,
                **{k: state[k] for k in ("frac_nobio", "veget_max", "veget", "totfrac_nobio", "soiltile")},
            ).surface
            return {
                **{
                    k: getattr(value.vegetation, k)
                    for k in ("frac_nobio", "veget_max", "veget", "totfrac_nobio", "soiltile")
                },
                "tot_bare_soil": value.tot_bare_soil,
                "lai": lai,
                "height": state["height"],
            }

        function = jax.jit(maintenance_then_held) if use_jit else maintenance_then_held
        held = function(held, *maintenance_inputs())
        x.update({k: np.asarray(held[k]) for k in SURFACE_FIELDS if k in x})
    return x, initialized, qsintmax, held, closure


def base_fixture():
    base = make_canopy_fixture(indices=(0,))
    payload = {k: np.repeat(v, 8, axis=0) for k, v in base.diffuco_precall.payload.items()}
    for key in ("frac_nobio", "snow_nobio", "frac_snow_nobio"):
        payload[key] = payload[key][:, :1]
    static = {**base.static_kwargs, "ok_laidev": np.zeros(14, dtype=bool)}
    return base._replace(
        case_names=CASES,
        diffuco_precall=base.diffuco_precall._replace(payload=payload),
        static_kwargs=static,
        driver={k: np.repeat(v, 8, axis=0) for k, v in base.driver.items()},
        albedo={k: np.repeat(v, 8, axis=0) for k, v in base.albedo.items()},
        thermal={k: np.repeat(v, 8, axis=0) for k, v in base.thermal.items()},
    )


def next_payload(x, surface, qsintmax):
    """Declared DIFFUCO tuple: actual producer fields plus explicit backgrounds."""
    base = base_fixture()
    payload = dict(base.diffuco_precall.payload)
    payload.update(
        {
            k: np.asarray(x[k])
            for k in (
                "veget",
                "veget_max",
                "lai",
                "height",
                "frac_nobio",
                "totfrac_nobio",
                "tot_bare_soil",
                "snow",
                "snow_nobio",
                "snowrho",
                "snowdz",
                "zlev",
                "u",
                "v",
                "temp_air",
                "pb",
                "drysoil_frac",
            )
        }
    )
    payload.update(
        {
            k: np.asarray(surface[k])
            for k in ("z0m", "z0h", "roughheight", "roughheight_pft", "frac_snow_veg", "frac_snow_nobio")
        }
    )
    payload.update(
        qsintmax=np.asarray(qsintmax),
        snow_age=x["snow_age"],
        snow_nobio_age=x["snow_nobio_age"],
        soilalbedo_bg=x["soilalb_bg"],
    )
    # Original never defines the PFT1 roughness diagnostic, and DIFFUCO's
    # PFT roughness loop starts at2. Supply an explicit unused background.
    payload["roughheight_pft"] = np.asarray(payload["roughheight_pft"]).copy()
    payload["roughheight_pft"][:, 0] = 0.0
    return base._replace(diffuco_precall=base.diffuco_precall._replace(payload=payload))


def run_next(*, use_jit=False, maintenance=False):
    from jax_orchidee.driver import orchestration as driver
    from jax_orchidee.sechiba.diffuco import run_pft14_local_enerbil_precall_from_first_step_precall

    x, initialized, qsintmax, held, closure = run_surface(use_jit=use_jit, maintenance=maintenance)
    surface = {k: np.asarray(getattr(closure.module, k)) for k in closure.module._fields}
    fixture = next_payload(x, surface, qsintmax)
    # Actual maintained mapper receives the component return and carried state;
    # Phase05 already owns comprehensive writer checks, so this is the new low-
    # LAI operand subset through that same writer/mapper, not a suite rerun.
    fields = driver._empty_previous_step_state_fields()
    driver._add_condveg_to_previous_fields(fields, closure)
    fields["slowproc_stomate_previous_step_state"].update({k: held[k] for k in SURFACE_FIELDS})
    operands, _, missing = driver._diffuco_restart_payload_from_state_fields(
        diffuco_state=fields["diffuco_previous_step_state"],
        hydrol_state={},
        slowproc_state=fields["slowproc_stomate_previous_step_state"],
    )
    # Unrelated declared background inputs complete the local process tuple.
    payload = {**fixture.diffuco_precall.payload, **operands}
    payload["roughheight_pft"] = np.asarray(payload["roughheight_pft"]).copy()
    payload["roughheight_pft"][:, 0] = 0.0
    fixture = fixture._replace(diffuco_precall=fixture.diffuco_precall._replace(payload=payload))
    result = run_pft14_local_enerbil_precall_from_first_step_precall(
        fixture.diffuco_precall, run_def_values={}, static_kwargs=fixture.static_kwargs, use_jit=use_jit
    )
    assert result.result is not None
    return fixture, result, missing


def actual_values(*, use_jit=False):
    groups = {}
    for maintenance in (False, True):
        _, initialized, qsintmax, held, closure = run_surface(use_jit=use_jit, maintenance=maintenance)
        fields = {"restart." + k: np.asarray(getattr(initialized, k)).ravel() for k in SURFACE_FIELDS}
        fields.update({"held." + k: np.asarray(held[k]).ravel() for k in SURFACE_FIELDS})
        fields["restart.qsintmax"] = np.asarray(qsintmax).ravel()
        for k in closure.module._fields:
            value = np.asarray(getattr(closure.module, k))
            if k == "roughheight_pft":
                value = value[:, 1:]
            fields["condveg." + k] = value.ravel()
        _, result, _ = run_next(use_jit=use_jit, maintenance=maintenance)
        diffuco = prior.diffuco_values(result)
        diffuco["diffuco.q_cdrag_pft"] = diffuco["diffuco.q_cdrag_pft"].reshape(8, 14)[:, 1:].ravel()
        prefix = "maintenance_" if maintenance else ""
        groups[prefix + "predecessor"] = fields
        groups[prefix + "next_diffuco"] = diffuco
    return groups


def original_span(start, end):
    raw = b"".join(SLOW.read_bytes().splitlines(keepends=True)[start - 1 : end])
    return raw.decode("latin1"), dict(
        source=SLOW.relative_to(ROOT).as_posix(),
        start_line=start,
        end_line=end,
        source_sha256=prior.sha(SLOW),
        span_sha256=hashlib.sha256(raw).hexdigest(),
    )


def compose_original(*, high_fc=False, maintenance=False):
    x = make_inputs(maintenance=maintenance)
    if high_fc:
        x["lai"][:, 13] = 1.01e-8
        x["veget"][:, 13] = 0.6
    source, spans = phase05.composed(x, predecessor=True)
    first = source.index("biomass=0d0;")
    end = source.index("call condveg_main(1,n", first)
    declarations = """integer::ji,jv,jst,date=17,year_length=365,kjit=1
logical::found_restart,LastTsDay=.false.,do_slow
real(r_std),parameter::val_exp=999999d0,dt_days=1d0,qsintcst=.1d0
real(r_std)::qsintmax(n,nvm)
real(r_std)::restart_lai(n,nvm),sla_calc(n,nvm)
logical::ok_LAIdev(nvm)=.false.
"""
    source = source.replace("integer,parameter::n=8", "integer,parameter::n=8,kjpindex=n")
    source = source.replace("logical::f_rot_sech(n)", "logical::f_rot_sech(n)\n" + declarations)
    # Complete present fields enter the actual original flag assignments;
    # no IOIPSL behavior, missing-data recovery or auxiliary processes claimed.
    fragments = []
    for a, b in (
        (1581, 1581),
        (1586, 1586),
        (1592, 1592),
        (1601, 1601),
        (1922, 1926),
        (2250, 2279),
        (654, 663),
        (1078, 1078),
        (1103, 1104),
        (1114, 1114),
        (1116, 1122),
    ):
        text, identity = original_span(a, b)
        fragments.append((text, identity))
    init = "biomass=0d0;litter_above=0d0;litter_below=0d0;carbon_32l=0d0;DOC=0d0;f_rot_sech=.false.\n"
    init += "\n".join(text for text, _ in fragments[:6])
    emits = []
    for k, shape in (
        ("lai", (8, 14)),
        ("height", (8, 14)),
        ("veget", (8, 14)),
        ("veget_max", (8, 14)),
        ("frac_nobio", (8, 1)),
        ("totfrac_nobio", (8,)),
        ("soiltile", (8, 6)),
        ("tot_bare_soil", (8,)),
    ):
        expression = f"{k}(i)" if len(shape) == 1 else f"{k}(i,j)"
        restart_expression = "restart_lai(i,j)" if k == "lai" else expression
        emits.append(prior.emit("restart." + k, restart_expression, shape))
        emits.append(prior.emit("held." + k, expression, shape))
    # Original initialize: STOMATE=true derivative branch at300–301.
    qsint, qsint_span = original_span(300, 301)
    init += qsint + "\nrestart_lai=lai\n" + "\n".join(text for text, _ in fragments[6:])
    first = source.index("biomass=0d0;")
    end = source.index("call condveg_main(1,n", first)
    source = source[:first] + init + "\n" + source[end:]
    # Offsets changed above; declarations are applied after replacing the body.
    source = source.replace("integer,parameter::n=8\n", "integer,parameter::n=8,kjpindex=n\n")
    if "integer::ji,jv,jst" not in source:
        source = source.replace("logical::f_rot_sech(n)", "logical::f_rot_sech(n)\n" + declarations)
    # Remove phase05's producer diagnostic tail; emit only exact new fields.
    lines = source.splitlines()
    lines = [line for line in lines if "'producer." not in line]
    source = "\n".join(lines) + "\n"
    if maintenance:
        resp = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_resp.f90"
        raw = resp.read_bytes().splitlines(keepends=True)
        # Source PFT14 assignment and bare-soil assignment, before the actual
        # held gate. Respiration magnitudes are outside this LAI write contract.
        resp_spans = []
        blocks = []
        for a, b in ((317, 317), (323, 327)):
            block = b"".join(raw[a - 1 : b])
            blocks.append(block.decode("latin1"))
            resp_spans.append(
                dict(
                    source=resp.relative_to(ROOT).as_posix(),
                    start_line=a,
                    end_line=b,
                    source_sha256=prior.sha(resp),
                    span_sha256=hashlib.sha256(block).hexdigest(),
                )
            )
        biomass, sla = maintenance_inputs()
        injection = (
            "j=14\n"
            + prior.old.assignment("biomass", biomass)
            + "\n"
            + prior.old.assignment("sla_calc", sla)
            + "\n"
            + "\n".join(blocks)
            + "\n"
            + "\n".join(text for text, _ in fragments[6:])
        )
        source = source.replace(
            "integer::ji,jv,jst,date", "integer,parameter::ibare_sechiba=1,ileaf=1,icarbon=1\ninteger::ji,jv,jst,date"
        )
        source = source.replace("albedo_snow_mean=pb01_snow_mean", "albedo_snow_mean=pb01_snow_mean\n" + injection)
        spans["maintenance_lai_fragments"] = resp_spans
    emits.append(prior.emit("restart.qsintmax", "qsintmax(i,j)", (8, 14)))
    source = source.replace("close(10)\nend program", "\n".join(emits) + "\nclose(10)\nend program")
    if high_fc:
        observation = "write(*,'(a,2(l1,1x),4(es25.17,1x))') 'PRECONDVEG ',found_restart,do_slow,lai(1,14),veget(1,14),veget_max(1,14),tot_bare_soil(1)\nflush(6)\n"
        source = source.replace("call condveg_main(1,n", observation + "call condveg_main(1,n", 1)
    spans["phase06_fragments"] = [identity for _, identity in fragments] + [qsint_span]
    return source, spans


def _reference_case(maintenance):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prefix = "maintenance_" if maintenance else ""
    source, spans = compose_original(maintenance=maintenance)
    target = OUTPUT / ("reference_" + prefix + "predecessor")
    phase05.build(source, target)
    prior.save(target / "source_spans.json", spans)
    raw = prior.old.read_csv(target / "fortran_outputs.csv")
    surface = {
        k: raw[k].reshape((8, 14) if k == "roughheight_pft" else (8, 1) if k == "frac_snow_nobio" else (8,))
        for k in ("roughheight_pft", "frac_snow_nobio", "frac_snow_veg", "z0m", "z0h", "roughheight")
    }
    x = make_inputs(maintenance=maintenance)
    for k in SURFACE_FIELDS:
        if k in x:
            x[k] = raw["held." + k].reshape(x[k].shape)
    fixture = next_payload(x, surface, raw["restart.qsintmax"].reshape(8, 14))
    previous = prior.old.make_canopy_fixture
    try:
        prior.old.make_canopy_fixture = lambda: fixture
        with tempfile.TemporaryDirectory(prefix="orchidee_phase06_compose_") as directory:
            path = Path(directory) / "oracle.f90"
            diff_spans = prior.old.compose(path)
            diff_source = path.read_text(encoding="utf-8").split("allocate(psold(n)")[0] + "close(10)\nend program\n"
            diff_source = diff_source.replace("integer,parameter::nnobio=2", "integer,parameter::nnobio=1")
            # The shared phase01 composer hardcodes its diagnostic LAIdev=true
            # case; bind the declared paper false switch instead. Initial
            # mismatched-dispatch outputs are retained in attempts/.
            assert diff_source.count("ok_LAIdev(14)=.true.;") == 1
            diff_source = diff_source.replace("ok_LAIdev(14)=.true.;", "ok_LAIdev(14)=.false.;")
    finally:
        prior.old.make_canopy_fixture = previous
    diff_target = OUTPUT / ("reference_" + prefix + "next_diffuco")
    phase05.build(diff_source, diff_target)
    prior.save(diff_target / "source_spans.json", diff_spans)
    diff = prior.old.read_csv(diff_target / "fortran_outputs.csv")
    diff = {k: v for k, v in diff.items() if k not in ("diffuco.qsintmax", "diffuco.air_qsat", "diffuco.air_dqsat")}
    diff["diffuco.cim"] = diff["diffuco.cim"].reshape(8, 14)[:, 13]
    diff["diffuco.q_cdrag_pft"] = diff["diffuco.q_cdrag_pft"].reshape(8, 14)[:, 1:].ravel()
    fields = {k: v for k, v in raw.items() if k.startswith(("restart.", "held."))}
    for k, v in raw.items():
        if k.startswith(("restart.", "held.")):
            continue
        fields["condveg." + k] = v.reshape(8, 14)[:, 1:].ravel() if k == "roughheight_pft" else v
    inputs = dict(
        surface=make_inputs(maintenance=maintenance),
        restart=restart_inputs(make_inputs(maintenance=maintenance)),
        next_diffuco=fixture.diffuco_precall.payload,
        diffuco_static=fixture.static_kwargs,
    )
    if maintenance:
        inputs["maintenance_biomass"], inputs["maintenance_sla_calc"] = maintenance_inputs()
    return fields, diff, spans, diff_spans, inputs


def reference():
    expected = {}
    identities = {}
    inputs = {}
    for maintenance in (False, True):
        fields, diff, spans, diff_spans, case_inputs = _reference_case(maintenance)
        prefix = "maintenance_" if maintenance else ""
        expected.update({prefix + "predecessor": fields, prefix + "next_diffuco": diff})
        identities.update({prefix + "predecessor": spans, prefix + "next_diffuco": diff_spans})
        inputs[prefix.rstrip("_") or "restart_held"] = case_inputs
    prior.save(OUTPUT / "input_tuples.json", inputs)
    prior.save(
        FIXTURE,
        dict(
            schema_version=1,
            cases=CASES,
            rtol=RTOL,
            atol=ATOL,
            inputs=inputs,
            input_tuples_sha256=prior.sha(OUTPUT / "input_tuples.json"),
            expected=expected,
            reference_identity=identities,
        ),
    )
    trapped, trapspans = compose_original(high_fc=True)
    trap = OUTPUT / "diagnostic_high_fc_restart"
    returncode = phase05.build(trapped, trap, may_fail=True)
    assert returncode != 0
    prior.save(trap / "source_spans.json", trapspans)
    run = json.loads((trap / "run.json").read_text(encoding="utf-8"))
    prior.save(
        trap / "observation.json",
        dict(
            returncode=returncode,
            precondveg=[line for line in run["stdout"].splitlines() if line.startswith("PRECONDVEG")],
            meaning="original complete-present restart and held false reach the original CONDVEG trap; write/flush instrumentation only",
        ),
    )


def certify():
    import jax

    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))["expected"]
    report = {}
    for mode in ("eager", "jit"):
        actual = actual_values(use_jit=mode == "jit")
        comparisons = [
            phase05.float_comparison(g + "." + k, np.asarray(v), actual[g][k], rtol=RTOL, atol=ATOL)
            for g, fields in expected.items()
            for k, v in fields.items()
        ]
        np.savez(
            OUTPUT / f"actual_{mode}.npz", **{g + "." + k: v for g, fields in actual.items() for k, v in fields.items()}
        )
        report[mode] = dict(passed=all(item["passed"] for item in comparisons), comparisons=comparisons)
    prior.save(OUTPUT / "comparison.json", report)


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("reference", "certify"))
    args = parser.parse_args()
    globals()[args.action]()
