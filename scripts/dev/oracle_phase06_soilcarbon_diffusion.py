"""Finite synthetic original-Fortran diffusion and nonzero cryoturbation joins.

The uninstrumented original procedures remain byte exact. A separate observer
clone records predicate masks and diffusion boundary arrays without modifying
process values. No model timestep/day/year or server computation is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from fortran_oracle_common import compiler_environment  # noqa: E402
from oracle_lane_soilcarbon_owner_probe import COMPILER, SOURCE, SPANS, span_bytes  # noqa: E402
from oracle_phase05_litter_soilcarbon import (  # noqa: E402
    ATOL,
    FLOAT_FIELDS,
    FLOAT_SAVE_2,
    FLOAT_SAVE_3,
    INT_FIELDS,
    INT_SAVE_2,
    INT_SAVE_3,
    LITTER_FIELDS,
    LITTER_INPUTS,
    LOGICAL_FIELDS,
    RTOL,
    SOIL_INPUTS,
    _compare,
    _read_fields,
    _wrapper,
    _write_inputs,
    case_inputs,
)

OUT = ROOT / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/soilcarbon-diffusion"
FIXTURE = ROOT / "tests/fixtures/phase06_soilcarbon_diffusion_reference.npz"
META = FIXTURE.with_suffix(".json")
SOIL_FIELDS = FLOAT_FIELDS + INT_FIELDS + FLOAT_SAVE_2 + INT_SAVE_2 + FLOAT_SAVE_3 + INT_SAVE_3 + LOGICAL_FIELDS
GEOMETRY_INPUTS = SOIL_INPUTS + [("zz", "@zz"), ("zzc", "@zzc")]
ARMS = ["peak_limiter", "lower_out_limiter", "upper_out_limiter", "up_limited", "up_normal", "down_limited", "down_normal", "update_else"]
SPECS = [dict(id="thin_nonuniform_limiters", route="diffusion"), dict(id="thin_nonuniform_monotone", route="diffusion"), dict(id="nonuniform_frozen", route="diffusion"), dict(id="nonzero_cryo_carry", route="cryoturbation")]


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inputs(spec):
    """Explicit legal nonnegative states and positive ordered layer geometry."""
    x, raw = case_inputs(dict(id=spec["id"], temperature=[280.0, 280.0], moisture=[0.4, 0.4], zero=True, restart_altmax=1.0))
    for field in ["fbact_litter", "fbact_doc", "fbact_soilcarbon", "wat_flux", "runoff_per_soil", "drainage_per_soil", "runoff2peat"]:
        x[field] = np.zeros_like(x[field])
    if spec["route"] == "diffusion":
        thickness = np.r_[np.array([0.0002, 0.00023, 0.00017, 0.001, 0.002]), np.full(27, (2.0 - 0.0036) / 27)]
    else:
        thickness = np.geomspace(0.005, 0.2, 32)
        thickness *= 2.0 / thickness.sum()
    bounds = np.r_[0.0, np.cumsum(thickness)]
    centers = 0.5 * (bounds[:-1] + bounds[1:])
    x["zf_soil_b"] = bounds
    raw.update(zz=centers, zzc=bounds[1:])
    profiles = np.broadcast_to(np.linspace(1.3, 2.8, 32), (7, 32)).copy()
    if spec["id"] == "thin_nonuniform_limiters":
        profiles[:, :3] = np.array([[1, 3, 1], [3.1, 3, 0.1], [0.1, 3, 3.1], [0.1, 1, 10], [10, 1, 0.1], [1, 1.01, 1.02], [1.02, 1.01, 1]])
    elif spec["id"] == "thin_nonuniform_monotone":
        profiles[:, :3] = np.array([[0.1, 1, 10], [10, 1, 0.1], [1, 1.01, 1.02], [1.02, 1.01, 1], [1, 2, 3], [3, 2, 1], [1.1, 1.15, 1.2]])
    elif spec["route"] == "cryoturbation":
        profiles = np.arange(1, 8)[:, None] * (1.0 + 2.0 * np.exp(-centers[None, :] / 0.2))
        x["carbon_32l"][:, :, 13, :] = np.array([3.0, 2.0, 1.0])[None, :, None] * (1 + np.exp(-centers / 0.3))[None, None, :] * thickness
        x["litter_below"][:, :, 13, :, 0] = np.array([2.0, 4.0])[None, :, None] * (1 + np.exp(-centers / 0.1))[None, None, :] * thickness
    x["doc"][:, 13, :, 0, :, 0] = (profiles * thickness).T
    x["doc"][:, 13, :, 1, :, 0] = (profiles * thickness * 0.3).T
    if spec["id"] == "nonuniform_frozen":
        x["tprof"][:, 1, :] = 272.15
        x["tprof"][1, 2, :] = 272.1500001
    if spec["route"] == "cryoturbation":
        x.update(ok_cryoturb=True, altmax_ind=np.full((2, 14), 31, np.int32), altmax_lastyear=raw["altmax"], fixed_cryoturbation_depth=raw["fixed"], veget_mask=np.ones((2, 14), bool), zi_soil=centers)
    return x, raw


def _conditions(block):
    """Read original WHERE predicates, preserving the source expression text."""
    found = re.findall(r"(?:WHERE|ELSEWHERE)\s*\((.*?)\)\s*\n", block, re.I | re.S)
    return [" ".join(c.replace("&", " ").split()) for c in found]


def soil_wrapper(*, observe=False, cryo=False):
    source, hashes = _wrapper("soil")
    text = source.decode("latin1")
    marker = "program run_soilcarbon_owner_oracle"
    head, program = text.split(marker, 1)
    program = marker + program
    old_read = "read(u) " + ",".join(a for a, _ in SOIL_INPUTS)
    program = program.replace(old_read, "read(u) " + ",".join(a for a, _ in GEOMETRY_INPUTS))
    if cryo:
        head = head.replace("if (trim(name)=='cryoturbate_doc_POC') value=.false.", "if (trim(name)=='cryoturbate_doc_POC') value=.true.")
        program = program.replace("  call execute_case(u)\n  close(u)", "  call execute_case(u)\n  itau=2\n  call execute_case(u)\n  itau=3\n  call execute_case(u)\n  close(u)")
        program = program.replace("    write(unit) alt,", "    write(unit) diff_k,xc_cryoturb,xd_cryoturb,alpha_C,beta_C,alpha_DOC,beta_DOC,alpha_litter_below,beta_litter_below\n    write(unit) alt,")
    if observe:
        declarations = "\n  real(r_std) :: oracle_entry(2,14,32,2,7,1)=0, oracle_after(2,14,32,2,7,1)=0, oracle_flux(2,14,32,7,1)=0\n  integer :: oracle_arms(2,14,32,7,1,8)=0\n"
        head = head.replace("  interface getin_p", declarations + "  interface getin_p", 1)
        before = "     DOC_old2(:,m,:,:,:,:)=DOC(:,m,:,:,:,:)"
        head = head.replace(before, before + "\n     oracle_entry(:,m,:,:,:,:)=DOC_old2(:,m,:,:,:,:)")
        start = head.index("           DO l =1, ndeep-2")
        end = head.index("           ENDDO", start)
        block = head[start:end]
        masks = _conditions(block)
        if len(masks) != 3:
            raise AssertionError(masks)
        observer = "\n".join(f"              oracle_arms(:,m,l,kkk,kk,{i+1})=merge(1,0,({condition}))" for i, condition in enumerate(masks))
        head = head[:start] + head[start:end].replace("           DO l =1, ndeep-2", "           DO l =1, ndeep-2\n" + observer) + head[end:]
        start = head.index("           DO l= 1, ndeep-1 ! Loop over soil layers", start)
        end = head.index("           ENDDO", start)
        block = head[start:end]
        masks = _conditions(block)
        if len(masks) != 4:
            raise AssertionError(masks)
        observer = "\n".join(f"              oracle_arms(:,m,l,kkk,kk,{i+4})=merge(1,0,({condition}))" for i, condition in enumerate(masks))
        observer += "\n              oracle_arms(:,m,l,kkk,kk,8)=1-sum(oracle_arms(:,m,l,kkk,kk,4:7),dim=2)"
        head = head[:start] + block.replace("           DO l= 1, ndeep-1 ! Loop over soil layers", "           DO l= 1, ndeep-1 ! Loop over soil layers\n" + observer) + head[end:]
        head = head.replace("        !! 2.3.9 DOC export", "     oracle_after(:,m,:,:,:,:)=DOC(:,m,:,:,:,:)\n     oracle_flux(:,m,:,:,:)=DOC_FLUX_DIFF(:,m,:,:,:)\n        !! 2.3.9 DOC export")
        program = program.replace("character(1024)::outfile,infile", "character(1024)::outfile,infile,observefile")
        program = program.replace("  close(u)\ncontains", "  close(u)\n  call get_command_argument(3,observefile)\n  open(newunit=u,file=trim(observefile),access='stream',form='unformatted',status='replace')\n  write(u) oracle_entry,oracle_after,oracle_flux,oracle_arms\n  close(u)\ncontains")
    return (head + program).encode("latin1"), hashes


COEFFICIENT_FIELDS = [("diff_k", (2, 32, 14)), ("xc_cryoturb", (2, 32, 14)), ("xd_cryoturb", (2, 32, 14)), ("alpha_c", (2, 3, 14, 32)), ("beta_c", (2, 3, 14, 32)), ("alpha_doc", (2, 14, 32, 2, 7)), ("beta_doc", (2, 14, 32, 2, 7)), ("alpha_litter_below", (2, 2, 14, 32)), ("beta_litter_below", (2, 2, 14, 32))]


def read_soil(path, cryo=False):
    fields = FLOAT_FIELDS[:-1] + tuple(COEFFICIENT_FIELDS) + FLOAT_FIELDS[-1:] + SOIL_FIELDS[len(FLOAT_FIELDS):] if cryo else SOIL_FIELDS
    if not cryo:
        return [_read_fields(path, fields)]
    size = sum(int(np.prod(shape)) * (4 if name.startswith("save_") and name not in {"save_alt", "save_altmax_lastyear", "save_z_root"} else 8) for name, shape in fields)
    data = path.read_bytes()
    if len(data) != 3 * size:
        raise AssertionError((len(data), size))
    result = []
    for i in range(3):
        part = path.with_suffix(f".call{i}.bin")
        part.write_bytes(data[i * size:(i + 1) * size])
        result.append(_read_fields(part, fields))
    return result


def read_observer(path):
    result = {}
    data = path.read_bytes()
    offset = 0
    for name, shape, dtype in [("entry", (2, 14, 32, 2, 7, 1), "<f8"), ("after", (2, 14, 32, 2, 7, 1), "<f8"), ("flux", (2, 14, 32, 7, 1), "<f8"), ("arms", (2, 14, 32, 7, 1, 8), "<i4")]:
        count = int(np.prod(shape))
        result[name] = np.frombuffer(data, dtype=dtype, offset=offset, count=count).copy().reshape(shape, order="F")
        offset += count * np.dtype(dtype).itemsize
    assert offset == len(data)
    return result


def actual_chain(x, *, cryo=False):
    """Use the maintained adapter and its real state writer, including carry."""
    from jax_orchidee.driver.orchestration import _paper_half_hour_ok_leak_state_updates
    from jax_orchidee.stomate.integration import stomate_ok_leak_explicit

    result = stomate_ok_leak_explicit(**x)
    litter, soil = result.littercalc, result.soilcarbon
    actual = {name: np.asarray(getattr(litter, name)) for name in ["litter_above", "litter_below", "lignin_struc_above", "lignin_struc_below", "litterpart", "dead_leaves", "deadleaf_cover", "resp_hetero_litter", "resp_hetero_flood", "soilcarbon_input_doc", "floodcarbon_input"]}
    actual.update({name: np.asarray(getattr(litter.fuel, name)) for name in ["fuel_1hr", "fuel_10hr", "fuel_100hr", "fuel_1000hr"]})
    actual.update({name: np.asarray(getattr(soil, name)) for name in ["carbon_32l", "doc", "doc_exp", "resp_hetero_soil", "resp_flood_soil"]})
    actual["interception_storage"] = np.asarray(result.interception_storage)
    writer = _paper_half_hour_ok_leak_state_updates(result)
    next_inputs = dict(x)
    next_inputs.update({"doc" if name == "DOC" else name: value for name, value in writer.items()})
    next_inputs["turnover"] = np.zeros_like(x["turnover"])
    next_inputs["bm_to_litter"] = np.zeros_like(x["bm_to_litter"])
    actual["soil_litter_below"] = np.asarray(result.soilcarbon.litter_below)
    actual["doc_flux_diff"] = np.asarray(result.soilcarbon.doc_flux_diff)
    actual["writer_litter_below"] = np.asarray(writer["litter_below"])
    if cryo:
        for name, _ in COEFFICIENT_FIELDS:
            actual[name] = np.asarray(getattr(result.soilcarbon.cryoturbation_coefficients, name))
        next_inputs["cryoturbation_coefficients"] = result.soilcarbon.cryoturbation_coefficients
    return actual, result, next_inputs


def _compile(source, name, temp, output):
    src = temp / f"{name}.f90"
    src.write_bytes(source)
    (output / src.name).write_bytes(source)
    exe = temp / f"{name}.exe"
    command = [str(COMPILER), "-std=legacy", "-cpp", "-fdefault-real-8", "-ffree-line-length-none", "-O0", "-fcheck=all", str(src), "-o", str(exe)]
    result = subprocess.run(command, cwd=temp, env=compiler_environment(COMPILER), capture_output=True, text=True, encoding="utf-8", errors="replace")
    (output / f"{name}.compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr)
    return exe, dict(name=name, command=command, source_sha256=hashlib.sha256(source).hexdigest(), returncode=result.returncode)


def _execute(exe, out, inp, temp, log, observer=None):
    command = [str(exe), str(out), str(inp)] + ([str(observer)] if observer else [])
    env = compiler_environment(COMPILER)
    env.update(ORACLE_NEWALTCALC="0", ORACLE_SPINUP="0")
    result = subprocess.run(command, cwd=temp, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def diagnose_initial(output=OUT):
    """Observe only the preserved initial tuple, without editing production.

    An in-memory copy of the actual core adds a callback immediately before
    diffusion. Its unmodified process return is compared with the real
    unobserved adapter before any captured internal array is relied upon.
    """
    import jax

    from jax_orchidee.stomate import integration, soilcarbon_kernels

    jax.config.update("jax_enable_x64", True)
    directory = output / "initial-equality-diagnostic"
    prior = np.load(directory / "raw-inputs-outputs.npz")
    captured, saved, rows = {}, {}, []

    def capture(doc, temperature, boundaries, coefficient):
        captured.update(entry=np.asarray(doc).copy(), temperature=np.asarray(temperature).copy(), boundaries=np.asarray(boundaries).copy(), coefficient=np.asarray(coefficient).copy())

    core_source = inspect.getsource(soilcarbon_kernels.soilcarbon_leak_core_step)
    insertion = "    jax.debug.callback(phase06_capture, water.doc, tprof, zf_soil_b, dif_doc, ordered=True)\n"
    needle = "    diffusion = soilcarbon_leak_doc_diffusion(water.doc, tprof, zf_soil_b, dif_doc)"
    assert core_source.count(needle) == 1
    observed_source = core_source.replace(needle, insertion + needle)
    core_namespace = dict(vars(soilcarbon_kernels)) | {"jax": jax, "phase06_capture": capture}
    exec(compile(observed_source, str(directory / "observed-core.py"), "exec"), core_namespace)
    observed_core = jax.jit(core_namespace["soilcarbon_leak_core_step"], static_argnames=soilcarbon_kernels.SOILCARBON_LEAK_CORE_STEP_JIT_STATIC_ARGNAMES)
    adapter_source = inspect.getsource(integration.stomate_ok_leak_explicit)
    adapter_namespace = dict(vars(integration)) | {"soilcarbon_leak_core_step_jit": observed_core}
    exec(compile(adapter_source, str(directory / "observed-adapter.py"), "exec"), adapter_namespace)
    (directory / "observed-core.py").write_text(observed_source, encoding="utf-8")
    (directory / "observed-adapter.py").write_text(adapter_source, encoding="utf-8")
    for spec in SPECS[:2]:
        ident = spec["id"]
        x, _ = inputs(spec)
        prefix = f"{ident}/input/"
        x.update({key.removeprefix(prefix): prior[key] for key in prior.files if key.startswith(prefix) and key != prefix + "zzc"})
        real = integration.stomate_ok_leak_explicit(**x)
        observed = adapter_namespace["stomate_ok_leak_explicit"](**x)
        jax.block_until_ready(observed.soilcarbon.doc)
        final_equal = np.array_equal(np.asarray(real.soilcarbon.doc), np.asarray(observed.soilcarbon.doc))
        real_doc, observed_doc = np.asarray(real.soilcarbon.doc), np.asarray(observed.soilcarbon.doc)
        original_entry = prior[f"{ident}/observer/entry"]
        original_after = prior[f"{ident}/observer/after"]
        source_final = prior[f"{ident}/call0/reference/doc"]
        old_entry, new_entry = original_entry[:, :, :, 0, :, :], captured["entry"][:, :, :, 0, :, :]
        difference = np.abs(captured["entry"] - original_entry)
        thickness = np.diff(captured["boundaries"])
        f_conc, j_conc = old_entry / thickness[None, None, :, None, None], new_entry / thickness[None, None, :, None, None]
        final_bad = np.argwhere(np.abs(real_doc - source_final) > ATOL + RTOL * np.maximum(np.abs(real_doc), np.abs(source_final)))
        branch_witness = []
        for position in final_bad:
            point, pft, layer, _fraction, pool, element = map(int, position)
            for interface in sorted(set([max(0, layer - 1), min(30, layer)])):
                source_c = f_conc[point, pft, interface:interface + 2, pool, element]
                actual_c = j_conc[point, pft, interface:interface + 2, pool, element]
                branch_witness.append(dict(output_index=position.tolist(), interface=interface, original_stock=old_entry[point, pft, interface:interface + 2, pool, element].tolist(), captured_stock=new_entry[point, pft, interface:interface + 2, pool, element].tolist(), original_concentration=source_c.tolist(), captured_concentration=actual_c.tolist(), original_concentration_hex=[float(v).hex() for v in source_c], captured_concentration_hex=[float(v).hex() for v in actual_c], original_relation=int(np.sign(source_c[0] - source_c[1])), captured_relation=int(np.sign(actual_c[0] - actual_c[1])), original_guard_mask=prior[f"{ident}/observer/arms"][point, pft, interface, pool, element].tolist()))
        helper_rows = {}
        for compiled in [False, True]:
            function = jax.jit(soilcarbon_kernels.soilcarbon_leak_doc_diffusion) if compiled else soilcarbon_kernels.soilcarbon_leak_doc_diffusion
            replay = function(captured["entry"], captured["temperature"], captured["boundaries"], captured["coefficient"])
            helper_rows["jit" if compiled else "eager"] = _compare({"doc": original_after, "doc_flux_diff": prior[f"{ident}/observer/flux"]}, {"doc": replay.doc, "doc_flux_diff": replay.doc_flux_diff})
            saved[f"{ident}/replay_{compiled}/doc"] = np.asarray(replay.doc)
        rows.append(dict(id=ident, real_strict_comparison=_compare({"doc": source_final}, {"doc": real_doc}), observed_final_byte_equal_to_real=bool(final_equal), observed_final_max_abs=float(np.abs(real_doc - observed_doc).max()), entry_exact_count=int(np.count_nonzero(difference)), entry_max_abs=float(difference.max()), entry_strict_pass=bool(np.all(difference <= ATOL + RTOL * np.maximum(np.abs(captured["entry"]), np.abs(original_entry)))), geometry_array_equal=bool(np.array_equal(captured["boundaries"], x["zf_soil_b"])), coefficient_array_equal=bool(np.array_equal(captured["coefficient"], np.full(2, 1e-5 * x["dt_days"]))), coefficient_source_formula="D_DOC * dt; stomate_soilcarbon.f90:1463", coefficient_hex=[float(v).hex() for v in captured["coefficient"]], temperature_array_equal=bool(np.array_equal(captured["temperature"], x["tprof"])), direct_helper_replay=helper_rows, branch_witness=branch_witness, limitation=None if final_equal else "Observation changed the compilation result; captured internal values are not established for the unobserved graph."))
        saved.update({f"{ident}/captured/{name}": value for name, value in captured.items()})
        saved[f"{ident}/unobserved/doc"] = real_doc
        saved[f"{ident}/observed/doc"] = observed_doc
    np.savez_compressed(directory / "bounded-entry-observation.npz", **saved)
    json_write(directory / "bounded-entry-observation.json", dict(kind="finite initial tuple only; no old annual attribution", rtol=RTOL, atol=ATOL, source_clone_change="one observational callback before diffusion; no process return edits", cases=rows))
    print(json.dumps(rows, indent=2))


def run(output=OUT, write_fixture=False):
    import jax

    from jax_orchidee.stomate.soilcarbon_kernels import soilcarbon_leak_doc_diffusion

    jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=True)
    saved, fixture, cases, receipts = {}, {}, [], []
    with tempfile.TemporaryDirectory(prefix="orcjax_phase06_soil_") as td:
        temp = Path(td)
        builds = {}
        for name, source in [("litter", _wrapper("litter")[0]), ("soil", soil_wrapper()[0]), ("observer", soil_wrapper(observe=True)[0]), ("cryo", soil_wrapper(cryo=True)[0])]:
            builds[name], receipt = _compile(source, name, temp, output)
            receipts.append(receipt)
        for spec in SPECS:
            ident, cryo = spec["id"], spec["route"] == "cryoturbation"
            x, raw = inputs(spec)
            for name, value in x.items():
                if isinstance(value, np.ndarray):
                    saved[f"{ident}/input/{name}"] = value
            saved[f"{ident}/input/zzc"] = raw["zzc"]
            for name, value in raw.items():
                if isinstance(value, np.ndarray):
                    saved[f"{ident}/raw/{name}"] = value
            inp, out = temp / "input.bin", temp / "output.bin"
            _write_inputs(inp, LITTER_INPUTS, x, raw)
            _execute(builds["litter"], out, inp, temp, output / f"{ident}.litter.log")
            litter = _read_fields(out, LITTER_FIELDS)
            # Zero activity/turnover makes the original litter boundary an
            # identity on the second call; this is checked, never assumed.
            np.testing.assert_array_equal(litter["litter_below"], x["litter_below"])
            sx = dict(x)
            sx.update({k: litter[k] for k in ["litter_above", "litter_below", "lignin_struc_above", "lignin_struc_below", "soilcarbon_input_doc", "floodcarbon_input", "bulk_dens"]})
            _write_inputs(inp, GEOMETRY_INPUTS, sx, raw)
            _execute(builds["cryo" if cryo else "soil"], out, inp, temp, output / f"{ident}.soil.log")
            soils = read_soil(out, cryo)
            source_log = (output / f"{ident}.soil.log").read_text(encoding="utf-8")
            row = dict(spec=spec, kind="finite synthetic", calls=[], original_mass_warning_count=source_log.count("Error: mass balance is not closed"))
            observer = None
            if not cryo:
                observed = temp / "observed.bin"
                observed_out = temp / "observed-output.bin"
                _execute(builds["observer"], observed_out, inp, temp, output / f"{ident}.observer.log", observed)
                np.testing.assert_array_equal(np.frombuffer(out.read_bytes(), np.uint8), np.frombuffer(observed_out.read_bytes(), np.uint8))
                observer = read_observer(observed)
                for name, value in observer.items():
                    saved[f"{ident}/observer/{name}"] = value
                    fixture[f"{ident}/observer/{name}"] = value
                row["original_guard_counts"] = {arm: int(observer["arms"][..., i].sum()) for i, arm in enumerate(ARMS)}
                row["observer_output_byte_identity"] = True
                row["physical_diffusion_doc_min"] = float(observer["after"].min())
                for compiled in [False, True]:
                    function = jax.jit(soilcarbon_leak_doc_diffusion) if compiled else soilcarbon_leak_doc_diffusion
                    helper = function(observer["entry"], x["tprof"], x["zf_soil_b"], np.full(2, 1e-5 * x["dt_days"]))
                    row[f"helper_{'jit' if compiled else 'eager'}"] = _compare({"doc": observer["after"], "doc_flux_diff": observer["flux"]}, {"doc": helper.doc, "doc_flux_diff": helper.doc_flux_diff})
            for index, soil in enumerate(soils):
                if cryo:
                    # Check the independently declared warm active-layer
                    # boundary; never feed oracle outputs into actual inputs.
                    np.testing.assert_array_equal(soil["save_altmax_ind"], x["altmax_ind"])
                    np.testing.assert_array_equal(soil["save_altmax_lastyear"], x["altmax_lastyear"])
                for name, value in x.items():
                    if isinstance(value, np.ndarray) or hasattr(value, "shape"):
                        saved[f"{ident}/call{index}/input/{name}"] = np.asarray(value)
                actual, result, next_x = actual_chain(x, cryo=cryo)
                expected = {k: litter[k] for k in ["litter_above", "lignin_struc_above", "lignin_struc_below", "litterpart", "dead_leaves", "deadleaf_cover", "resp_hetero_litter", "resp_hetero_flood", "soilcarbon_input_doc", "floodcarbon_input", "fuel_1hr", "fuel_10hr", "fuel_100hr", "fuel_1000hr"]}
                expected.update({k: soil[k] for k in ["carbon_32l", "doc", "doc_exp", "resp_hetero_soil", "resp_flood_soil", "interception_storage"]})
                expected["soil_litter_below"] = soil["litter_below"]
                expected["writer_litter_below"] = soil["litter_below"]
                if cryo:
                    expected.update({k: soil[k] for k, _ in COEFFICIENT_FIELDS})
                else:
                    expected["doc_flux_diff"] = observer["flux"]
                comparison = _compare(expected, actual)
                cryo_delta = {name: float(np.max(np.abs(np.asarray(getattr(result.transfers, f"cryoturbation_{name}_transfer"))))) for name in ["carbon", "doc", "litter"]}
                row["calls"].append(dict(index=index, comparison=comparison, cryoturbation_delta_max=cryo_delta, original_doc_min=float(soil["doc"].min()), actual_doc_min=float(actual["doc"].min()), finite=bool(all(np.isfinite(a).all() for a in expected.values()))))
                for name, value in expected.items():
                    fixture[f"{ident}/call{index}/{name}"] = value
                    saved[f"{ident}/call{index}/reference/{name}"] = value
                    saved[f"{ident}/call{index}/actual/{name}"] = actual[name]
                x = next_x
            cases.append(row)
    np.savez_compressed(output / "raw-inputs-outputs.npz", **saved)
    report = dict(status="reviewable finite evidence; no formal numbering", source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), spans={name: dict(start=start, end=end, sha256=hashlib.sha256(span_bytes(start, end)).hexdigest()) for name, start, end in SPANS}, rtol=RTOL, atol=ATOL, cases=cases, compile_receipts=receipts, annual_doc_boundary_observation=None, annual_mass_residual=None, annual_heterotrophic_respiration=None, limitations=["Finite synthetic component inputs, not natural historical observations.", "Numerical agreement and original guard witnesses do not establish physical nonnegativity or all-branch coverage.", "No annual DOC/respiration/floor observation; prior strict failures remain failed.", "Graph transport unavailable; exact-source fallback only."])
    report["preserved_initial_strict_failures"] = dict(path="initial-equality-diagnostic/", status="failed", cases=2, cause="Bounded pre-diffusion rounding plus strict equal-concentration update discontinuity; see entry observation. No source-semantic repair implemented.")
    report["caller_before_repair"] = "source-caller-before/"
    report["input_revision"] = "Sloped-background finite diffusion tuples are independent of preserved initial equality tuples. Three-call cryoturbation input is unchanged before/after root repair."
    json_write(output / "report.json", report)
    if write_fixture:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(FIXTURE, **fixture)
        json_write(META, {k: report[k] for k in ["source_sha256", "spans", "rtol", "atol"]} | {"specs": SPECS, "provenance": "Original uninstrumented Fortran complete procedures plus observer clone; see Phase06 report and raw arrays."})
    print(json.dumps({"cases": len(cases), "failed": [[case["spec"]["id"], call["index"], [r["field"] for r in call["comparison"] if not r["passed"]]] for case in cases for call in case["calls"] if any(not r["passed"] for r in call["comparison"])]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--write-fixture", action="store_true")
    parser.add_argument("--diagnose-initial", action="store_true")
    args = parser.parse_args()
    if args.diagnose_initial:
        diagnose_initial(args.output)
    else:
        run(args.output, args.write_fixture)
