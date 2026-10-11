"""Bounded forward audit of two exact Git revisions at production carbon entries.

No global physical rollout, external data, training or GPU execution. Git-backed
imports keep each worker's entire jax_orchidee module context at its declared pin.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import linecache
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ATOL, RTOL = 1e-8, 1e-10
FRACTIONS = np.array([0.002545695295720209, 0.011313425835462453, 0.0434304139324322, 0.9427104649363851])
AGES = np.array([3080.6829585821606, 3082.6110769428087, 3085.793962047229, 3103.533185648637])
CASE_NAMES = (
    "ordinary_growth",
    "ordinary_pumping",
    "tiny_active_mass",
    "tiny_active_maintenance",
    "zero_mass",
    "negative_mass",
    "below_cutoff",
    "cutoff_previous",
    "cutoff_exact",
    "cutoff_next",
    "absent_pft",
    "laidev_mask",
    "bare_soil",
    "natural_grass",
)


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


class GitModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Load only tracked Python source at one pin; never fall back to HEAD."""

    def __init__(self, pin):
        self.pin, self.sources = pin, {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "jax_orchidee" and not fullname.startswith("jax_orchidee."):
            return None
        relative = fullname.replace(".", "/")
        package = (ROOT / relative).is_dir()
        return importlib.util.spec_from_loader(fullname, self, is_package=package)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        relative = module.__name__.replace(".", "/")
        path = relative + ("/__init__.py" if module.__spec__.submodule_search_locations is not None else ".py")
        if module.__spec__.submodule_search_locations is not None and not (ROOT / path).exists():
            module.__path__ = [str(ROOT / relative)]
            return
        source = git("show", f"{self.pin}:{path}").decode("utf-8")
        self.sources[path] = source
        filename = f"git:{self.pin}:{path}"
        module.__file__ = str(ROOT / path)
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        exec(compile(source, filename, "exec"), module.__dict__)


def fixture(case):
    """Explicit small post-allocation boundary, patterned on integration tests.

    Tests/unit/test_stomate_integration.py's kill/gap/turnover composition supplies
    the ordinary parameter layout. Tiny masses/classes reuse the imported
    turnover conditioning regression; no process formula is implemented here.
    """

    def f(shape, value=0.0):
        return np.full(shape, value, dtype=np.float64)

    cutoff = np.sqrt(np.finfo(np.float64).tiny)
    masses = [
        100.0,
        10.0,
        5.804965023785553e-139,
        1.0,
        0.0,
        -1e-139,
        1e-155,
        np.nextafter(cutoff, 0.0),
        cutoff,
        np.nextafter(cutoff, np.inf),
        100.0,
        100.0,
        100.0,
        100.0,
    ]
    i, nvm, pft = CASE_NAMES.index(case), 14, (0 if case == "bare_soil" else 13)
    mass = masses[i]
    biomass = f((1, nvm, 12, 1))
    # carbon_kernels.py: ILEAF=0, ISAPABOVE=1, IROOT=5, IFRUIT=6.
    biomass[0, pft, [0, 1, 5, 6], 0] = mass * np.array([1.0, 0.3, 0.4, 0.1])
    present = np.zeros((1, nvm), bool)
    present[0, pft] = case != "absent_pft"
    allocation, maintenance = f((1, nvm, 12)), f((1, nvm, 12))
    allocation[0, pft, [0, 5]] = [0.6, 0.4]
    gpp = f((1, nvm))
    if case.startswith("ordinary"):
        gpp[0, pft] = 10.0 if case == "ordinary_growth" else 1.0
        maintenance[0, pft, [0, 5]] = np.array([0.7, 0.3]) * (1.0 if case == "ordinary_growth" else 3.0)
    elif case == "tiny_active_maintenance":
        gpp[0, pft] = 1e-176
        maintenance[0, pft] = [
            1.1959175918532169e-176,
            9.470227295278714e-178,
            6.318572123333474e-178,
            3.023573544806176e-188,
            2.8914988965610444e-188,
            6.142238652329241e-177,
            9.605021040731308e-180,
            1.8174336179312717e-188,
            3.2129167169936666e-178,
            6.488322052701464e-179,
            2.920848646739295e-188,
            3.080744086822504e-188,
        ]
    ages, fractions = f((1, nvm, 4)), f((1, nvm, 4))
    ages[0, pft] = 0.0 if case.startswith("cutoff") or case == "below_cutoff" else AGES
    fractions[0, pft] = FRACTIONS
    tree = np.zeros(nvm, bool)
    tree[pft] = case != "natural_grass"
    laidev = np.zeros(nvm, bool)
    laidev[pft] = case == "laidev_mask"
    result = dict(
        gpp_daily=gpp,
        biomass_before=biomass,
        resp_maint_part=maintenance,
        f_alloc=allocation,
        pft_present=present,
        frac_growthresp=f(nvm, 0.25),
        leaf_age=ages,
        leaf_frac=fractions,
        age=f((1, nvm), 2.0),
        is_tree=tree,
        natural=np.ones(nvm, bool),
        sla_age1=f((1, nvm), 0.015),
        sla_calc=f((1, nvm), 0.02),
        sla_max=f(nvm, 0.05),
        sla_min=f(nvm, 0.01),
        herbivores=f((1, nvm)),
        maxmoiavail_lastyear=f((1, nvm), 1.0),
        minmoiavail_lastyear=f((1, nvm)),
        moiavail_week=f((1, nvm), 0.1),
        t2m_longterm=f(1, 296.73314693372953),
        t2m_month=f(1, 293.15),
        t2m_week=f(1, 293.15),
        gdd_from_growthinit=f((1, nvm)),
        lai=f((1, nvm), 2.0),
        turnover_time=f((1, nvm)),
        nrec=np.zeros((1, nvm), np.int32),
        senescence_type=np.zeros(nvm, np.int32),
        is_grassland_manag=np.zeros(nvm, bool),
        ok_laidev=laidev,
        min_leaf_age_for_senescence=f(nvm, 30.0),
        gdd_senescence=f(nvm, 100.0),
        senescence_temp=f((nvm, 3)),
        hum_frac=f(nvm, 0.5),
        senescence_hum=f(nvm, 0.2),
        nosenescence_hum=f(nvm, 0.8),
        max_turnover_time=f(nvm, 80.0),
        min_turnover_time=f(nvm, 10.0),
        leaffall=f(nvm, 10.0),
        lai_max=f(nvm, 4.0),
        leafagecrit=f(nvm, 730.0),
        lai_initmin=f(nvm, 0.3),
        tau_fruit=f(nvm, 90.0),
        tau_sap=f(nvm, 730.0),
        lm_lastyearmax=f((1, nvm), 50.0),
        ind=f((1, nvm), 1.0),
        cn_ind=f((1, nvm), 2.0),
        bm_to_litter=f((1, nvm, 12, 1)),
        senescence=np.zeros((1, nvm), bool),
        rip_time=f((1, nvm), 1.0),
        when_growthinit=f((1, nvm), 6.0),
        everywhere=f((1, nvm), 1.0),
        veget_max=present.astype(np.float64),
        height=f((1, nvm), 1.0),
        pasture=np.zeros(nvm, bool),
        npp_longterm=f((1, nvm), 100.0),
        turnover_longterm=f((1, nvm, 12, 1)),
        t2m_min_daily=f(1, 280.0),
        tmin_spring_time=f((1, nvm)),
        availability_fact=f(nvm, 0.14),
        residence_time=f(nvm, np.inf),
        tmin_crit=f(nvm, np.inf),
        leaf_tab=np.zeros(nvm, np.int32),
        pheno_type=np.zeros(nvm, np.int32),
        maxdia=f(nvm, 10.0),
    )
    return result


def callchain(pin):
    targets = {
        "jax_orchidee/driver/global_batch.py": ("make_batch_transition", "make_scalar_transition"),
        "jax_orchidee/driver/orchestration.py": (
            "_paper_later_day_block_transition_factory",
            "paper_1961_driver_later_day_runtime_result",
            "_paper_day_stomate_daily_carbon_from_bundles",
        ),
        "jax_orchidee/stomate/integration.py": (
            "stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_outer_compiled",
            "_daily_carbon_static_dispatch",
            "stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit",
            "stomate_daily_carbon_kill_gap_turnover_explicit",
            "stomate_daily_carbon_explicit",
        ),
        "jax_orchidee/stomate/carbon_kernels.py": ("npp_closed_update", "turnover_step", "turnover_leaf_age_fall"),
    }
    names = {name for group in targets.values() for name in group}
    records = []
    for path, wanted in targets.items():
        source = git("show", f"{pin}:{path}").decode("utf-8")
        source_lines = source.splitlines()
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                calls = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        callee = getattr(child.func, "id", getattr(child.func, "attr", None))
                        if callee in names or callee == "runner":
                            calls.append(
                                {
                                    "callee": callee,
                                    "line": child.lineno,
                                    "source_line": source_lines[child.lineno - 1].strip(),
                                    "route_arguments": {
                                        keyword.arg: ast.get_source_segment(source, keyword.value)
                                        for keyword in child.keywords
                                        if keyword.arg
                                        in {
                                            "module_jit",
                                            "use_static_jit_daily_carbon",
                                            "outer_compiled_daily_carbon_dispatch",
                                            "outer_compiled_static_dispatch",
                                            "static_dispatch",
                                        }
                                    },
                                }
                            )
                records.append(
                    {
                        "file": path,
                        "function": node.name,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "calls": sorted(calls, key=lambda call: call["line"]),
                        "indirect_callable_bindings": [
                            {
                                "line": child.lineno,
                                "end_line": child.end_lineno,
                                "source": ast.get_source_segment(source, child),
                            }
                            for child in ast.walk(node)
                            if isinstance(child, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == "runner" for target in child.targets)
                        ],
                    }
                )
    return records


def worker(pin, output):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    import jax
    import jaxlib

    @jax.tree_util.register_pytree_node_class
    class StaticText:
        """Keep returned trace text as static metadata across JIT/vmap."""

        def __init__(self, value):
            self.value = value

        def tree_flatten(self):
            return (), self.value

        @classmethod
        def tree_unflatten(cls, value, children):
            return cls(value)

        def __repr__(self):
            return f"StaticText({self.value!r})"

    def with_static_text(fn):
        def wrapped(inputs):
            return jax.tree_util.tree_map(
                lambda leaf: StaticText(leaf) if isinstance(leaf, str) else leaf,
                fn(inputs),
            )

        return wrapped

    if any(name == "jax_orchidee" or name.startswith("jax_orchidee.") for name in sys.modules):
        raise RuntimeError("worker must begin without a pre-imported scientific module")
    loader = GitModules(pin)
    sys.meta_path.insert(0, loader)
    from jax_orchidee.stomate import integration
    from jax_orchidee.stomate.carbon_kernels import turnover_step

    fixtures = [fixture(case) for case in CASE_NAMES]
    digest = hashlib.sha256()
    for case, inputs in zip(CASE_NAMES, fixtures, strict=True):
        digest.update(case.encode())
        for key, value in sorted(inputs.items()):
            digest.update(f"{key}:{value.shape}:{value.dtype}".encode())
            digest.update(value.tobytes())
    npp_keys = set(
        integration.stomate_daily_carbon_explicit.__code__.co_varnames[
            : integration.stomate_daily_carbon_explicit.__code__.co_argcount
            + integration.stomate_daily_carbon_explicit.__code__.co_kwonlyargcount
        ]
    )
    turnover_keys = set(
        turnover_step.__code__.co_varnames[
            : turnover_step.__code__.co_argcount + turnover_step.__code__.co_kwonlyargcount
        ]
    )

    def npp(inputs):
        return integration.stomate_daily_carbon_explicit(
            **{key: value for key, value in inputs.items() if key in npp_keys}
        )

    def post_npp(inputs):
        return integration.stomate_daily_carbon_kill_gap_turnover_explicit(
            **inputs, lpj_gap_const_mort=True, ok_dgvm=False
        )

    def turnover(inputs):
        arguments = {key: value for key, value in inputs.items() if key in turnover_keys}
        arguments["biomass"] = inputs["biomass_before"]
        return turnover_step(**arguments)

    data, structures = {}, {}
    stacked = jax.tree_util.tree_map(lambda *parts: np.stack(parts), *fixtures)
    for entry, fn in (("npp_adapter", npp), ("post_npp_production_adapter", post_npp), ("turnover_step", turnover)):
        fn = with_static_text(fn)
        print(json.dumps({"pin": pin, "entry": entry, "phase": "start"}), flush=True)
        compiled = jax.jit(fn)
        for mode in ("scalar_eager", "scalar_jit", "vmapped_jit"):
            if mode == "vmapped_jit":
                value = jax.jit(jax.vmap(fn))(stacked)
            else:
                values = [(fn if mode == "scalar_eager" else compiled)(inputs) for inputs in fixtures]
                value = jax.tree_util.tree_map(lambda *parts: np.stack([np.asarray(part) for part in parts]), *values)
            value = jax.device_get(value)
            structures[f"{entry}/{mode}"] = str(jax.tree_util.tree_structure(value))
            for path, leaf in jax.tree_util.tree_flatten_with_path(
                value, is_leaf=lambda leaf: isinstance(leaf, StaticText)
            )[0]:
                array = np.full(len(CASE_NAMES), leaf.value) if isinstance(leaf, StaticText) else np.asarray(leaf)
                data[f"{entry}/{mode}/{jax.tree_util.keystr(path)}"] = array
            print(json.dumps({"pin": pin, "entry": entry, "mode": mode, "phase": "complete"}), flush=True)
    np.savez_compressed(output.with_suffix(".npz"), **data)
    summary = {
        "pin": pin,
        "pid": os.getpid(),
        "fixture_sha256": digest.hexdigest(),
        "cases": list(CASE_NAMES),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "platform": jax.devices()[0].platform,
            "x64": bool(jax.config.jax_enable_x64),
        },
        "structures": structures,
        "imported_science_files": sorted(loader.sources),
        "array_records": len(data),
        "callchain": callchain(pin),
        "fixture_controls": {
            "lpj_gap_const_mort": True,
            "ok_dgvm": False,
            "wire_cover": False,
            "wire_vmax": False,
            "residence_time": "positive infinity: zero gap mortality; isolates turnover cutoff",
            "min_stomate": 0.0,
            "nvm": 14,
            "nparts": 12,
            "nelements": 1,
            "npts": 1,
            "outer_vmap_lanes": len(CASE_NAMES),
        },
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def compare_leaf(reference, candidate):
    result = {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "shape_equal": reference.shape == candidate.shape,
        "dtype_equal": reference.dtype == candidate.dtype,
    }
    if not result["shape_equal"] or not result["dtype_equal"]:
        return {**result, "passed": False}
    exact = np.array_equal(reference, candidate)
    result["exact"] = bool(exact)
    if reference.dtype.kind not in "fc":
        return {**result, "discrete_exact": bool(exact), "passed": bool(exact)}
    finite = np.isfinite(reference).all() and np.isfinite(candidate).all()
    if not finite:
        return {**result, "finite": False, "passed": False}
    delta = np.abs(candidate - reference)
    nonzero = reference != 0
    relative = np.divide(delta, np.abs(reference), out=np.zeros_like(delta), where=nonzero)
    result.update(
        finite=True,
        max_abs=float(delta.max(initial=0)),
        max_relative_nonzero=float(relative.max(initial=0)),
        reference_zero_count=int((~nonzero).sum()),
        changed_reference_zero_count=int(((~nonzero) & (delta != 0)).sum()),
        max_bound_ratio=float((delta / (ATOL + RTOL * np.abs(reference))).max(initial=0)),
    )
    result["passed"] = result["max_bound_ratio"] <= 1
    return result


def compare_workers(directory, baseline, candidate):
    old = json.loads((directory / "baseline.json").read_text())
    new = json.loads((directory / "candidate.json").read_text())
    records = []
    with (
        np.load(directory / "baseline.npz", allow_pickle=False) as before,
        np.load(directory / "candidate.npz", allow_pickle=False) as after,
    ):
        keys_equal = set(before.files) == set(after.files)
        for key in sorted(set(before.files) & set(after.files)):
            metrics = compare_leaf(before[key], after[key])
            field = key.rsplit(".", 1)[-1]
            stock = field in {"biomass", "biomass_before_alloc", "bm_alloc", "bm_to_litter", "turnover"}
            metrics.update(path=key, stock_output=stock)
            if stock and metrics.get("exact") is not True:
                metrics["passed"] = False
            records.append(metrics)
    meta_passed = (
        old["pin"] == baseline
        and new["pin"] == candidate
        and old["pid"] != new["pid"]
        and old["fixture_sha256"] == new["fixture_sha256"]
        and old["runtime"] == new["runtime"]
        and old["structures"] == new["structures"]
        and keys_equal
        and bool(records)
    )
    leaf_fraction_records = [record for record in records if record["path"].endswith(".leaf_frac")]
    numeric_records = [record for record in records if "finite" in record]
    stock_records = [record for record in records if record["stock_output"]]
    discrete_records = [record for record in records if "discrete_exact" in record]
    result = {
        "baseline": baseline,
        "candidate": candidate,
        "atol": ATOL,
        "rtol": RTOL,
        "passed": meta_passed and all(record["passed"] for record in records),
        "metadata_gates_passed": meta_passed,
        "stock_exact_required": True,
        "relative_metric": "abs(candidate-baseline)/abs(baseline) on nonzero baseline coordinates; zeros counted separately",
        "array_records": len(records),
        "case_count": len(CASE_NAMES),
        "summary": {
            "exact_records": sum(record.get("exact", False) for record in records),
            "changed_records": [record["path"] for record in records if record.get("exact") is False],
            "non_leaf_fraction_records_exact": all(
                record.get("exact") is True for record in records if record not in leaf_fraction_records
            ),
            "stock_records": len(stock_records),
            "stock_records_exact": all(record["exact"] for record in stock_records),
            "discrete_and_text_records": len(discrete_records),
            "discrete_and_text_records_exact": all(record["discrete_exact"] for record in discrete_records),
            "numeric_values_finite": all(record["finite"] for record in numeric_records),
            "leaf_frac_max_abs": max((record.get("max_abs", 0.0) for record in leaf_fraction_records), default=0.0),
            "leaf_frac_max_relative_nonzero": max(
                (record.get("max_relative_nonzero", 0.0) for record in leaf_fraction_records), default=0.0
            ),
            "leaf_frac_max_bound_ratio": max(
                (record.get("max_bound_ratio", 0.0) for record in leaf_fraction_records), default=0.0
            ),
            "leaf_frac_reference_zero_count": sum(
                record.get("reference_zero_count", 0) for record in leaf_fraction_records
            ),
            "leaf_frac_changed_reference_zero_count": sum(
                record.get("changed_reference_zero_count", 0) for record in leaf_fraction_records
            ),
        },
        "worker_evidence": {"baseline": old, "candidate": new},
        "records": records,
        "failures": [record for record in records if not record["passed"]],
        "limitations": [
            "Local CPU runtime, not canonical JAX0.4.38 CUDA/GPU validation",
            "One explicit post-allocation production boundary; no prescribe/allocation, half-hour driver, complete-day or multi-day execution",
            "Synthetic declared small inputs, no FULL64/Teacher/reference/golden mutation",
            "Forward-only comparison; imported AD regressions are reported separately, no training/AD method change",
            "Unchanged native dtype, masks and complete returned leaves; no GPU readiness or source certificate admission inferred",
        ],
    }
    (directory / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-pin")
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_pin:
        worker(git("rev-parse", args.worker_pin).decode().strip(), args.worker_output)
        return 0
    baseline, candidate = [git("rev-parse", pin).decode().strip() for pin in (args.baseline, args.candidate)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for name, pin in (("baseline", baseline), ("candidate", candidate)):
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--worker-pin",
            pin,
            "--worker-output",
            str(args.output_dir / name),
        ]
        with (args.output_dir / f"{name}-execution.log").open("w", encoding="utf-8") as log:
            status = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if status:
            print(json.dumps({"status": "worker_failed", "worker": name, "returncode": status}), flush=True)
            return status
    result = compare_workers(args.output_dir, baseline, candidate)
    print(
        json.dumps({key: result[key] for key in ("passed", "baseline", "candidate", "array_records", "case_count")}),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
