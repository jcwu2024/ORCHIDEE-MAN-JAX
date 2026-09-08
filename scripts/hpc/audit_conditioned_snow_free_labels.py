"""Read a bounded prepared-v6 panel on free CPU; never execute the Teacher."""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from research.daily_coarse_graining.daily_markov_contract import daily_markov_contract_from_metadata
from research.daily_coarse_graining.daily_operator.process_derivations import NativeForcingFieldLayout
from research.daily_coarse_graining.daily_operator.snow_free_boundary import HYDROL, warm_snow_free_regime


def run(root, output):
    jax.config.update("jax_enable_x64", True)
    prepared = root / "runtime/outputs/training/target-oriented-prepared-1000-a243029"
    manifest = json.loads((prepared / "prepared_manifest.json").read_text())
    dataset = json.loads((root / "runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json").read_text())
    contract = daily_markov_contract_from_metadata(dataset["markov_contract"])
    names = contract.native_forcing.fields
    rain, snow, temp = (names.index(n) for n in ("Rainf", "Snowf", "Tair"))
    layout = NativeForcingFieldLayout(rain, rain + 1, snow, snow + 1, temp, temp + 1)
    report = {"teacher_regenerated": False, "rows": []}
    expected = {"snow": 0., "snowdz": 0., "snowliq": 0., "snow_nobio": 0.,
                "snowheat": 0., "snowtemp": 273.15, "snowrho": 50., "snow_age": 0., "snowgrain": 0.}
    for update in (1, 322, 442):
        entry = next(e for e in manifest["entries"] if e["update"] == update)
        with gzip.open(prepared / entry["path"], "rb") as handle:
            artifact = pickle.load(handle)
        for horizon, batch in ((7, artifact.batch.seven_day), (30, artifact.batch.thirty_day)):
            selected = np.asarray(warm_snow_free_regime(batch.inputs.continuous_state,
                                                      batch.inputs.forcing, contract, layout))
            row = {"artifact": update, "horizon": horizon, "days": int(selected.size),
                   "selected": int(selected.sum()), "selected_by_window": selected.sum(axis=1).tolist(), "endpoints": {}}
            for name, reference in expected.items():
                leaf = next(v for v in contract.state_leaves if v.key == f"{HYDROL}.{name}")
                values = np.asarray(batch.expected_next_state.other)[..., leaf.start:leaf.stop]
                defined = np.asarray(batch.expected_next_state_defined.other)[..., leaf.start:leaf.stop]
                active = np.broadcast_to(selected[..., None], values.shape)
                row["endpoints"][name] = {
                    "undefined_selected": int(np.sum(active & ~defined)),
                    "nonfinite_selected": int(np.sum(active & ~np.isfinite(values))),
                    "max_abs_error": float(np.max(np.abs(values[active] - reference), initial=0.)),
                }
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
    output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
