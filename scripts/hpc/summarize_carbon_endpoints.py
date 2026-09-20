"""Summarize the existing forward-only endpoint probe, without model execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(report):
    rows, groups = [], []
    hard = {}
    incomplete = 0
    for panel, samples in report["panels"].items():
        for sample in samples:
            for key, value in sample["hard"].items():
                hard[key] = hard.get(key, 0) + value
            for owner, item in sample["owners"].items():
                def compact(side, key):
                    nonlocal incomplete
                    record = item[side][key]["compact"] if side else item[key]["compact"]
                    incomplete += int(not record["complete"])
                    return record["value_gC_m2"]
                row = {"panel": panel, "selection": sample["selection"], "owner": owner,
                       "start": compact(None, "start")}
                for side in ("predicted", "teacher"):
                    for key in ("outgoing_amount", "incoming_amount", "pre_tail_endpoint", "post_tail_endpoint"):
                        row[f"{side}_{key}"] = compact(side, key)
                    row[f"{side}_effective_outgoing_fraction"] = (
                        row[f"{side}_outgoing_amount"] / row["start"] if row["start"] else None)
                row["outgoing_error_contribution"] = row["teacher_outgoing_amount"] - row["predicted_outgoing_amount"]
                row["incoming_error_contribution"] = row["predicted_incoming_amount"] - row["teacher_incoming_amount"]
                row["pre_tail_error"] = row["predicted_pre_tail_endpoint"] - row["teacher_pre_tail_endpoint"]
                row["post_tail_error"] = row["predicted_post_tail_endpoint"] - row["teacher_post_tail_endpoint"]
                row["tail_error_contribution"] = row["post_tail_error"] - row["pre_tail_error"]
                row["decomposition_residual"] = (row["pre_tail_error"] - row["outgoing_error_contribution"]
                                                  - row["incoming_error_contribution"])
                rows.append(row)
        for owner in sorted({r["owner"] for r in rows if r["panel"] == panel}):
            selected = [r for r in rows if r["panel"] == panel and r["owner"] == owner]
            numeric = [key for key in selected[0] if key not in ("panel", "owner", "selection")]
            group = {"panel": panel, "owner": owner, "count": len(selected)}
            for key in numeric:
                values = [r[key] for r in selected if r[key] is not None]
                group[key] = {"mean": float(np.mean(values)), "min": float(np.min(values)),
                              "max": float(np.max(values))} if values else None
            groups.append(group)
    return {"status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
            "new_optimizer_updates": report["new_optimizer_updates"],
            "gradient_evaluations": report["gradient_evaluations"],
            "sample_counts": {k: len(v) for k, v in report["panels"].items()},
            "hard_totals": hard, "incomplete_compact_records": incomplete,
            "max_abs_decomposition_residual": max(abs(r["decomposition_residual"]) for r in rows),
            "groups": groups, "rows": rows,
            "limits": ["16 selected one-day inputs, not all training/development data.",
                       "Outgoing fraction is start-stock-weighted, not an arithmetic mean over coordinates.",
                       "Endpoint outgoing/incoming encode pool changes, not identified physical transfers.",
                       "Attribution does not prove the cause of the GPP/NPP recurrent rebound."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.report.read_text(encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "endpoint-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Carbon endpoint attribution", "", "Panel means, gC/m2. No new inference or training.", "",
             "| Panel | Owner | Teacher outgoing | Predicted outgoing | Teacher incoming | Predicted incoming | Pre-tail error | Tail error contribution |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    fields = ("teacher_outgoing_amount", "predicted_outgoing_amount", "teacher_incoming_amount",
              "predicted_incoming_amount", "pre_tail_error", "tail_error_contribution")
    for row in result["groups"]:
        values = " | ".join(f"{row[key]['mean']:.7g}" for key in fields)
        lines.append(f"| {row['panel']} | {row['owner']} | {values} |")
    lines += ["", f"Incomplete compact records: {result['incomplete_compact_records']}.",
              f"Maximum arithmetic decomposition residual: {result['max_abs_decomposition_residual']:.3g} gC/m2.",
              "", *[f"- {item}" for item in result["limits"]]]
    (args.output_dir / "endpoint-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("groups", "rows")}, indent=2))


if __name__ == "__main__":
    main()
