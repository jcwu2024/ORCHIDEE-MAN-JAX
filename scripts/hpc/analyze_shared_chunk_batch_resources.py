"""Completed free-GPU launcher accounting, including startup and failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    rows=[]
    for path in sorted(args.root.glob("shared-command-buffer-loops-*-20261001-vmap*.launcher.txt")):
        values=dict(line.split("=",1) for line in path.read_text().splitlines() if "=" in line)
        if values.get("runtime_profile") != "legacy":
            raise ValueError("unexpected runtime profile in vmap accounting")
        rows.append({"path":str(path),"exit_status":int(values["exit_status"]),
            "gpu_started":values["gpu_started"]=="true","gpu_count":int(values["gpu_count"]),
            "observed_gpu_memory_util":values["observed_gpu_memory_util"],
            "launcher_wall_seconds":float(values["launcher_wall_seconds"]),
            "launcher_card_hours":float(values["launcher_card_hours"])})
    report={"schema":"shared_chunk_batch_completed_launcher_accounting_v1",
        "scope":"completed_launcher_records_only_active_process_exposure_not_included",
        "completed_launches":len(rows),"gpu_process_launches":sum(r["gpu_started"] for r in rows),
        "paid_jobs":0,"launcher_wall_seconds":sum(r["launcher_wall_seconds"] for r in rows),
        "launcher_card_hours":sum(r["launcher_card_hours"] for r in rows),"records":rows}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="records"}))


if __name__=="__main__":
    main()
