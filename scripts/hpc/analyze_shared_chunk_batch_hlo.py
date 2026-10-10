"""Extract exact optimized GEMM dimensions/configs from saved chunk probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    stems = {"original_map":"shared-command-buffer-loops-threshold-component-default-20261001.default",
        "whole_vmap":"shared-command-buffer-loops-chunk-vmap-default-20261001-vmap2.vmap",
        "preserve128":"shared-command-buffer-loops-shape-preserved-default-20261001-vmap.preserve128"}
    evidence = {}
    for name,stem in stems.items():
        evidence[name] = {}
        for kind in ("forward","native_vjp"):
            path = args.root / f"{stem}.{kind}.hlo.txt"
            if not path.exists():
                continue
            lines = path.read_text().splitlines()
            records = [{"hlo":line.strip(),"backend_config":json.loads(line.split("backend_config=")[-1])}
                for line in lines if 'custom_call_target="__cublas' in line]
            evidence[name][kind] = {"source":str(path),"gemms":records,
                "reductions":[line.strip() for line in lines if " reduce(" in line and "metadata=" in line]}
    args.output.write_text(json.dumps({"schema":"shared_chunk_batch_optimized_hlo_evidence_v1",
        "interpretation":"optimized shapes/configs are evidence of changed execution, not sufficient localization of every numerical difference",
        "evidence":evidence},indent=2)+"\n")
    for name,rows in evidence.items():
        for kind,record in rows.items():
            print(name,kind,len(record["gemms"]),"GEMMs")
            for gemm in record["gemms"][:2]:
                print(gemm["hlo"].split("metadata=")[0],gemm["backend_config"])


if __name__ == "__main__":
    main()
