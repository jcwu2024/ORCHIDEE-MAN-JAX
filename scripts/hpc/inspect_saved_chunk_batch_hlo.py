"""Export a preserved daily optimized HLO without model execution/compilation."""
from __future__ import annotations

import argparse
import hashlib
import gzip
import pickle
from pathlib import Path

from probe_shared_command_buffer_loops import forbid_component_compile,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executable",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--compress-hlo",action="store_true")
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing evidence")
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax,devices=initialize_gpu_backend()
    from jax.experimental.serialize_executable import deserialize_and_load
    # Register prepared-data identities and finish import-time constants before
    # forbidding model compilation during saved-binary deserialization.
    from research.daily_coarse_graining.daily_operator import shared_process_head
    with forbid_component_compile(True):
        with args.executable.open("rb") as stream:
            loaded=deserialize_and_load(*pickle.load(stream))
        hlo=loaded.as_text()
    path=args.output.with_suffix(".optimized.hlo.txt.gz" if args.compress_hlo else ".optimized.hlo.txt")
    if args.compress_hlo:
        with gzip.open(path,"wt",compresslevel=1) as stream:
            stream.write(hlo)
    else:
        path.write_text(hlo)
    selected=[line for line in hlo.splitlines() if "custom_call_target=" in line and
        ("f32[512,256]" in line or "f32[512,1]" in line or "f32[256,1]" in line)]
    write(args.output,{"status":"exported","source_executable":str(args.executable),
        "jax_version":jax.__version__,"devices":[str(d) for d in devices],
        "hlo_artifact":str(path),"hlo_sha256":hashlib.sha256(hlo.encode()).hexdigest(),
        "zero_model_compile":True,"model_calls":0,"possible_original_micro4_readout_gemms":selected})


if __name__=="__main__":
    main()
