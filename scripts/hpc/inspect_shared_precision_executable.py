"""Bounded compact compiled-HLO dtype evidence without a model compilation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-evidence", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous inspection")
    started = time.perf_counter()
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend()
    from jax.experimental.serialize_executable import deserialize_and_load
    from jaxlib import xla_client
    from jax._src import compiler
    previous = compiler.backend_compile
    import_helper_compilations = Counter()

    def import_helper_compile(*positional, **keywords):
        module = positional[1]
        name = str(module.operation.attributes["sym_name"])
        if "jit_objective" in name:
            raise RuntimeError("inspection import must not compile a model objective")
        import_helper_compilations[name] += 1
        return previous(*positional, **keywords)

    # Pickle treedef types import original modules that initialize small fixed
    # tables (iota/broadcast arithmetic). Record these helper compilations,
    # then forbid every compile before loading the serialized model program.
    compiler.backend_compile = import_helper_compile
    try:
        with jax.default_device(jax.devices("cpu")[0]), args.executable.open("rb") as stream:
            serialized = pickle.load(stream)
    finally:
        compiler.backend_compile = previous

    def forbidden(*args, **kwargs):
        raise RuntimeError("inspection must not compile a model")

    compiler.backend_compile = forbidden
    try:
        executable = deserialize_and_load(*serialized)
        del serialized
        options = xla_client._xla.HloPrintOptions.short_parsable()
        options.print_metadata = False
        options.print_large_constants = False
        options.print_backend_config = True
        options.print_operand_shape = True
        options.print_operand_names = args.scale_evidence
        options.print_program_shape = False
        modules = executable.runtime_executable().hlo_modules()
        counts, signatures, examples = Counter(), Counter(), {}
        lengths, digests = [], []
        scale_evidence=[]
        for module in modules:
            # Exclude source metadata and large constants before rendering.
            # The compact representation stays in memory; no full HLO artifact.
            compact = module.to_string(options)
            lengths.append(len(compact))
            digests.append(hashlib.sha256(compact.encode()).hexdigest())
            if len(compact) > 256 * 2**20:
                raise MemoryError("compact representation exceeded 256 MiB inspection bound")
            if args.scale_evidence:
                from collect_shared_precision_scale_hlo import collect
                scale_evidence.append(collect(compact))
            for line in io.StringIO(compact):
                rhs = line.partition(" = ")[2]
                if "custom_call_target=" in rhs and re.search(r'custom_call_target="__(?:cublas|cudnn)', rhs):
                    kind = "compiled_gpu_gemm_or_dnn_custom_call"
                elif re.search(r"(?:^|\s)dot\(", rhs):
                    kind = "compiled_dot"
                elif re.search(r"(?:^|\s)convert\(", rhs):
                    counts["compiled_convert"] += 1
                    continue
                else:
                    continue
                counts[kind] += 1
                output = rhs.split("(", 1)[0]
                types = re.findall(r"\b(f16|f32|f64|bf16)\[", rhs)
                operand_types = re.findall(r"\b(f16|f32|f64|bf16)\[", rhs.split("custom-call(", 1)[-1].split("),", 1)[0]) if "custom-call(" in rhs else []
                target = re.search(r'custom_call_target="([^"]+)"', rhs)
                signature = json.dumps({"kind": kind, "target": target.group(1) if target else None,
                                        "float_types_in_instruction": types,
                                        "operand_float_types": operand_types,
                                        "output_type_prefix": output[:120]}, sort_keys=True)
                signatures[signature] += 1
                if signature not in examples and len(examples) < 32:
                    examples[signature] = rhs[:2400].rstrip()
            del compact
        report = {
            "status": "inspected", "source": str(args.executable), "jax_version": jax.__version__,
            "gpu_count": len(devices), "paid_jobs": 0, "model_compilations": 0,
            "model_evaluations": 0, "modules": len(modules), "instruction_counts": dict(counts),
            "CPU_import_table_helper_compilations": dict(import_helper_compilations),
            "dynamic_scale_compiled_dataflow": scale_evidence,
            "float_signatures": [{**json.loads(signature), "count": count,
                                  "example": examples.get(signature)} for signature, count in signatures.items()],
            "compact_hlo_characters": lengths, "compact_hlo_sha256": digests,
            "print_metadata": False, "print_large_constants": False, "full_hlo_exported": False,
            "elapsed_seconds": time.perf_counter() - started,
            "claim_limit": "compiled HLO operand/output dtype signatures only; does not establish device-level TensorCore utilization or all-neural-dot coverage",
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "float_signatures"}))
    finally:
        compiler.backend_compile = previous


if __name__ == "__main__":
    main()
