"""Reproduce scalar-loss readout AD and test one exact XLA pass repair."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous diagnostic")
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, _devices = initialize_gpu_backend()
    import jax.numpy as jnp
    from jaxlib import xla_client
    from probe_shared_neural_precision import algorithm_linear
    from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import _linear
    jax.config.update("jax_enable_x64", True)
    report = {"status": "running", "scope": "synthetic_scalar_FP64_loss_native_readout_parameter_state_AD",
              "scientific_acceptance": False, "training_updates": 0, "custom_AD": False,
              "xla_flags": os.environ.get("XLA_FLAGS", ""), "rows": []}
    cases = [(4,128,256),(4,14,256),(4,28,256),(4,72,256),(4,24,256),(4,2,256)]
    rng = np.random.default_rng(20261001)
    for shape in cases:
        parameters = {"kernel": jnp.asarray(rng.normal(0,.03,(256,1)),jnp.float32), "bias":jnp.array([.02],jnp.float32)}
        state = jnp.asarray(rng.normal(size=shape),jnp.float64)
        mask = jnp.asarray((np.indices(shape[:-1]).sum(0)%3 != 0)[...,None])
        row = {"state_shape": list(shape), "kernel_shape": [256,1], "policies": {}}
        def objective(p,x,mask,linear):
            y = linear(p,x)
            physical = jnp.asarray(y,jnp.float64)
            loss = jnp.sum(jnp.where(mask,physical,0)**2)*.01
            return loss, y
        for policy, linear, options in (("original_FP32",_linear,{}),
                                       ("preset_env_without_operand_upcaster",algorithm_linear,{})):
            started = time.perf_counter()
            try:
                function = jax.jit(jax.value_and_grad(lambda p,x,m:objective(p,x,m,linear),argnums=(0,1),has_aux=True))
                lowered = function.lower(parameters,state,mask)
                executable = lowered.compile(compiler_options=options)
                result = jax.device_get(jax.block_until_ready(executable(parameters,state,mask)))
                prints = xla_client._xla.HloPrintOptions.short_parsable()
                prints.print_metadata=False; prints.print_large_constants=False
                prints.print_operand_shape=True; prints.print_operand_names=False
                hlo=executable.runtime_executable().hlo_modules()[0].to_string(prints)
                samples=[line.strip()[:1800] for line in hlo.splitlines() if 'cublas' in line or re.search(r'(?:^|\s)dot\(',line)]
                row["policies"][policy] = {"status":"supported", "elapsed_seconds":time.perf_counter()-started,
                    "all_finite":all(np.all(np.isfinite(v)) for v in jax.tree_util.tree_leaves(result)),
                    "loss":float(result[0][0]), "result_dtypes":[str(np.asarray(v).dtype) for v in jax.tree_util.tree_leaves(result)],
                    "parameter_gradient_nonzero":sum(int(np.count_nonzero(v)) for v in jax.tree_util.tree_leaves(result[1][0])),
                    "state_gradient_nonzero":int(np.count_nonzero(result[1][1])),
                    "compiler_options":options,"compiled_dot_gemm_examples":samples[:12]}
            except Exception as error:
                row["policies"][policy] = {"status":"unsupported_or_error","error":str(error),
                    "error_type":type(error).__name__,"elapsed_seconds":time.perf_counter()-started,"compiler_options":options}
        report["rows"].append(row)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    report["status"]="capability_measured"
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == "__main__":
    main()
