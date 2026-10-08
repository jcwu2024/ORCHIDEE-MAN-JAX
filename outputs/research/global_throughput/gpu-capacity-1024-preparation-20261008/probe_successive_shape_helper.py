"""Bounded real DIFFUCO helper probe, never a physical-model transition."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.code_dir, text=True).strip()
    clean = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=args.code_dir, text=True)
    if head != args.expected_code_sha or clean:
        raise ValueError("declared helper source checkout must be exact and clean")
    sys.path.insert(0, str(args.code_dir))
    import jax
    import numpy as np

    from jax_orchidee.sechiba import diffuco

    jax.config.update("jax_enable_x64", True)
    if jax.__version__ != "0.4.38" or any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("canonical CPU JAX0.4.38 required")
    diffuco._diffuco_pft_column_mask.cache_clear()
    evidence = []
    lowered_stages = []
    for size in (512, 1024):
        backgrounds = [np.arange(size * 14, dtype=np.float64).reshape(size, 14) + index for index in range(5)]
        columns = [np.full((size, 1), 100 + index, dtype=np.float64) for index in range(5)]

        def insertion(*values):
            return diffuco._diffuco_insert_pft_output_columns(*values, pft_index=13)

        with jax.checking_leaks():
            lowered = jax.jit(insertion).lower(*(backgrounds + columns))
        info = lowered._lowering.compile_args["all_args_info"]
        public = len(jax.tree_util.tree_leaves(backgrounds + columns))
        internal = len(info.in_avals)
        if public != 10 or internal != public:
            raise AssertionError("tiny helper input signature drift")
        outputs = lowered.compile()(*(backgrounds + columns))
        for output, background, column in zip(outputs, backgrounds, columns, strict=True):
            expected = background.copy()
            expected[:, 13:14] = column
            actual = np.asarray(output)
            np.testing.assert_array_equal(actual, expected)
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise AssertionError("native helper shape/dtype drift")
        cached = diffuco._diffuco_pft_column_mask(14, 13)
        if not isinstance(cached, np.ndarray) or cached.dtype != np.bool_ or cached.flags.writeable:
            raise AssertionError("mask host cache must stay read-only NumPy Boolean")
        lowered_stages.append(lowered)
        evidence.append({"shape": size, "public_inputs": public, "internal_inputs": internal,
            "five_native_outputs_equal_independent_numpy": True,
            "host_mask_is_immutable_numpy_bool": True})
    if len(lowered_stages[0]._lowering.compile_args["all_args_info"].in_avals) != 10:
        raise AssertionError("earlier helper signature changed after successive shape")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=args.code_dir, text=True):
        raise AssertionError("helper probe wrote to immutable checkout")
    result = {"status": "passed", "jax": jax.__version__, "code_commit": head,
        "affinity": sorted(os.sched_getaffinity(0)), "stages": evidence,
        "earlier_signature_unchanged": True, "wall_seconds": time.perf_counter() - started,
        "scope": "actual insertion helper only; no real model preparation/lowering/compile/invocation or GPU initialization",
        "fixed1024_model_runtime_acceptance": False}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
