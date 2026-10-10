"""Semantic engineering metrics for the isolated four-sample precision screen.

The scientific objective and native AD are supplied by shared_daily_dual_ad.
These host diagnostics never modify gradients or select a training policy.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


def parameter_groups(parameters):
    result = {}
    for root, value in parameters.items():
        if root == "shared_transformer":
            result.update({root + "." + key: child for key, child in value.items()})
        else:
            result[root] = value
    return result


def semantic_states(gradient, contract):
    values = np.asarray(gradient)
    leaves = [leaf for leaf in contract.state_leaves if not leaf.discrete]
    if values.shape != (4, 3854) or len(leaves) != 257:
        raise ValueError("requires four complete states and 257 continuous fields")
    coverage = np.zeros(3854, dtype=np.int32)
    for leaf in leaves:
        coverage[leaf.start:leaf.stop] += 1
    if not np.all(coverage == 1):
        raise ValueError("state fields do not exactly partition each sample")
    return {
        f"sample{sample}/{leaf.key}": values[sample, leaf.start:leaf.stop]
        for sample in range(4) for leaf in leaves
    }


def alias_metadata(modules):
    aliases = []
    for module in modules:
        for name in ("_linear", "_ffn"):
            if hasattr(module, name):
                aliases.append({"module": module.__name__, "alias": name,
                                "original_owner": getattr(module, name).__module__})
    return {
        "patched_aliases": aliases,
        "algorithm": "F16_F16_F32", "storage": "FP32", "accumulation": "FP32",
        "macro_readout": "both captured _linear aliases in _decode_mapped_rows readout_first/readout_second are patched",
        "unpatched": ["strong_baseline_architecture tokenizer helpers", "carbon encoders using original helpers",
                      "attention score einsum", "attention value aggregation einsum", "physical processes FP64"],
        "all_neural_dot_kinds_patched": False,
    }


def source_mask_evidence(jax, context, adapter_path):
    """Audit exact mask sources and evaluate the same source-derived masks per row.

    Raw decoder masks depend on layout/PFT masks and token *shape*, never the
    precision-sensitive token values. Undefined-state gradients are checked
    independently against each input assembler's actual state_defined mask.
    """
    from research.daily_coarse_graining.daily_operator import shared_transformer as st
    package = "research.daily_coarse_graining.daily_operator."
    # Load the exact staged siblings before auditing; the caller subsequently
    # reloads them in the candidate owner, so compare the file source itself.
    adapter_source = Path(adapter_path).read_text()
    backbone_source = Path(adapter_path).with_name("shared_backbone_execution.py").read_text()
    macro_source = Path(adapter_path).with_name("shared_decoder_macro_packed.py").read_text()
    baseline = sys.modules[package + "strong_baseline_architecture"]
    base_source = inspect.getsource(baseline.assemble_strong_baseline_context)
    active_source, active_line = inspect.getsourcelines(st._field_active_and_pft_owner)
    active_source = "".join(active_source)
    checks = {
        "base_pft_mask_is_input_boolean": "pft_mask = jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)" in base_source,
        "backbone_copies_base_pft_mask": "pft_mask=jnp.asarray(base.pft_mask, dtype=jnp.bool_)" in backbone_source,
        "active_uses_pft_boolean_and_shape": "active &= context.pft_mask[safe]" in active_source and "context.pft_tokens.shape[0]" in active_source,
        "active_does_not_use_neural_helpers": all(name not in active_source for name in ("_linear", "_ffn", "einsum", "dot_general")),
        "field_masks_restore_full_active": "masks_by_index[field_index] = full_active.reshape(field.shape)" in adapter_source,
        "gpp_returns_source_active_mask": "masks_by_index[field_index] = active" in adapter_source and "return raw.reshape(field.shape), full_active.reshape(field.shape)" in inspect.getsource(st._decode_field),
        "macro_masks_inactive_readout": "return jnp.where(on, raw, 0.0)" in macro_source,
    }
    if not all(checks.values()):
        raise ValueError(f"source mask audit does not match declared route: {checks}")
    defined_masks, raw_rows = [], []
    head = context["head"]
    for row in range(4):
        sample = jax.tree_util.tree_map(lambda value: value[row], context["daily4"])
        assembled = head.input_assembler.assemble(sample.inputs)
        defined = np.asarray(assembled.state_defined.other)
        if defined.dtype != np.bool_ or defined.shape != (3854,):
            raise ValueError("input assembler did not provide each actual boolean state mask")
        defined_masks.append(defined)
        pft = np.asarray(assembled.pft.active_pft_mask)
        dummy = SimpleNamespace(pft_tokens=np.empty((pft.size, 0)), pft_mask=pft)
        rows = {}
        for field, ownership in zip(head.learned_schema.fields, head.layout.ledger, strict=True):
            _coords, active, _owner = st._field_active_and_pft_owner(dummy, field, ownership, head.layout)
            active = np.asarray(active, dtype=np.bool_)
            rows[field.name] = {"coordinates": int(active.size), "active": int(np.count_nonzero(active)),
                                "mask_sha256": hashlib.sha256(active.tobytes()).hexdigest()}
        raw_rows.append(rows)
    masks = np.stack(defined_masks)
    return masks, {
        "checks": checks, "source_audit_passed": all(checks.values()),
        "raw_mask_precision_invariance": "exact source dependency audit: same layout, PFT booleans and token shapes; no precision-sensitive token values enter masks",
        "raw_mask_numeric_outputs_observed": False,
        "raw_masks_by_sample": raw_rows,
        "undefined_state_coordinates_by_sample": [int(np.count_nonzero(~mask)) for mask in masks],
        "source": {"active": {"file": inspect.getfile(st._field_active_and_pft_owner), "line": active_line, "sha256": hashlib.sha256(active_source.encode()).hexdigest()},
                   "staged_adapter": str(adapter_path), "staged_adapter_sha256": hashlib.sha256(adapter_source.encode()).hexdigest()},
        "structural_inactive_parameter_and_output_gradient_audit": "incomplete; arbitrary reference zeros are descriptive only",
    }


def observed_validation(jax, result, parameters):
    gradients, structure = jax.tree_util.tree_flatten(result[1][0])
    leaves, expected = jax.tree_util.tree_flatten(parameters)
    if structure != expected or len(gradients) != 300:
        raise ValueError("full 300-leaf parameter tree changed")
    if any(np.shape(a) != np.shape(b) or np.asarray(a).dtype != np.asarray(b).dtype
           for a, b in zip(gradients, leaves, strict=True)):
        raise ValueError("parameter gradient shape/dtype changed")
    if np.shape(result[1][1]) != (4, 3854):
        raise ValueError("four state gradient shapes changed")
    values = jax.tree_util.tree_leaves(result)
    hard = {name: int(np.asarray(value)) for name, value in result[0][1]["hard"].items()}
    return {"gradient_leaves": len(gradients), "gradient_coordinates": sum(np.size(v) for v in gradients),
            "gradient_nonzero": sum(int(np.count_nonzero(v)) for v in gradients),
            "initial_state_gradient_coordinates": int(np.size(result[1][1])),
            "initial_state_gradient_nonzero": int(np.count_nonzero(result[1][1])),
            "all_values_finite": all(np.all(np.isfinite(np.asarray(v))) for v in values),
            "hard_counts": hard, "hard_count_total": len(hard), "physical_hard_passed": len(hard) == 11 and not any(hard.values())}


def undefined_states_zero(gradient, masks):
    values, defined = np.asarray(gradient), np.asarray(masks)
    if values.shape != (4, 3854) or defined.shape != values.shape or defined.dtype != np.bool_:
        raise ValueError("actual per-sample defined-mask shape/dtype differs")
    return [bool(np.all(values[row][~defined[row]] == 0)) for row in range(4)]


def engineering_comparison(jax, precision, actual, reference, contract, masks, mask_evidence):
    groups = parameter_groups(actual[1][0])
    expected = parameter_groups(reference[1][0])
    count = sum(np.size(v) for v in jax.tree_util.tree_leaves(groups))
    if len(groups) != 16 or count != 64_884_481 or groups.keys() != expected.keys():
        raise ValueError("16 semantic parameter groups do not cover full model")
    parameters = {name: precision.vector_metrics(jax, value, expected[name]) for name, value in groups.items()}
    state_a, state_b = semantic_states(actual[1][1], contract), semantic_states(reference[1][1], contract)
    states = {name: precision.vector_metrics(jax, value, state_b[name]) for name, value in state_a.items()}
    loss_a, loss_b = float(actual[0][0]), float(reference[0][0])
    delta = abs(loss_a - loss_b)
    relative = None if abs(loss_b) <= 1e-12 else delta / abs(loss_b)
    loss_passed = np.isfinite(loss_a) and np.isfinite(loss_b) and (delta <= 1e-6 if relative is None else relative <= 1e-3)
    all_finite = all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves((actual, reference)))
    undefined = {
        name: undefined_states_zero(value[1][1], masks)
        for name, value in (("candidate", actual), ("field", reference))
    }
    hard = {name: int(np.asarray(value)) for name, value in actual[0][1]["hard"].items()}
    physical = len(hard) == 11 and not any(hard.values())
    source_masks = mask_evidence["source_audit_passed"] and all(mask_evidence["checks"].values())
    rows = list(parameters.values()) + list(states.values())
    passed = loss_passed and all_finite and physical and source_masks and all(all(v) for v in undefined.values()) and all(row["screening_passed"] for row in rows)
    return {"parameters": parameters, "state_fields_by_sample": states,
            "parameter_groups": len(parameters), "parameter_coordinates": count, "state_groups": len(states),
            "loss_absolute_difference": delta, "loss_relative_difference": relative, "loss_screening_passed": bool(loss_passed),
            "all_values_finite": bool(all_finite), "undefined_input_state_gradients_zero_by_sample": undefined,
            "raw_masks_source_invariance_passed": bool(source_masks), "candidate_hard_counts": hard,
            "physical_hard_passed": physical, "screening_passed": bool(passed),
            "failed_parameter_groups": [name for name, row in parameters.items() if not row["screening_passed"]],
            "failed_state_fields_by_sample": [name for name, row in states.items() if not row["screening_passed"]],
            "inactive_reference_gradient_coordinates": "not used as a structural-inactive gate; incomplete source audit disclosed"}
