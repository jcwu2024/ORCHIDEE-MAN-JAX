"""Gate-E1 true-daily conservative operator skeleton."""

from .audit import (
    audit_executable_jaxpr,
    audit_inference_source_graph,
    audit_process_label_bindings,
)
from .canonical_retained_tail import (
    CanonicalRetainedTailAdapter,
    CanonicalRetainedTailStaticOwner,
    build_canonical_retained_tail_adapter,
)
from .conservative_update import (
    apply_conservative_inventory_update,
    apply_conservative_update,
    combine_signed_inventory,
    split_signed_inventory,
    validate_conservative_result,
)
from .input_assembly import CanonicalDailyOperatorInputAssembler
from .model import InputAssembler, ProcessHead, bind_daily_operator, daily_operator_transition
from .native_forcing import (
    assemble_native_forcing_features,
    encode_native_forcing,
    validate_native_forcing,
)
from .process_derivations import (
    NativeForcingFieldLayout,
    derive_carbon_decomposition,
    derive_litter_input,
    integrate_native_precipitation,
)
from .process_heads import (
    PROCESS_LABEL_BINDINGS,
    classify_differentiability_case,
    parameterize_inventory_prediction,
    parameterize_thermal_prediction,
    run_minimal_process_heads,
)
from .retained_tail import (
    ExactRetainedTailAdapter,
    RetainedTailAdapter,
    plumbing_test_retained_tail_adapter,
)
from .types import *  # noqa: F403

__all__ = [
    "ExactRetainedTailAdapter",
    "CanonicalDailyOperatorInputAssembler",
    "CanonicalRetainedTailAdapter",
    "CanonicalRetainedTailStaticOwner",
    "InputAssembler",
    "PROCESS_LABEL_BINDINGS",
    "ProcessHead",
    "RetainedTailAdapter",
    "apply_conservative_inventory_update",
    "apply_conservative_update",
    "assemble_native_forcing_features",
    "audit_executable_jaxpr",
    "audit_inference_source_graph",
    "audit_process_label_bindings",
    "bind_daily_operator",
    "build_canonical_retained_tail_adapter",
    "classify_differentiability_case",
    "combine_signed_inventory",
    "daily_operator_transition",
    "derive_carbon_decomposition",
    "derive_litter_input",
    "encode_native_forcing",
    "parameterize_inventory_prediction",
    "parameterize_thermal_prediction",
    "plumbing_test_retained_tail_adapter",
    "run_minimal_process_heads",
    "integrate_native_precipitation",
    "NativeForcingFieldLayout",
    "split_signed_inventory",
    "validate_conservative_result",
    "validate_native_forcing",
]
