"""Proposed FFN-only native F16 diagnostic; requires a new experiment approval.

Neural formula provenance: strong_baseline_architecture.py::_linear/_ffn
886-898. Only three active FFN aliases change; physics/loss remain external.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys

BASE_SHA256 = 'f2e4233fd3d4c8bbe7a81da75b54d98dc7c2d0dd71bcaf65dde5b8ffeb39f45d'
BASE_PATH = Path(__file__).with_name('probe_shared_neural_precision.py')
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256:
    raise ValueError('native policy helper base version drift')
spec = importlib.util.spec_from_file_location('ffn_precision_locked_base', BASE_PATH)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
CONTRACT = dict(base.CONTRACT)
vector_metrics = base.vector_metrics
algorithm_ffn = base.algorithm_ffn
TARGETS = ('shared_transformer', 'shared_backbone_projection_reuse', 'shared_decoder_macro_packed')
_EVIDENCE = {}


def alias_metadata(modules):
    return {'patched_aliases': [{'module': m.__name__, 'alias': '_ffn'} for m in modules
                                if m.__name__.rsplit('.', 1)[-1] in TARGETS],
            'algorithm': 'F16_F16_F32', 'storage': 'FP32', 'accumulation': 'FP32',
            'all_linear_aliases': 'original FP32; including readout/input/bridge/local/attention',
            'physics': 'FP64', 'all_neural_dot_kinds_patched': False}


@contextmanager
def neural_policy(modules, *, mixed):
    """Patch exactly the three active FFNs and restore aliases on all exits."""
    global _EVIDENCE
    indexed = {m.__name__.rsplit('.', 1)[-1]: m for m in modules}
    if len(indexed) != len(modules) or set(indexed) != set(TARGETS) | {'shared_backbone_execution'}:
        raise ValueError('active four policy module set changed')
    package = indexed['shared_transformer'].__name__.rsplit('.', 1)[0]
    original = sys.modules[package + '.strong_baseline_architecture']
    saved_linear = [(m, m._linear) for m in modules]
    saved_ffn = [(indexed[name], indexed[name]._ffn) for name in TARGETS]
    if any(fn is not original._linear for _, fn in saved_linear) or any(fn is not original._ffn for _, fn in saved_ffn):
        raise ValueError('FFN-only policy requires original FP32 aliases')
    macro = indexed['shared_decoder_macro_packed']
    adapter = sys.modules.get(package + '.shared_decoder_expert_execution')
    decoder = getattr(adapter, 'expert_decode_fields', None)
    if decoder is None or decoder.__globals__ is not macro.__dict__:
        raise ValueError('active expert decoder does not resolve the actual macro globals')
    _EVIDENCE = {'helper_base_file': str(BASE_PATH), 'helper_base_sha256': BASE_SHA256,
                 'active_ffn_aliases': [{'module': m.__name__, 'file': str(Path(m.__file__).resolve()),
                    'sha256': hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
                    'alias': '_ffn', 'original_owner': fn.__module__} for m,fn in saved_ffn],
                 'expert_decode_fields_globals_is_macro_module_dict': True,
                 'macro_sys_modules_identity': sys.modules[macro.__name__] is macro,
                 'linear_aliases_preserved': True, 'mixed': mixed,
                 'algorithm': 'F16_F16_F32', 'custom_AD': False}
    try:
        if mixed:
            for module, _ in saved_ffn:
                module._ffn = algorithm_ffn
        yield
        if any(m._linear is not fn for m,fn in saved_linear):
            raise ValueError('an FP32 linear alias changed during FFN-only lowering')
    finally:
        for module, fn in saved_ffn:
            module._ffn = fn
        for module, fn in saved_linear:
            module._linear = fn


def policy_evidence():
    return dict(_EVIDENCE)
