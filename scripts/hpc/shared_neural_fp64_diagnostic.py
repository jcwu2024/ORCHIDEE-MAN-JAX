"""Temporary source-driven FP64 policy for one diagnostic process only.

Recompile neural functions with only jnp.float32 dtype references changed.
Preserve serialized class identities and all physical functions. Restore aliases
and policy properties on exit; this module is never imported by production.
"""
from __future__ import annotations
import ast
import __future__
from contextlib import contextmanager
import inspect
import hashlib
import importlib
import sys
import numpy as np

PACKAGE = 'research.daily_coarse_graining.daily_operator.'
MODULES = ('input_conditioning', 'strong_baseline_architecture', 'shared_transformer',
           'shared_decoder_dense', 'shared_decoder_macro_packed', 'shared_decoder_expert_execution',
           'shared_backbone_projection_reuse', 'shared_backbone_execution')


class NeuralDtypes(ast.NodeTransformer):
    def __init__(self):
        self.lines = []
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'jnp' and node.attr == 'float32':
            self.lines.append(node.lineno)
            node.attr = 'float64'
        return self.generic_visit(node)


@contextmanager
def neural_fp64_functions(modules):
    """Apply to already loaded canonical and admitted staged neural modules."""
    if tuple(m.__name__.removeprefix(PACKAGE) for m in modules) != MODULES:
        raise ValueError('exact audited neural module set/order required')
    originals = []
    replacements = {}
    manifests = []
    properties = []
    try:
        for module in modules:
            source = inspect.getsource(module)
            tree = ast.parse(source)
            functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            converter = NeuralDtypes()
            transformed = converter.visit(ast.Module(body=functions, type_ignores=[]))
            transformed = ast.fix_missing_locations(transformed)
            before = {node.name:module.__dict__[node.name] for node in functions}
            exec(compile(transformed, module.__file__, 'exec', flags=__future__.annotations.compiler_flag), module.__dict__)
            for name, old in before.items():
                new = module.__dict__[name]
                originals.append((module, name, old))
                replacements[id(old)] = new
            manifests.append(dict(module=module.__name__, source=module.__file__,
                                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                  fp32_dtype_lines_changed=sorted(converter.lines),
                                  functions_recompiled=sorted(before)))
        # Keep existing dataclass objects; update only diagnostic dtype getters.
        policy = modules[1].NeuralPrecisionPolicy
        for name in ('attention_score_dtype', 'source_family_pooling_dtype'):
            old = getattr(policy, name)
            properties.append((policy, name, old))
            # Imported jnp belongs to the already loaded neural module.
            setattr(policy, name, property(lambda self, jnp=modules[1].jnp: jnp.float64))
        # from-import bindings must point to the same transformed functions.
        # This includes the staged adapters and shared_process_head bindings.
        for module in list(sys.modules.values()):
            if module is None or not getattr(module, '__name__', '').startswith(PACKAGE):
                continue
            for name, value in list(vars(module).items()):
                new = replacements.get(id(value))
                if new is not None and new is not value:
                    originals.append((module, name, value))
                    setattr(module, name, new)
        yield replacements, manifests
    finally:
        for owner, name, value in reversed(originals):
            setattr(owner, name, value)
        for owner, name, value in properties:
            setattr(owner, name, value)


@contextmanager
def exact_physical_constant_widening():
    """Keep the canonical R4 epsilon value (2**-23), represented exactly in FP64.

    No process function is recompiled or formula changed. Season lines26,
    304/310/576 and permafrost lines25/57/142/170 use this source threshold.
    Their prior FP32 constant -> FP64 conversions are numerically unchanged.
    """
    saved=[];defaults=[];manifest=[]
    try:
        for name in ('season','permafrost'):
            module=importlib.import_module('jax_orchidee.stomate.'+name)
            old=module.FORTRAN_EPSILON_R4
            new=np.float64(old)
            if float(old)!=2.0**-23 or float(new)!=float(old):
                raise ValueError('canonical R4 epsilon identity differs')
            saved.append((module,old))
            module.FORTRAN_EPSILON_R4=new
            changed=[]
            for function in vars(module).values():
                if not inspect.isfunction(function) or function.__module__!=module.__name__:continue
                kw=function.__kwdefaults__
                if kw and kw.get('epsilon') is old:
                    defaults.append((function,kw))
                    function.__kwdefaults__={**kw,'epsilon':new}
                    changed.append(function.__name__)
            manifest.append(dict(module=module.__name__,source=module.__file__,symbol='FORTRAN_EPSILON_R4',
                original_dtype=str(np.asarray(old).dtype),prepared_dtype=str(new.dtype),
                original_value=float(old),prepared_value=float(new),exact_value_unchanged=True,
                epsilon_defaults=changed,process_code_unchanged=True))
        yield manifest
    finally:
        for function,kw in reversed(defaults):function.__kwdefaults__=kw
        for module,value in reversed(saved):module.FORTRAN_EPSILON_R4=value
