"""Public compatibility and fresh-process import-boundary regressions."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research.daily_coarse_graining.daily_operator"
# Ordered [public name, relative module, defining attribute] triples parsed from
# the eager initializer at 273157c. CI does not need the local preservation tag.
ORIGINAL_PUBLIC_MAP_SHA256 = "52c1caaa62df80ab4a01c0a138b87dd08109826ea85203a02bb4edf40d00ad58"


def fresh(code):
    env = os.environ.copy()
    env.update(JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def test_every_declared_export_and_order_matches_precleanup_snapshot():
    package = importlib.import_module(PACKAGE)
    assert len(package.__all__) == len(set(package.__all__)) == 350
    assert list(package._PUBLIC_EXPORTS) == package.__all__
    triples = [[public, module, attribute] for public, (module, attribute) in package._PUBLIC_EXPORTS.items()]
    fingerprint = hashlib.sha256(json.dumps(triples, separators=(",", ":")).encode()).hexdigest()
    assert fingerprint == ORIGINAL_PUBLIC_MAP_SHA256


def test_all_lazy_bindings_are_original_objects_and_cached():
    package = importlib.import_module(PACKAGE)
    for public, (module, attribute) in (package._PUBLIC_EXPORTS | package._COMPAT_EXPORTS).items():
        expected = getattr(importlib.import_module(module, PACKAGE), attribute)
        assert getattr(package, public) is expected
        assert vars(package)[public] is expected
    for name, module in package._COMPAT_MODULES.items():
        assert getattr(package, name) is importlib.import_module(module, PACKAGE)


def test_types_legacy_aliases_and_pickle_defining_modules_are_unchanged():
    package = importlib.import_module(PACKAGE)
    types = importlib.import_module(PACKAGE + ".types")
    assert package._COMPAT_EXPORTS == {
        name: (".types", name) for name in vars(types) if not name.startswith("_") and name not in package.__all__
    }
    for public, module in (
        ("CANDIDATE_NAME", "hierarchical_coordinate_attention"),
        ("COORDINATE_COUPLED_DOC_CANDIDATE", "coordinate_coupled_doc_attention"),
        ("DOC_STATUS_BRANCH_MAGNITUDE_CANDIDATE", "doc_status_branch_magnitude_attention"),
        ("TARGET_ORIENTED_ATTENTION_CANDIDATE", "target_oriented_attention"),
    ):
        assert getattr(package, public) is getattr(importlib.import_module(PACKAGE + "." + module), "CANDIDATE_NAME")
    for name in ("DailyOperatorInput", "NativeForcingInput", "CanonicalDailyOperatorInputAssembler", "StrongBaselineConfig"):
        obj = getattr(package, name)
        assert pickle.loads(pickle.dumps(obj)) is obj
        assert obj.__module__ != PACKAGE


def test_explicit_star_missing_attribute_and_dir_semantics():
    package = importlib.import_module(PACKAGE)
    namespace = {}
    exec(f"from {PACKAGE} import *", namespace)
    assert [name for name in namespace if name != "__builtins__"] == package.__all__
    assert "DailyOperatorInput" not in namespace  # Explicit-only compatibility type, as before.
    for name in package.__all__:
        assert namespace[name] is getattr(package, name)
    assert set(package.__all__) | set(package._COMPAT_EXPORTS) | set(package._COMPAT_MODULES) <= set(dir(package))
    with pytest.raises(AttributeError, match="has no attribute 'not_a_daily_operator_export'"):
        getattr(package, "not_a_daily_operator_export")


def test_fresh_package_and_current_child_do_not_load_unrelated_architectures():
    result = fresh(f"""
import json,sys
import {PACKAGE} as package
package_count = len(sys.modules)
assert [n for n in sys.modules if n.startswith(package.__name__)] == [package.__name__]
assert 'jax_orchidee.stomate.permafrost' in sys.modules
assert set(package.__all__) <= set(dir(package))
assert [n for n in sys.modules if n.startswith(package.__name__)] == [package.__name__]
import {PACKAGE}.shared_transformer
for old in ('successor_training','target_oriented_training','target_oriented_attention',
            'coordinate_coupled_doc_attention','doc_status_branch_magnitude_attention','hierarchical_coordinate_attention'):
    assert package.__name__ + '.' + old not in sys.modules, old
print(json.dumps({{'package_total_modules': package_count, 'child_total_modules':len(sys.modules),
                  'child_package_modules':len([n for n in sys.modules if n.startswith(package.__name__)])}}))
""")
    assert result["child_package_modules"] < 30


def test_fresh_first_public_lookups_inside_jit_keep_physical_initialization_safe():
    result = fresh(f"""
import json,jax,sys
import {PACKAGE} as package
def first_lookup(value):
    # The original eager package supported any first public lookup here.
    for name in package.__all__:
        getattr(package,name)
    return package.combine_signed_inventory(value,0.0)
value = jax.jit(first_lookup)(2.0)
assert float(value) == 2.0
for name,module in list(sys.modules.items()):
    if name.startswith((package.__name__,'jax_orchidee')):
        assert not any(isinstance(v,jax.core.Tracer) for v in vars(module).values()), name
from jax_orchidee.stomate.permafrost import _MOYANO_PEAT_MOISTURE_TABLE
import numpy as np
assert isinstance(_MOYANO_PEAT_MOISTURE_TABLE,np.ndarray)
assert np.isfinite(_MOYANO_PEAT_MOISTURE_TABLE).all()
print(json.dumps({{'result':float(value),'x64':jax.config.jax_enable_x64}}))
""")
    assert result == {"result": 2.0, "x64": True}


def test_fresh_current_child_import_inside_native_trace_is_safe():
    result = fresh(f"""
import json,jax,jax.numpy as jnp
import {PACKAGE} as package
def evaluate(value):
    from {PACKAGE}.shared_transformer import SharedTransformerConfig
    from {PACKAGE}.strong_baseline_architecture import _ffn
    config = SharedTransformerConfig(width=8,ffn_width=32)
    params = {{'first':{{'kernel':jnp.ones((8,32))*0.1,'bias':jnp.zeros((32,))}},
              'second':{{'kernel':jnp.ones((32,8))*0.1,'bias':jnp.zeros((8,))}}}}
    return jnp.sum(_ffn(params,value))
value,grad = jax.jit(jax.value_and_grad(evaluate))(jnp.ones((3,8)))
assert bool(jnp.isfinite(value)) and bool(jnp.isfinite(grad).all())
print(json.dumps({{'gradient_shape':list(grad.shape)}}))
""")
    assert result["gradient_shape"] == [3, 8]
