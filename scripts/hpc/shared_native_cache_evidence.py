"""Observational native JAX cache metadata; never construct or replace keys."""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import logging
import os
import platform
import sys
from contextlib import contextmanager

import numpy as np


def _observed(call):
    try:
        return {"status": "observed", "value": call()}
    except Exception as error:
        return {"status": "unavailable", "error_type": type(error).__name__, "error": str(error)}


def native_computation_digest(module):
    """Use the runtime's canonicalizer, without modifying the supplied module."""
    from jax._src import cache_key
    result = _observed(lambda: hashlib.sha256(
        cache_key._canonicalize_ir(module, cache_key.IgnoreCallbacks.NO)).hexdigest())
    result['scope'] = 'JAX_canonicalized_IR_ignore_callbacks_NO_not_full_native_cache_key'
    return result


def runtime_key_inputs(backend, devices, compile_options, cache_key):
    import jax
    import jaxlib
    from jax._src.lib import xla_client
    serialized = _observed(lambda: compile_options.SerializeAsString())
    if serialized["status"] == "observed":
        payload = serialized.pop("value")
        serialized.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                          scope="raw_actual_options_not_native_normalized_component")
    configurations = {}
    for name in ("compilation_cache_include_metadata_in_key", "enable_compilation_cache",
                 "persistent_cache_enable_xla_caches", "persistent_cache_min_compile_time_secs",
                 "persistent_cache_min_entry_size_bytes", "compilation_cache_dir"):
        configurations[name] = _observed(lambda name=name: getattr(cache_key.config, name).value)
    observed_devices = []
    for device in np.asarray(devices, dtype=object).flat:
        observed_devices.append({name: _observed(lambda name=name, device=device: getattr(device, name))
                                 for name in ("id", "process_index", "device_kind", "platform")})
    hook = cache_key.custom_hook
    return {"jax_version": jax.__version__, "jaxlib_version": jaxlib.__version__,
            "optional_runtime_distribution_versions": {name: _observed(lambda name=name: importlib.metadata.version(name))
                for name in ('jax-cuda12-plugin', 'jax-cuda12-pjrt', 'nvidia-cuda-runtime-cu12', 'nvidia-cublas-cu12')},
            "native_cache_key_source_file": inspect.getfile(cache_key),
            "python_version": sys.version, "host_platform": platform.platform(),
            "actual_process_argv": list(sys.argv),
            "actual_environment_flags": {name: os.environ.get(name) for name in
                ("XLA_FLAGS", "LIBTPU_INIT_ARGS", "JAX_PLATFORMS", "JAX_ENABLE_X64",
                 "JAX_COMPILATION_CACHE_DIR", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")},
            "native_flag_prefixes": _observed(lambda: cache_key.get_flag_prefixes()),
            "native_key_configuration": configurations,
            "backend": {name: _observed(lambda name=name: getattr(backend, name))
                        for name in ("platform", "platform_version", "runtime_type")},
            "device_array_shape": list(np.asarray(devices, dtype=object).shape),
            "topology_serialized_sha256": _observed(lambda: hashlib.sha256(
                xla_client.get_topology_for_devices(tuple(np.asarray(devices, dtype=object).flat)).serialize()).hexdigest()),
            "devices": observed_devices, "raw_serialized_compile_options": serialized,
            "custom_hook_identity": {"module": getattr(hook, "__module__", None),
                                     "qualified_name": getattr(hook, "__qualname__", None)},
            "custom_hook_value": "not_called_by_observer; use_native_component_digest"}


@contextmanager
def observe_native_cache_keys(active_event):
    """Capture actual cache_key.get invocations and its native DEBUG digests.

    DEBUG causes JAX itself to hash each component again, including custom_hook.
    The observer does not invoke the hook or reproduce JAX's key algorithm.
    """
    from jax._src import cache_key
    original = cache_key.get
    signature = inspect.signature(original)
    logger = cache_key.logger
    old_level, old_propagate = logger.level, logger.propagate
    current = None

    class DigestHandler(logging.Handler):
        def emit(self, record):
            if current is None or len(record.args) != 2:
                return
            templates = {"get_cache_key hash of serialized %s: %s": "component_sha256",
                         "get_cache_key hash after serializing %s: %s": "cumulative_sha256"}
            field = templates.get(record.msg)
            if field:
                name, digest = record.args
                current[field][str(name)] = str(digest)

    def observed(*arguments, **keywords):
        nonlocal current
        event = active_event()
        if event is None:
            return original(*arguments, **keywords)
        bound = signature.bind(*arguments, **keywords)
        bound.apply_defaults()
        inputs = bound.arguments
        entry = {"status": "observing_native_get", "component_sha256": {}, "cumulative_sha256": {}}
        event.setdefault("native_cache_key_calls", []).append(entry)
        entry["inputs"] = runtime_key_inputs(inputs["backend"], inputs["devices"],
                                              inputs["compile_options"], cache_key)
        entry["compression_algorithm"] = inputs.get("compression_algorithm")
        entry["ignore_callbacks"] = str(inputs.get("ignore_callbacks"))
        previous, current = current, entry
        try:
            key = original(*arguments, **keywords)
            entry.update(status="observed_native_key", key=key)
            return key
        except Exception as error:
            entry.update(status='native_get_failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            current = previous

    handler = DigestHandler(level=logging.DEBUG)
    cache_key.get = observed
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        yield
    finally:
        cache_key.get = original
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def compare_recorded_signatures(old, new):
    """Retain legacy count checks; compare ordered types whenever recorded."""
    coarse = ("argument_leaves", "argument_nominal_bytes", "result_leaves", "result_nominal_bytes")
    if any(old.get(key) != new.get(key) for key in coarse if key in old):
        raise ValueError("completed stage module/signature changed")
    ordered = ("argument_types", "result_types")
    available = all(key in old for key in ordered)
    if available and any(old[key] != new[key] for key in ordered):
        raise ValueError("completed stage ordered interface changed")
    return {"ordered_prior_types": "recorded_and_equal" if available else "unavailable_in_legacy_report",
            "cache_compatibility": "requires_actual_native_cache_hit"}
