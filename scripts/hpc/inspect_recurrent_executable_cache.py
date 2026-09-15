"""Inspect trusted local JAX 0.4.38 cache entries without compiling or training."""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path


def hlo_summary(text):
    operations, sources = Counter(), Counter()
    for line in text.splitlines():
        operation = re.search(r'\s([\w-]+)\([^)]*\)', line.partition(' = ')[2])
        if ' = ' not in line or operation is None:
            continue
        operations[operation[1]] += 1
        origin = re.search(r'source_file="([^"]+)"', line)
        if origin:
            sources[origin[1].split('/')[-1]] += 1
    return dict(text_bytes=len(text.encode()), instruction_count=sum(operations.values()),
                operations=dict(operations), source_instruction_counts=dict(sources))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-file', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    # The canonical GPU launcher supplies these; no Teacher payload is read.
    for name in ('dataset-manifest', 'acceptance-report', 'base-statistics',
                 'process-statistics', 'fast-statistics', 'prepared-manifest'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    from scripts.hpc.check_daily_training_runtime import _atomic_json
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend

    began = time.perf_counter()
    report = dict(status='running', entries=[], compiled_new_programs=0,
                  teacher_data_read=False, parameter_updates=0)

    def emit(stage):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-began)
        _atomic_json(args.output, report)
        print(json.dumps(dict(stage=stage, seconds=report['elapsed_seconds'])), flush=True)

    try:
        emit('initialize_gpu')
        jax, devices = initialize_gpu_backend()
        from jax._src import compilation_cache, xla_bridge
        from jax._src.lib import xla_client

        report.update(jax_version=jax.__version__, device=str(devices[0]))
        backend = xla_bridge.get_backend('gpu')
        for path in args.cache_file:
            row = dict(path=str(path))
            report['entries'].append(row)
            emit('read_cache')
            started = time.perf_counter()
            compressed = path.read_bytes()
            packed = compilation_cache.decompress_executable(compressed)
            serialized, seconds = compilation_cache.extract_executable_and_time(packed)
            row.update(compressed_bytes=len(compressed), serialized_bytes=len(serialized),
                       original_backend_compile_seconds=seconds,
                       read_decompress_seconds=time.perf_counter()-started)
            del compressed, packed
            emit('deserialize_executable')
            started = time.perf_counter()
            executable = backend.deserialize_executable(serialized, xla_client.CompileOptions())
            row['deserialize_seconds'] = time.perf_counter()-started
            del serialized
            emit('inspect_hlo')
            row['modules'] = [hlo_summary(module.to_string()) for module in executable.hlo_modules()]
            del executable
            emit('entry_completed')
        report['status'] = 'completed'
        emit('completed')
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        emit('failed')
        raise


if __name__ == '__main__':
    main()
