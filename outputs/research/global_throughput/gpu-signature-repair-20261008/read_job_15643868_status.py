"""Read compact saved diagnostics only on test02; no model or JAX import."""
import json
from pathlib import Path

root = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512/capacity')
report = json.loads((root / 'report.json').read_text())
manifest = json.loads((root / 'stage_manifest.json').read_text())
print(json.dumps({'report': {key: value for key, value in report.items()
                             if any(word in key.lower() for word in ('status', 'error', 'exception', 'traceback', 'completed', 'decision'))},
                  'timing': report.get('timing'), 'compilations': report.get('compilations'),
                  'stages': {name: {key: value for key, value in record.items()
                                    if key in ('status', 'reason', 'sha256', 'measurement_status', 'strict_numerical_passed', 'startup', 'checkpoint_io')}
                             for name, record in manifest.get('stages', {}).items()}}, indent=2))
