"""Prepare 24 disjoint validation windows from existing v6 data on free CPU."""
import json
from pathlib import Path

from research.daily_coarse_graining.daily_operator.conditioned_independent_panel import build_panel
from research.daily_coarse_graining.daily_operator.corrected_continuation import load_contract
from scripts.hpc import prepare_target_oriented_validation as prepare

ROOT = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
CONSUMED = ('target-oriented-validation-9b82b5a', 'structured-gpp-fresh-validation-c3ea51a',
            'successor-staged-validation-a2c2638')


def main():
    parser = prepare._parser()
    parser.add_argument('--continuation-contract', type=Path, required=True)
    args = parser.parse_args()
    plan = load_contract(args.continuation_contract).plan
    panels = [json.loads((ROOT / 'runtime/outputs/training' / name / 'validation_panel.json').read_text()) for name in CONSUMED]
    result = prepare.run(args, panel_builder=lambda index, target: build_panel(index, target, plan, panels))
    print(json.dumps({'status': result['status'], 'windows': result['window_count']}))


if __name__ == '__main__':
    main()
