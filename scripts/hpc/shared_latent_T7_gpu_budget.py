"""Independent 14400 GPU-second, one-card corrected latent-reader T7 grant."""
import argparse
import importlib.util
from pathlib import Path
import sys

LIMIT=14400
LEDGER_NAME='latent-reader-T7-20261005.gpu-budget.json'
SCOPE='latent_reader_corrected_T7_14400_20261005'
APPROVAL=('User authorization 2026-10-05: independent cumulative14400 GPU-seconds, '
          'each session<=7200 seconds, at most one idle free V100. Corrected original '
          'and latent32x384/private reader64x2 canonical field/chunk64 full T7 native '
          'parameter/initial-state AD; at most one necessary original and one candidate '
          'model graph; immediately save successful executables and never recompile. '
          'No paid jobs, T30 graph or optimizer updates. Startup, compilation, recovery, '
          'persistence and cleanup charged; historical funding excluded.')

def validate_grant(data):
    if (data.get('limit_gpu_seconds')!=LIMIT or data.get('session_limit_seconds')!=7200
            or data.get('independent_funding_scope')!=SCOPE or data.get('approval_note')!=APPROVAL):
        raise ValueError('independent T7 grant identity differs')
    for row in data.get('sessions',()):
        if row.get('cards')!=1 or not 20<=row.get('session_cap_seconds',0)<=7200:
            raise ValueError('T7 grant contains invalid session/card record')

def run(args):
    global LIMIT, LEDGER_NAME, SCOPE, APPROVAL
    migration = getattr(args, 'migration_7200', False)
    if migration:
        LIMIT = 7200
        LEDGER_NAME = 'latent-reader-T7-20261006-free-migration.gpu-budget.json'
        SCOPE = 'latent_reader_T7_free_migration_7200_20261006'
        APPROVAL = ('User authorization 2026-10-06: move the same not-yet-run two-hour paid T7 '
                    'comparison to one genuinely idle free gln01 V100; new cumulative7200 GPU-seconds '
                    'including startup, compile, recovery, matched pairs, saving and cleanup, never '
                    'reset per retry. Reuse corrected original and every successful candidate graph; '
                    'no T30, learning, architecture, data, loss, precision, AD, sampling or gate changes. '
                    'Historical14400 ledger and all prior charges are preserved and excluded.')
    if args.ledger.name!=LEDGER_NAME or not 20<=args.seconds<=7200:
        raise ValueError('exact isolated T7 ledger and bounded session required')
    spec=importlib.util.spec_from_file_location('latent_T7_isolated_budget',Path(__file__).with_name('shared_gpu_budget.py'))
    budget=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=budget
    spec.loader.exec_module(budget)
    budget.LIMIT=LIMIT
    with budget.locked(args.ledger) as data:
        if not data.get('sessions') and 'independent_funding_scope' not in data:
            data.update(independent_funding_scope=SCOPE,approval_note=APPROVAL,
                        authorized_gpu_seconds=LIMIT,historical_funds_excluded=True)
            if migration:
                prior = args.ledger.with_name('latent-reader-T7-20261005.gpu-budget.json')
                import json
                old = json.loads(prior.read_text())
                if any(row.get('status') != 'finished' for row in old['sessions']):
                    raise ValueError('historical T7 ledger must be settled before migration')
                data['historical_ledger_reference'] = {'path': str(prior),
                    'settled_sessions': len(old['sessions']),
                    'charged_gpu_seconds': sum(budget.charge(row) for row in old['sessions'])}
        validate_grant(data)
    args.cards=1
    args.approved_limit_gpu_seconds=LIMIT
    args.approval_note=APPROVAL
    args.funding_scope=None
    return budget.run(args)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--stage',choices=('runtime-preflight','compile-original','compile-candidate','matched','saved-recovery'),required=True)
    parser.add_argument('--seconds',type=int,required=True)
    parser.add_argument('--migration-7200',action='store_true')
    parser.add_argument('command',nargs=argparse.REMAINDER)
    raise SystemExit(run(parser.parse_args()))
