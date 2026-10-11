"""Write a false paid proposal from real source/commit identity, never approval.

Run after the accepted stage commit in the independent checkout. This does
not deploy, open scope, query GPU, reserve a ledger or submit Slurm. Candidate
mode retains code_git_head=null until a genuine clean accepted commit exists.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.hpc import run_shared_latent_T7_batch_numerics_paid as paid

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--candidate-source-only', action='store_true')
    parser.add_argument('--artifact-version', type=int, default=1)
    args = parser.parse_args(argv)
    if args.artifact_version < 1:
        raise ValueError('positive artifact version required')
    scope = paid.proposed_scope()
    actual_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, timeout=4, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, timeout=4, text=True)
    if not args.candidate_source_only:
        if dirty.strip():
            raise ValueError('freeze requires genuine committed clean accepted code')
        scope['code_git_head'] = actual_head
        scope['identity_status'] = 'actual clean committed code; proposed false, not approval'
    manifest = dict(actual_base_head=actual_head, source_files=scope['source_files'],
        scope_id=paid.SCOPE_ID, execution_authorized=False, code_git_head=scope['code_git_head'],
        candidate_source_only=args.candidate_source_only,
        checkout_target=paid.CHECKOUT_TARGET.as_posix(),
        inputs=scope['input_sha256'], output_directory=paid.OUTPUT_DIR.as_posix(),
        source_index_scope='Verify; E-ORCHIDEE-MAN-JAX generation2026-10-10T08:46:50Z ready; operated daily paths excluded/not_tracked under runtime/worktrees; exact-source fallback')
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for name, value in [(f'scope.proposed.v{args.artifact_version}.json', scope),
                        (f'deployment-manifest.v{args.artifact_version}.json', manifest)]:
        with (args.output_directory / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
    print(json.dumps(dict(status='proposal_saved_false', code_git_head=scope['code_git_head'],
                          output_directory=str(args.output_directory), GPU_executed=False)))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
