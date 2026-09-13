"""Create/verify a separate real-data fixture simulating interruption after one chunk."""
import argparse
import json
import shutil
from pathlib import Path


def run(args):
    evidence = args.output_dir/'fixture.json'
    if args.verify:
        fixture = json.loads(evidence.read_text())
        report = json.loads((args.output_dir/'prepared_manifest.json').read_text())
        retained = args.output_dir/fixture['retained_file']
        passed = (retained.stat().st_mtime_ns == fixture['mtime_ns']
                  and report['status'] == 'passed' and report['stage'] == 'completed'
                  and report['persisted_reader_exact'] and report['entries_reused'] == 1
                  and report['entries_prepared_this_invocation'] == 1)
        result = {'passed': passed, 'committed_file_not_rewritten': retained.stat().st_mtime_ns == fixture['mtime_ns'],
                  'entries_reused': report['entries_reused'],
                  'entries_prepared_this_invocation': report['entries_prepared_this_invocation'],
                  'invocation_seconds': report['invocation_elapsed_seconds'],
                  'fixture_only': True}
        (args.output_dir/'resume_probe.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        if not passed:
            raise ValueError('real-data partial resume fixture failed')
        return
    report = json.loads((args.source_dir/'prepared_manifest.json').read_text())
    if report['status'] != 'passed' or len(report['entries']) != 2:
        raise ValueError('fixture source must contain two accepted prepared chunks')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    first = report['entries'][0]
    target = args.output_dir/first['path']
    shutil.copy2(args.source_dir/first['path'], target)
    report.update(status='running', stage='point_year_completed', entries=[first],
                  simulated_interruption_fixture=True)
    report.pop('persisted_reader_exact', None)
    (args.output_dir/'prepared_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    evidence.write_text(json.dumps({'source': str(args.source_dir), 'retained_file': first['path'],
                                    'mtime_ns': target.stat().st_mtime_ns}, indent=2)+'\n')
    print('Separate partial-worker fixture created; source artifacts unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    run(parser.parse_args())
