"""Archive inactive checkout working files, retaining all Git dependency stores.

Run on a free CPU node. Default is an inventory; --apply packs, compares against
the originals, and only then removes working files of explicitly selected trees.
This utility never removes .git, training data, checkpoints, or compilation caches.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime')


def process_references(root):
    """Expose only checkout references, never unrelated environment values."""
    references = []
    prefixes = [str(root / name) + '/' for name in ('checkouts', 'worktrees')]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        # Privilege-separated sshd/Singularity supervisors can hide cwd/environ;
        # keep available fields and report those limitations, including the PID.
        values, unavailable = [], []
        for field in ('cmdline', 'cwd', 'environ'):
            try:
                if field == 'cwd':
                    values.append(os.readlink(entry / field))
                else:
                    items = (entry / field).read_bytes().decode(errors='replace').split('\0')
                    if field == 'environ':
                        items = [item.partition('=')[2] for item in items
                                 if item.partition('=')[0] in ('CODE_ROOT', 'PYTHONPATH',
                                     'ORCHIDEE_REPO_ROOT', 'SINGULARITYENV_PYTHONPATH',
                                     'SINGULARITYENV_ORCHIDEE_REPO_ROOT')]
                    values += items
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                unavailable.append(field)
        found = set()
        for value in values:
            for prefix in prefixes:
                for suffix in value.split(prefix)[1:]:
                    tail = suffix.split('/')[0].split(':')[0]
                    if tail:
                        found.add(str(Path(prefix) / tail))
        if found or unavailable:
            try:
                command_name = (entry / 'comm').read_text().strip()
            except (FileNotFoundError, ProcessLookupError):
                continue
            references.append(dict(pid=int(entry.name), paths=sorted(found),
                                   unavailable_fields=unavailable,
                                   command_name=command_name))
    return references


def count_working_files(path):
    files = entries = size = 0
    for base, directories, names in os.walk(path, followlinks=False):
        if Path(base) == path:
            directories[:] = [name for name in directories if name != '.git']
            names = [name for name in names if name != '.git']
        entries += len(directories) + len(names)
        for name in names:
            item = Path(base) / name
            if not item.is_symlink():
                files += 1
                size += item.stat().st_size
    return dict(files=files, entries=entries, bytes=size)


def working_snapshot(path):
    """Detect additions/removals/changes that tar --compare alone cannot see."""
    snapshot = {}
    for base, directories, names in os.walk(path, followlinks=False):
        if Path(base) == path:
            directories[:] = [name for name in directories if name != '.git']
            names = [name for name in names if name != '.git']
        elif '.git' in directories or '.git' in names:
            raise ValueError(f'nested Git repository requires separate archival: {base}')
        for name in directories + names:
            item = Path(base) / name
            state = item.lstat()
            snapshot[str(item.relative_to(path))] = (
                state.st_mode, state.st_size, state.st_mtime_ns,
                os.readlink(item) if item.is_symlink() else None)
    return snapshot


def validate_targets(root, names):
    paths = []
    for name in names:
        relative = Path(name)
        if len(relative.parts) != 2 or relative.parts[0] not in ('checkouts', 'worktrees'):
            raise ValueError(f'not a single declared checkout: {name}')
        if '..' in relative.parts or relative.is_absolute():
            raise ValueError(f'unsafe target: {name}')
        path = root / relative
        if path.is_symlink() or path.resolve() != path or not path.is_dir():
            raise ValueError(f'not an ordinary checkout directory: {path}')
        if not (path / '.git').exists():
            raise ValueError(f'no Git metadata: {path}')
        paths.append(path)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('empty or duplicate target list')
    return paths


def check_activity(root, paths, nodes, python, script):
    if not {'test01', 'test02', 'test03', 'test04', 'gln01'}.issubset(nodes):
        raise ValueError('activity scan must cover the five free compute nodes')
    active = {}
    scheduler = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'cln01',
         "/rmprog/slurm/v22.05.7/bin/squeue -h -u liwei -o '%i|%Z|%o'"],
        check=True, capture_output=True, text=True, timeout=30)
    active['scheduler'] = [line for line in scheduler.stdout.splitlines()
                           if str(root.parent) in line]
    for line in active['scheduler']:
        if any(str(path) in line for path in paths):
            raise RuntimeError(f'checkout referenced by queued/running job: {line}')
    for node in nodes:
        result = subprocess.run(
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node,
             python, str(script), '--processes-only'],
            check=True, capture_output=True, text=True, timeout=30)
        rows = json.loads(result.stdout)
        active[node] = rows
        for row in rows:
            if row.get('unavailable_fields') and row.get('command_name') not in (
                    'sshd', 'starter-suid'):
                raise RuntimeError(f'unresolved process inspection on {node}: {row}')
            # Observed sshd and Singularity privilege-separated supervisors hide
            # cwd/environ. Their ordinary shell/worker processes are still scanned.
            for reference in row['paths']:
                if Path(reference) in paths:
                    raise RuntimeError(f'active checkout {reference} on {node}, pid {row["pid"]}')
    return active


def run(args):
    if args.processes_only:
        print(json.dumps(process_references(ROOT)))
        return
    if not args.manifest or not args.archive:
        raise ValueError('manifest and archive required')
    plan = json.loads(args.manifest.read_text())
    paths = validate_targets(ROOT, plan['targets'])
    if set(paths) & {ROOT / Path(name) for name in plan.get('protected', [])}:
        raise ValueError('target list includes a protected checkout')
    archive = args.archive.absolute()
    if archive.parent != ROOT / 'archives' or not archive.name.endswith('.tar.gz'):
        raise ValueError('archive must be a named runtime/archives/*.tar.gz')
    if archive.exists() or archive.with_suffix('.partial').exists():
        raise FileExistsError(archive)
    report = dict(started=time.strftime('%Y-%m-%dT%H:%M:%S'), targets={},
                  archive=str(archive), git_metadata_retained=True, status='inventoried')
    for path in paths:
        report['targets'][str(path.relative_to(ROOT))] = count_working_files(path)
    report['activity_before'] = check_activity(
        ROOT, paths, plan['nodes'], plan['python'], Path(__file__).absolute())
    report_path = archive.with_suffix('.json')

    def save(status):
        report['status'] = status
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(status=status, target_count=len(paths),
                              report=str(report_path))), flush=True)

    save('inventoried')
    if not args.apply:
        return
    snapshots = {path: working_snapshot(path) for path in paths}
    partial = archive.with_suffix('.partial')
    relative_names = b''.join(str(path.relative_to(ROOT)).encode() + b'\0' for path in paths)
    subprocess.run(['tar', '--create', '--gzip', '--numeric-owner', '--file', str(partial),
                    '--exclude=.git', '--directory', str(ROOT), '--null',
                    '--files-from=-'], input=relative_names, check=True)
    save('packed')
    # Byte comparison with the source also reads/decompresses every archived file.
    subprocess.run(['tar', '--compare', '--gzip', '--numeric-owner', '--file', str(partial),
                    '--directory', str(ROOT)], check=True)
    partial.rename(archive)
    report['archive_bytes'] = archive.stat().st_size
    save('verified')
    report['activity_before_removal'] = check_activity(
        ROOT, paths, plan['nodes'], plan['python'], Path(__file__).absolute())
    validate_targets(ROOT, plan['targets'])
    for path in paths:
        if working_snapshot(path) != snapshots[path]:
            raise RuntimeError(f'checkout changed while archiving: {path}')
    # Leave every .git file/directory at its original path: shared-object clones
    # and Git worktree administrative references therefore retain their targets.
    for path in paths:
        for item in path.iterdir():
            if item.name == '.git':
                continue
            if item.is_symlink() or item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
            else:
                raise ValueError(f'unsupported checkout entry: {item}')
        marker = path / 'ARCHIVED_WORKING_FILES.md'
        marker.write_text(
            f'Working files archived to `{archive}`. Git metadata remains here.\n'
            f'Restore before use:\n\n'
            f'    tar -xzf {archive} -C {ROOT} {path.relative_to(ROOT)}\n')
    report['remaining_working_entries'] = {
        str(path.relative_to(ROOT)): count_working_files(path) for path in paths}
    save('archived_and_working_files_removed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--processes-only', action='store_true')
    run(parser.parse_args())
