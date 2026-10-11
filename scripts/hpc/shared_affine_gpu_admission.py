"""CPU-only cooperative mutex and bounded management-chain free admission.

The declared launch owner dispatches paid only while free dispatch is paused.
This does not constrain the unchanged b676 launcher or arbitrary old copies.
No old checkout, scope or ledger is modified; unknown state fails closed.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import time
import uuid

RUNTIME = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime')
SCOPE_LOCK = RUNTIME / 'locks/T7-summary-affine.shared.lock'
FREE_CHECKOUT = RUNTIME / 'checkouts/T7-batch-numerics-20261011-v1'
FREE_HEAD = 'b676353bbaeb633ba9ccab075aaeadcf9408c7d7'
FREE_OUTPUT = RUNTIME / 'outputs/diagnostics/T7-summary-affine-gpu-20261011'
FREE_SCOPE = FREE_OUTPUT / 'scope.v1.json'
FREE_LEDGER = FREE_OUTPUT / 'T7-summary-affine-gpu-20261011-v1.ledger.json'
MANAGEMENT = dict(launch_owner='daily-affine-execution-owner', paid_dispatch_only=True,
    free_dispatch_paused=True, free_approval_preserved=True,
    known_free_checkout=FREE_CHECKOUT.as_posix(), known_free_head=FREE_HEAD,
    known_free_scope=FREE_SCOPE.as_posix(), known_free_ledger=FREE_LEDGER.as_posix(),
    enforcement='authorized management chain only; unchanged b676 and arbitrary copies do not cooperate')

class AffineScopeLock:
    def __init__(self, path, scope_id):
        self.path = Path(path)
        self.scope_id = scope_id
        self.nonce = uuid.uuid4().hex
        self.acquired = False

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError as error:
            raise ValueError('affine shared mutex already held or stale; no automatic takeover') from error
        self.acquired = True
        (self.path / 'owner.json').write_text(json.dumps(dict(scope_id=self.scope_id,
            nonce=self.nonce, pid=os.getpid(), host=socket.gethostname(), epoch=time.time())), encoding='utf-8')
        return self

    def release(self, *, release_verified):
        if not self.acquired or not release_verified:
            return False
        owner = json.loads((self.path / 'owner.json').read_text(encoding='utf-8'))
        if owner.get('nonce') != self.nonce or owner.get('scope_id') != self.scope_id:
            raise ValueError('shared mutex owner differs; never remove another scope lock')
        (self.path / 'owner.json').unlink()
        self.path.rmdir()
        self.acquired = False
        return True

def read_free_admission(*, checkout=FREE_CHECKOUT, output=FREE_OUTPUT,
                        scope_path=FREE_SCOPE, ledger_path=FREE_LEDGER):
    """Read only the declared entry and its bounded evidence paths.

    The approved free grant was never opened/reserved. Any newly present scope,
    ledger or execution evidence is an existing attempt or unknown state, so
    this paid dispatch stops. No directory scan or old-entry mutation occurs.
    """
    checkout, output = Path(checkout), Path(output)
    scope_path, ledger_path = Path(scope_path), Path(ledger_path)
    if scope_path.parent != output or ledger_path.parent != output:
        raise ValueError('known free scope/ledger paths differ from declared output')
    for path in (checkout, output):
        observed = path.lstat()
        if not stat.S_ISDIR(observed.st_mode) or path.is_symlink():
            raise ValueError('known free entry/output unavailable; state unknown')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, timeout=4, text=True).strip()
    if head != FREE_HEAD:
        raise ValueError('known free deployed entry HEAD differs; state unknown')
    evidence = [scope_path, ledger_path, output / 'result.v1.json',
        output / 'result.v1.outer-timeout.json', output / 'result.v1.runtime-preflight.json',
        output / 'result.v1.supervisor.json', output / 'result.v1.results.npz', output / 'result.v1.partial.npz']
    absent = []
    for path in evidence:
        try:
            path.lstat()
        except FileNotFoundError:
            absent.append(path.as_posix())
        else:
            raise ValueError('known free scope/ledger/execution evidence exists; paid dispatch stopped')
    return dict(status='known_free_unopened_unreserved', read_only=True,
        known_free_head=head, absent_evidence_paths=absent,
        free_approval_preserved=True, free_dispatch_paused_by_owner=True,
        enforcement=MANAGEMENT['enforcement'], checked_epoch=time.time())
