"""Upload this sealed candidate once; no job submission or model execution."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PYTHON = __import__('sys').executable
IO = HERE.parent / 'cpu-scalar-pool-pilot-20261014/cluster_io.py'
REMOTE = '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/'
EVIDENCE = REMOTE + 'outputs/research/global_throughput/' + HERE.name
TRANSFER = REMOTE + 'transfer/global-cpu-scalar-pool-v2-20261015'


def io(*args):
    subprocess.run([PYTHON, '-B', '-X', 'utf8', str(IO), *map(str, args)], cwd=ROOT, check=True)


def main():
    global HERE, EVIDENCE, TRANSFER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=['v2', 'v3'], default='v2')
    args = parser.parse_args()
    if args.version == 'v3':
        HERE = HERE / 'v3'
        EVIDENCE += '/v3'
        TRANSFER = TRANSFER.replace('-v2-', '-v3-')
    pin = json.loads((HERE / 'execution-code-pin.json').read_bytes())
    for name, expected in [('scalar-admission.json', pin['scalar_admission']['sha256']),
                           ('PROPOSED_SUBMISSION.sh', pin['proposed_submission']['sha256']),
                           (pin['bundle']['name'], pin['bundle']['sha256'])]:
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected
    io('run', HERE / 'prepare-remote-directories.sh', HERE / 'prepare-remote-directories.log')
    names = ['execution-code-pin.json', 'scalar-admission.json', 'PROPOSED_SUBMISSION.sh',
             'deploy-candidate.sh', 'verify-target.sh', 'run-free-verification.sh']
    for name in names:
        io('put', HERE / name, EVIDENCE + '/' + name)
    io('put', HERE / pin['bundle']['name'], TRANSFER + '/' + pin['bundle']['name'])
    checksum = hashlib.sha256((HERE / 'execution-code-pin.json').read_bytes()).hexdigest()
    deployment = 'bash ' + EVIDENCE + '/deploy-candidate.sh ' + checksum + '\n'
    (HERE / 'dispatch-deployment.sh').write_bytes(deployment.encode())
    io('run', HERE / 'dispatch-deployment.sh', HERE / 'deployment.log')
    io('get', EVIDENCE + '/deployment-receipt.json', HERE / 'deployment-receipt.json')
    io('get', EVIDENCE + '/registry-before.json', HERE / 'registry-before.json')


if __name__ == '__main__':
    main()
