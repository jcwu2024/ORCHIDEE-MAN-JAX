"""Pure CPU saved-IR structural audit; no JAX import, model or executable load."""
import hashlib
import json
from pathlib import Path
import sys
import shared_bank_adjoint_accumulation as policy

def main():
    manifest=json.loads(Path(sys.argv[1]).read_text());rows=[]
    raw_report=Path(manifest['compile_report_path']).read_bytes()
    if hashlib.sha256(raw_report).hexdigest()!=manifest['compile_report_sha256']:
        raise ValueError('original compile report hash differs in CPU audit')
    evidence=json.loads(raw_report)['policy_evidence']
    if evidence!=manifest['policy_evidence']:raise ValueError('manifest bank evidence differs from pinned original')
    for module in manifest['modules']:
        raw=Path(module['path']).read_bytes()
        if len(raw)!=module['bytes'] or hashlib.sha256(raw).hexdigest()!=module['sha256']:
            raise ValueError('actual saved IR size/hash differs')
        rows.append({'path':module['path'],'bytes':len(raw),'sha256':module['sha256'],
            'proof':policy.audit_gpu_hlo(raw.decode(),evidence)})
    report={'passed':bool(rows) and all(row['proof']['passed'] for row in rows),'modules':rows,
        'policy_source_sha256':hashlib.sha256(Path(policy.__file__).read_bytes()).hexdigest(),
        'compile_report_sha256':manifest['compile_report_sha256'],'GPU_calls':0,'model_values':0,
        'claim_limit':'field/deduplicated-bank-shape; not same-shaped occurrence-exhaustive'}
    Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
