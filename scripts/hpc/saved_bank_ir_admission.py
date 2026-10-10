"""Validate a reviewed actual-IR certificate before saved-binary model values.

This module reads bytes/JSON only. It does not import JAX, load an executable,
construct a model or infer proof from numerical values.
"""
import hashlib
import json
from pathlib import Path

FAMILY_NAME='shared-bank-adjoint-saved-binary-T7-600-20261003'
COMPILE_REPORT_SHA256='95357c6d5acaee22fd0440bf629ee9add20aaa4243f16539d175c108bf1bcddf'

def _digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*2**20),b''):value.update(chunk)
    return value.hexdigest()

def validate_saved_IR_certificate(root,expected_certificate_sha256,candidate_sha256,expected_identity):
    family=root/'runtime/outputs/diagnostics'/FAMILY_NAME
    certificate_path=family/'model/actual-IR-admission.json'
    if _digest(certificate_path)!=expected_certificate_sha256:
        raise ValueError('reviewed actual-IR certificate hash differs')
    certificate=json.loads(certificate_path.read_text())
    if certificate['schema_version']!='bank_adjoint_actual_IR_admission_v1' or certificate['passed'] is not True:
        raise ValueError('actual-IR certificate does not admit structural proof')
    if certificate['candidate_sha256']!=candidate_sha256 or certificate['candidate_bytes']!=220794649:
        raise ValueError('certificate is bound to a different candidate binary')
    if certificate['compile_report_sha256']!=COMPILE_REPORT_SHA256:
        raise ValueError('certificate compile-time policy differs')
    original_family=root/'runtime/outputs/diagnostics/shared-bank-adjoint-accumulation-T7-20261002/model'
    if _digest(original_family/'model.json')!=COMPILE_REPORT_SHA256:
        raise ValueError('original pinned compile report changed')
    binary=original_family/'candidate.executable.pkl'
    if binary.stat().st_size!=certificate['candidate_bytes'] or _digest(binary)!=candidate_sha256:
        raise ValueError('actual saved binary differs from reviewed IR candidate')
    report_path=family/'model/model.json'
    if _digest(report_path)!=certificate['capture_model_report_sha256']:
        raise ValueError('actual capture report hash differs')
    captured=json.loads(report_path.read_text())
    if (captured['candidate_sha256']!=candidate_sha256 or captured['compile_evidence']['model_backend_attempts']!=0
        or captured['identity']!=expected_identity or certificate['identity']!=expected_identity
        or captured['binary_input_ABI']!='actual treedef/shapes/dtypes validated'):
        raise ValueError('actual IR capture was not tied to the exact saved binary')
    captures=[event for event in captured['stage_events'] if event['event']=='actual_compact_IR_saved']
    if len(captures)!=1:raise ValueError('expected exact actual runtime IR capture stage')
    capture_modules=captures[0]['actual_IR']['modules']
    sources=certificate['capture_source_sha256']
    required={'probe_shared_bank_saved_binary_T7.py','audit_saved_bank_hlo_cpu.py','shared_bank_adjoint_accumulation.py'}
    if not required.issubset(sources):raise ValueError('missing required actual capture source identities')
    for name,digest in sources.items():
        if Path(name).name!=name or _digest(family/'used-source-snapshots'/name)!=digest:
            raise ValueError('actual IR capture source snapshot differs: '+name)
    modules=certificate['modules']
    if not isinstance(modules,list) or not modules:raise ValueError('missing actual IR modules')
    if len(modules)!=len(capture_modules):raise ValueError('capture and certificate module count differs')
    for index,module in enumerate(modules):
        recorded=capture_modules[index]
        if module['sha256']!=recorded['sha256'] or module['bytes']!=recorded['bytes']:
            raise ValueError('certificate IR differs from actual runtime capture')
        path=family/f'model/candidate.compact-{index}.hlo'
        if path.stat().st_size!=module['bytes'] or _digest(path)!=module['sha256'] or module['proof']['passed'] is not True:
            raise ValueError('actual saved IR size/hash/strict proof differs')
    analysis_path=family/'model/offline-actual-proof.json'
    if _digest(analysis_path)!=certificate['offline_analysis_sha256']:
        raise ValueError('accepted offline strict analysis hash differs')
    analysis=json.loads(analysis_path.read_text())
    if (analysis['passed'] is not True or analysis['modules']!=modules
        or analysis['compile_report_sha256']!=COMPILE_REPORT_SHA256):
        raise ValueError('certificate module proofs differ from accepted analysis')
    analysis_source=family/'offline-analysis-source/shared_bank_adjoint_accumulation.py'
    if _digest(analysis_source)!=analysis['policy_source_sha256']:
        raise ValueError('accepted strict analyzer source differs')
    review_path=family/'independent-proof-admission-review.json'
    if _digest(review_path)!=certificate['review_report_sha256']:
        raise ValueError('independent admission review differs')
    review=json.loads(review_path.read_text())
    if review['status']!='accepted_no_material_findings' or review['findings']:
        raise ValueError('independent proof/admission review has unresolved findings')
    if (review['offline_analysis_sha256']!=certificate['offline_analysis_sha256']
        or review['policy_source_sha256']!=analysis['policy_source_sha256']
        or review['candidate_sha256']!=candidate_sha256
        or review['capture_model_report_sha256']!=certificate['capture_model_report_sha256']):
        raise ValueError('independent review is bound to a different analysis/candidate/capture')
    return {'passed':True,'status':'reviewed_actual_IR_certificate_reused',
        'certificate_sha256':expected_certificate_sha256,'candidate_sha256':candidate_sha256,
        'modules':modules,'new_HLO_print_calls':0,'new_structural_analyzer_calls':0,
        'claim_limit':certificate['claim_limit']}
