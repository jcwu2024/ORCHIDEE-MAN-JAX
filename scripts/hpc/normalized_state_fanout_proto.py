"""Read native optimized HLO protobuf without rendering the full HLO graph.

Wire fields follow OpenXLA hlo.proto/xla_data.proto (official source snapshots
in the preparation record). Unknown operations fail the scoped certificate.
Only metadata/shapes/SSA/call relationships are retained; literals are skipped.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

MARKER = 'normalized_state_fanout_fp64'


def varint(data, position):
    value = shift = 0
    while position < len(data):
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, position
        shift += 7
        if shift > 70:
            raise ValueError('invalid protobuf varint')
    raise ValueError('truncated protobuf varint')


def fields(data):
    data = memoryview(data)
    position = 0
    while position < len(data):
        key, position = varint(data, position)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, position = varint(data, position)
        elif wire == 2:
            size, position = varint(data, position)
            value = data[position:position+size]
            position += size
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[position:position+size]
            position += size
        else:
            raise ValueError('unsupported protobuf wire type')
        if position > len(data) or not number:
            raise ValueError('truncated protobuf field')
        yield number, wire, value


def numbers(wire, value):
    if wire == 0:
        return [value]
    if wire != 2:
        raise ValueError('invalid numeric protobuf field')
    result, position = [], 0
    while position < len(value):
        number, position = varint(value, position)
        result.append(number)
    return result


def shape(data):
    result = dict(dtype=0, dimensions=[], tuples=[], dynamic_dimensions=[])
    for field, wire, value in fields(data):
        if field == 2:
            result['dtype'] = value
        elif field == 3:
            result['dimensions'].extend(numbers(wire, value))
        elif field == 4:
            result['tuples'].append(shape(value))
        elif field == 6:
            result['dynamic_dimensions'].extend(numbers(wire,value))
    return result


def parse_module(data):
    computations, instructions = {}, {}
    entry = None
    for field, wire, value in fields(data):
        if field == 6:
            entry = value
        if field != 3:
            continue
        computation = dict(name='', id=None, root=None, instructions=[])
        rows = []
        for cfield, cwire, cvalue in fields(value):
            if cfield == 1:
                computation['name'] = bytes(cvalue).decode()
            elif cfield == 5:
                computation['id'] = cvalue
            elif cfield == 6:
                computation['root'] = cvalue
            elif cfield == 2:
                row = dict(id=None, name='', opcode='', shape=None, operands=[], calls=[],
                           metadata={}, parameter_number=0, tuple_index=0)
                for ifield, iwire, ivalue in fields(cvalue):
                    if ifield in (1, 2):
                        row['name' if ifield == 1 else 'opcode'] = bytes(ivalue).decode()
                    elif ifield == 3:
                        row['shape'] = shape(ivalue)
                    elif ifield == 7:
                        for mfield, mwire, mvalue in fields(ivalue):
                            if mfield in (2, 3):
                                row['metadata']['op_name' if mfield == 2 else 'source_file'] = bytes(mvalue).decode()
                            elif mfield == 4:
                                row['metadata']['source_line'] = mvalue
                    elif ifield == 35:
                        row['id'] = ivalue
                    elif ifield in (36, 38):
                        row['operands' if ifield == 36 else 'calls'].extend(numbers(iwire, ivalue))
                    elif ifield in (9, 13):
                        row['parameter_number' if ifield == 9 else 'tuple_index'] = ivalue
                rows.append(row)
        if computation['id'] is None or computation['root'] is None:
            raise ValueError('native computation identity/root fields missing')
        for row in rows:
            if row['id'] is None or row['shape'] is None or row['id'] in instructions:
                raise ValueError('native instruction identity/shape missing or duplicate')
            row['computation'] = computation['id']
            computation['instructions'].append(row['id'])
            instructions[row['id']] = row
        computations[computation['id']] = computation
    if entry not in computations:
        raise ValueError('native entry computation field missing')
    return entry, computations, instructions


def extract(data, *, source_file='strong_baseline_architecture.py', source_span=(1100,1355),
            consumer_span=(957,973), normalization_file='input_conditioning.py',
            normalization_span=(12,52),public_state_output_index=None):
    entry, computations, instructions = parse_module(data)
    selected = {}
    def is_vector(row, dtype):
        return row['shape']['dtype'] == dtype and row['shape']['dimensions'] == [3854] and not row['shape']['tuples'] and not any(row['shape']['dynamic_dimensions'])
    state_root=instructions[computations[entry]['root']]
    if public_state_output_index is None:
        public_state_output_index=len(state_root['shape']['tuples'])-1
    output_shapes=state_root['shape']['tuples']
    if not 0<=public_state_output_index<len(output_shapes):
        raise ValueError('public state output tuple index missing')
    state_shape=output_shapes[public_state_output_index]
    if state_shape['dtype']!=12 or state_shape['dimensions']!=[3854] or any(state_shape['dynamic_dimensions']):
        raise ValueError('public initial-state output shape/dtype/static binding differs')
    root_reachable=set(); root_bindings={}; unsupported=[]; visited=set(); active=set()
    # A deliberately small supported numeric/dataflow vocabulary. Branch,
    # asynchronous and side-effect boundaries are not interpreted generically.
    root_arities={'parameter':0,'constant':0,'convert':1,'copy':1,'bitcast':1,
        'reshape':1,'broadcast':1,'transpose':1,'slice':1,'negate':1,'abs':1,
        'add':2,'subtract':2,'multiply':2,'divide':2,'maximum':2,'minimum':2,
        'sqrt':1,'rsqrt':1,'exponential':1,'log':1,'log-plus-one':1,'tanh':1,
        'sine':1,'cosine':1,'concatenate':None}
    def root_dependencies(number,path,bindings,context=()):
        key=(number,tuple(path),context)
        if key in active:
            unsupported.append(dict(id=number,call_context=list(context),reason='active SSA dependency cycle'));return
        if key in visited:return
        active.add(key)
        try:root_step(number,path,bindings,context)
        finally:active.remove(key)
        visited.add(key)
    def root_step(number,path,bindings,context):
        row=instructions[number]; operation=row['opcode']; operands=row['operands']
        identity=(number,context)
        root_reachable.add(identity); root_bindings.setdefault(identity,bindings)
        if operation=='parameter' and row['parameter_number'] in bindings:
            external,outer,outer_context=bindings[row['parameter_number']]
            root_dependencies(external,path,outer,outer_context); return
        if operation=='tuple':
            if path:
                if not 0<=path[0]<len(operands):raise ValueError('native tuple path outside operands')
                root_dependencies(operands[path[0]],path[1:],bindings,context)
            else:
                for operand in operands:root_dependencies(operand,(),bindings,context)
            return
        if operation=='get-tuple-element':
            if len(operands)!=1:raise ValueError('native GTE arity differs')
            root_dependencies(operands[0],(row['tuple_index'],)+tuple(path),bindings,context); return
        if operation in ('fusion','call') and len(row['calls'])==1:
            if number in context:
                unsupported.append(dict(id=number,reason='recursive call context'));return
            callee=computations[row['calls'][0]]
            bound={index:(operand,bindings,context) for index,operand in enumerate(operands)}
            root_dependencies(callee['root'],path,bound,context+(number,)); return
        if operation=='while' and len(operands)==1 and len(row['calls'])==2:
            # Native while body is the called computation returning the exact
            # carry shape. Inspect its selected carry root, plus initial carry.
            bodies=[computations[c] for c in row['calls'] if instructions[computations[c]['root']]['shape']==row['shape']]
            if len(bodies)!=1:
                unsupported.append(dict(id=number,reason='unknown while body/carry binding')); return
            root_dependencies(operands[0],path,bindings,context)
            root_dependencies(bodies[0]['root'],path,{0:(operands[0],bindings,context)},context+(number,)); return
        if row['calls'] or path or operation not in root_arities:
            unsupported.append(dict(id=number,opcode=operation,reason='unsupported called/tuple result boundary'))
            return
        arity=root_arities[operation]
        if (arity is not None and len(operands)!=arity) or (arity is None and not operands):
            unsupported.append(dict(id=number,opcode=operation,reason='unsupported numeric operand arity'));return
        for operand in operands:root_dependencies(operand,(),bindings,context)
    root_dependencies(state_root['id'],(public_state_output_index,),{})
    def sourced(row,span,filename_bound=source_file):
        metadata=row['metadata']; filename=metadata.get('source_file','').replace('\\','/').rsplit('/',1)[-1]
        return filename==filename_bound and span[0]<=metadata.get('source_line',-1)<=span[1]
    def resolve(number, bindings, stack, context):
        identity=(number,context)
        if identity in stack or number not in instructions:
            return None
        row = instructions[number]
        selected[number] = row
        stack = stack | {identity}
        operation = row['opcode']
        operands = row['operands']
        if operation == 'parameter' and row['parameter_number'] in bindings:
            external, outer, outer_context = bindings[row['parameter_number']]
            return resolve(external, outer, stack, outer_context)
        if operation in ('fusion', 'call') and len(row['calls']) == 1:
            callee = computations[row['calls'][0]]
            bound = {index:(operand, bindings, context) for index, operand in enumerate(operands)}
            return resolve(callee['root'], bound, stack,context+(number,))
        if operation in ('copy', 'bitcast', 'reshape') and len(operands) == 1 and is_vector(row, 12):
            return resolve(operands[0], bindings, stack,context)
        if operation == 'convert' and len(operands) == 1 and is_vector(row, 12) and is_vector(instructions[operands[0]], 11):
            selected[operands[0]] = instructions[operands[0]]
            return dict(partials=[identity], adds=[])
        if operation == 'add' and len(operands) == 2 and is_vector(row, 12):
            children = [resolve(operand, bindings, stack,context) for operand in operands]
            if any(child is None for child in children):
                return None
            return dict(partials=children[0]['partials']+children[1]['partials'],
                        adds=children[0]['adds']+children[1]['adds']+[identity])
        return None
    targets = [(instructions[number],context) for number,context in root_reachable if
        instructions[number]['opcode']=='convert' and is_vector(instructions[number],11)
        and len(instructions[number]['operands'])==1
        and is_vector(instructions[instructions[number]['operands'][0]],12)
        and MARKER in instructions[number]['metadata'].get('op_name','')]
    certificates = []
    for target,context in targets:
        selected[target['id']] = target
        found = resolve(target['operands'][0], root_bindings.get((target['id'],context),{}), set(),context)
        partials=[] if found is None else found['partials']
        adds=[] if found is None else found['adds']
        consumer_labels=[]
        for number,_partial_context in partials:
            match=re.search(r'normalized_state_consumer_(field|axis_[0-6])(?:/|\)|$)',instructions[number]['metadata'].get('op_name',''))
            consumer_labels.append(None if match is None else match.group(1))
        expected_labels=['field']+[f'axis_{i}' for i in range(7)]
        # This deliberately bounded immediate-use assertion does not invent a
        # continuation when optimization loses/splits normalization metadata.
        normalization_uses=[row for row in instructions.values()
            if target['id'] in row['operands'] and row['computation']==target['computation']
            and (row['id'],context) in root_reachable
            and (is_vector(row,11) or is_vector(row,12))
            and 'transpose' in row['metadata'].get('op_name','')
            and sourced(row,normalization_span,normalization_file)]
        valid=(found is not None and len(partials)==len(set(partials))==8 and len(adds)==len(set(adds))==7
            and sourced(target,source_span) and all(sourced(instructions[n],consumer_span) for n,_ in partials)
            and sorted(label for label in consumer_labels if label is not None)==sorted(expected_labels)
            and bool(normalization_uses))
        for row in normalization_uses:selected[row['id']]=row
        certificates.append(dict(narrow=target['id'], call_context=list(context),computation=target['computation'],
            metadata=target['metadata'], incoming_f32_partial_converts=[] if found is None else found['partials'],
            f64_adds=adds,consumer_labels=consumer_labels,public_state_output_root_connected=True,
            original_normalization_transpose_immediate_uses=[dict(id=row['id'],
                call_context=list(context),metadata=row['metadata']) for row in normalization_uses],
            source_bound=sourced(target,source_span),passed=valid))
    # Add the target definition closure without traversing unrelated physical
    # branches or printing constants; preserve called root/parameter identities.
    pending = list(selected)
    while pending:
        number = pending.pop()
        row = instructions[number]
        for dependency in row['operands']:
            producer = instructions[dependency]
            if dependency not in selected and (is_vector(producer,11) or is_vector(producer,12)):
                selected[dependency] = producer
                if producer['opcode'] in ('convert','add','copy','bitcast','reshape'):
                    pending.append(dependency)
        for called in row['calls']:
            for dependency in computations[called]['instructions']:
                producer = instructions[dependency]
                if producer['opcode']=='parameter' or dependency==computations[called]['root']:
                    selected.setdefault(dependency,producer)
    verified=bool(certificates) and all(c['passed'] for c in certificates) and not unsupported
    return dict(status='verified_scoped_native_fanin' if verified else 'local_pattern_only_uncertified',
        passed=verified,
        proto_bytes=len(data), proto_sha256=hashlib.sha256(data).hexdigest(),
        entry=entry,public_state_output_index=public_state_output_index,
        public_state_root_dependency_instructions=len(root_reachable),total_instructions=len(instructions),
        unsupported_root_boundaries=unsupported,
        supported_root_numeric_opcodes=sorted(root_arities),
        identity_semantics='instruction ID plus complete enclosing call context',
        definition_frontier='Scoped vector definitions only; other F32 producers/external dependencies remain a frontier, not whole-public-graph closure.',
        certificates=certificates, relevant_definitions=list(selected.values()),
        relevant_computations=[{k:v for k,v in computations[i].items() if k!='instructions'}
            for i in sorted({row['computation'] for row in selected.values()})],
        limitations=['Source/marker/shape scoped positive certificate; not occurrence exhaustive.',
            'Unknown opcodes, select/async/side-effect/tuple/custom-call/command-buffer boundaries and active SSA cycles fail closed.',
            'Normalization reverse binding requires an immediate source-bound vector transpose use; lost/fused-away/call-split continuation is uncertified.',
            'Does not establish public-loss numerical admission or gradient causation.'],
        full_HLO_text_rendered=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('proto',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--public-state-output-index',type=int,
        help='bind the recorded flattened public-state output leaf (default last leaf)')
    args=parser.parse_args()
    result=extract(args.proto.read_bytes(),public_state_output_index=args.public_state_output_index)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','passed','proto_bytes','total_instructions','full_HLO_text_rendered')}))


if __name__=='__main__':
    main()
