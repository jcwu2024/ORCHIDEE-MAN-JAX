"""Compact dataflow evidence for a last-entry dynamic scalar through fusions."""
from __future__ import annotations
from collections import Counter
import io
import re


def collect(compact):
    blocks, entry, current = {}, None, None
    for raw in io.StringIO(compact):
        line=raw.strip()
        if raw and not raw[0].isspace() and line.endswith('{'):
            name=line.split()[1] if line.startswith('ENTRY ') else line.split()[0]
            name=name.split('(')[0].lstrip('%')
            blocks[name]=[];current=name
            if line.startswith('ENTRY '):entry=name
        elif line=='}':current=None
        elif current is not None and ' = ' in line:
            name,rhs=line.removeprefix('ROOT ').split(' = ',1)
            match=re.search(r'\s([a-z][a-z0-9-]*)\(',rhs)
            if match is None:continue
            op=match.group(1);begin=match.end();depth=1;end=begin
            while end<len(rhs) and depth:
                depth += (rhs[end]=='(')-(rhs[end]==')');end+=1
            body=rhs[begin:end-1]
            operands=re.findall(r'(?:^|\s)([%A-Za-z_][\w.%\-]*)(?=\s*(?:,|$))',body)
            called=re.search(r'\bcalls=([%\w.\-]+)',rhs)
            index=re.search(r'\bindex=(\d+)',rhs)
            blocks[current].append({'name':name.lstrip('%'),'op':op,'operands':[x.lstrip('%') for x in operands],
                'parameter':int(body) if op=='parameter' else None,'called':called.group(1).lstrip('%') if called else None,
                'index':int(index.group(1)) if index else None,'rhs':rhs,'root':line.startswith('ROOT ')})
    if entry is None:raise ValueError('compiled HLO has no parsed entry')
    parameters=[x for x in blocks[entry] if x['op']=='parameter']
    scale=max(parameters,key=lambda x:x['parameter'])
    if not scale['rhs'].startswith('f64[]'):
        raise ValueError('last live entry argument is not the dynamic FP64 scalar')
    counts=Counter();examples=[];visited_calls=0

    def present(tag):
        return any(present(v) for v in tag) if isinstance(tag,tuple) else bool(tag)

    def walk(block_name,incoming,chain):
        nonlocal visited_calls
        visited_calls+=1
        if visited_calls>10000:raise RuntimeError('scale traversal exceeded bounded call count')
        tags,paths={},{}
        root_tag=False
        for instruction in blocks[block_name]:
            name,op=instruction['name'],instruction['op']
            incoming_tags=[tags.get(x,False) for x in instruction['operands']]
            active=any(present(t) for t in incoming_tags)
            parent=next((paths.get(x,chain) for x,t in zip(instruction['operands'],incoming_tags) if present(t)),chain)
            path=(parent+[{'computation':block_name,'name':name,'op':op,'rhs':instruction['rhs'][:1200]}])[-12:]
            if op=='parameter':
                tag=incoming.get(instruction['parameter'],False);path=chain+[{'computation':block_name,'name':name,'op':op,'rhs':instruction['rhs'][:1200]}]
            elif op=='fusion' and instruction['called'] in blocks:
                tag=walk(instruction['called'],dict(enumerate(incoming_tags)),path)
            elif op=='tuple':tag=tuple(incoming_tags)
            elif op=='get-tuple-element' and incoming_tags and isinstance(incoming_tags[0],tuple):
                tag=incoming_tags[0][instruction['index']]
            else:tag=active
            tags[name]=tag;paths[name]=path
            if op=='convert' and active and instruction['rhs'].startswith('f16['):
                counts['S_dependent_F16_conversions']+=1
                if len(examples)<12:examples.append(path)
            if op=='divide' and active:
                dtype=instruction['rhs'].split('[',1)[0]
                counts[f'S_dependent_{dtype}_divides']+=1
            if instruction['root']:root_tag=tag
        return root_tag

    walk(entry,{p['parameter']:p['name']==scale['name'] for p in parameters},[])
    return {'entry':entry,'scale_parameter':{'name':scale['name'],'index':scale['parameter'],'rhs':scale['rhs']},
        'counts':dict(counts),'S_to_F16_cast_examples':examples,'visited_computation_calls':visited_calls,
        'claim_limit':'operand dependency through parsed fusion bodies, with tuple-element routing; reports supporting paths only, not exhaustive algebraic sensitivity or device instructions',
        'full_hlo_exported':False}
