"""CPU-only same-computation admission for the separately proposed compile."""
from __future__ import annotations
import hashlib
import re


def canonical_ir(text):
    """Remove only MLIR locations and alpha-rename SSA/block/symbol identifiers.

    Numeric literals, operations, types, attributes, strings, ordering and
    control-flow relationships are retained. Public @main stays named @main.
    """
    text=re.sub(r'(?m)^\s*#loc\w*\s*=\s*loc\(.*\)\s*\n?', '', text)
    locations=re.compile(r'\bloc\(')
    parts=[];cursor=0
    for match in locations.finditer(text):
        if match.start()<cursor:continue
        depth=1;quoted=False;escaped=False;end=match.end()
        while end<len(text) and depth:
            char=text[end]
            if quoted:
                if escaped:escaped=False
                elif char=='\\':escaped=True
                elif char=='"':quoted=False
            elif char=='"':quoted=True
            elif char=='(':depth+=1
            elif char==')':depth-=1
            end+=1
        if depth:raise ValueError('unclosed MLIR source location')
        # A location marker in a quoted string is not metadata we may remove.
        line=text[text.rfind('\n',0,match.start())+1:match.start()]
        if len(re.findall(r'(?<!\\)"',line))%2:continue
        parts.append(text[cursor:match.start()].rstrip());cursor=end
    parts.append(text[cursor:]);text=''.join(parts)
    names={};counts={'%':0,'@':0,'^':0}
    pattern=re.compile(r'"(?:\\.|[^"\\])*"|[%@^][A-Za-z0-9_.$-]+')
    def rename(match):
        token=match.group()
        if token.startswith('"') or token=='@main':return token
        if token not in names:
            prefix=token[0];names[token]=prefix+'identifier_'+str(counts[prefix]);counts[prefix]+=1
        return names[token]
    return pattern.sub(rename,text).strip()


def graph_identity(candidate,reference):
    raw=candidate==reference
    candidate_sha=hashlib.sha256(candidate.encode()).hexdigest()
    reference_sha=candidate_sha if raw else hashlib.sha256(reference.encode()).hexdigest()
    if raw:
        normalized=True;left_sha=right_sha=None
    else:
        left=canonical_ir(candidate);right=canonical_ir(reference)
        normalized=left==right
        left_sha=hashlib.sha256(left.encode()).hexdigest();right_sha=hashlib.sha256(right.encode()).hexdigest()
    return dict(passed=bool(raw or normalized),raw_bytes_identical=raw,
        normalized_computation_identical=normalized,
        candidate_sha256=candidate_sha,reference_sha256=reference_sha,
        normalization_applied=not raw,candidate_normalized_sha256=left_sha,
        reference_normalized_sha256=right_sha,
        normalization='MLIR source locations and bijective SSA/block/symbol alpha-renaming only',
        operations_types_constants_attributes_order_control_flow_unchanged=normalized)
