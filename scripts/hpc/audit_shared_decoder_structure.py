"""Audit the CURRENT expert128/source-reuse decoder using metadata and algebra.

No neural initialization, parameter allocation, Teacher/weight loading, model
forward, AD, GPU, architecture implementation or performance measurement.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
DEFAULT_OUTPUT=ROOT/'outputs/diagnostics/shared-decoder-structure-20261004'

def linear_parameters(inputs,outputs):return (inputs+1)*outputs
def block_parameters(width,hidden):return 4*width*width+2*width*hidden+hidden+9*width
def padded_rows(rows,chunk):
    if not rows:return dict(rows=0,padded_rows=0,padding_rows=0,chunk=0,chunks=0)
    chunk=min(rows,chunk);count=math.ceil(rows/chunk)
    return dict(rows=rows,padded_rows=chunk*count,padding_rows=chunk*count-rows,chunk=chunk,chunks=count)

def source_ref(relative,symbol):
    path=ROOT/relative;tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    matches=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==symbol]
    if len(matches)!=1:raise ValueError('Source symbol ambiguous/missing: '+symbol)
    node=matches[0]
    return dict(file=relative,symbol=symbol,start_line=node.lineno,end_line=node.end_lineno)

def current_layout():
    # This existing audited helper replaces both fixture neural initializers
    # with empty dictionaries; it returns only static schema/layout metadata.
    from scripts.hpc.audit_shared_decoder_cost import _load_static_layout
    from research.daily_coarse_graining.daily_operator.shared_transformer import _decoder_dynamic_plan
    static=_load_static_layout();plan=_decoder_dynamic_plan(schema=static['schema'],layout=static['layout'])
    old=json.loads((ROOT/'outputs/shared-decoder-cost-audit-20260923.json').read_text())
    packing=json.loads((ROOT/'outputs/shared-decoder-packing-audit-20260923.json').read_text())
    macro=json.loads((ROOT/'outputs/shared-decoder-macro-layout-20260925.json').read_text())
    if static['fields']!=old['fields']:raise ValueError('Saved field audit differs from current source/layout')
    rows=static['fields'];source_macro={f['field_index']:f for f in macro['fields']};fields=[];experts=[]
    for index,row in enumerate(rows):
        record=dict(row,field_index=index,field_axis_bank_rows=int(plan.field_axis_bank_indices[index].size),
            field_local_bank_rows=int(plan.field_local_bank_indices[index].size),axis_slots=row['dynamic_tokens_per_query']-1)
        if not row['uses_carbon_bank']:
            saved=source_macro[index]
            for key,value in (('field_name',row['field']),('expert_index',row['expert_index']),('eligible_rows',row['eligible_coordinates']),('axis_slots',record['axis_slots']),('axis_bank_size',record['field_axis_bank_rows']),('local_bank_size',record['field_local_bank_rows'])):
                if saved[key]!=value:raise ValueError('Saved macro field metadata differs: '+row['field']+'/'+key)
            if not np.array_equal(saved['axis_indices'],plan.field_axis_indices[index]) or not np.array_equal(saved['local_indices'],plan.field_local_indices[index]):raise ValueError('Saved projection identities differ')
        fields.append(record)
    for expert,name in enumerate(static['layout'].expert_names):
        group=[r for r in fields if r['expert_index']==expert and not r['uses_carbon_bank']]
        size=sum(r['eligible_coordinates'] for r in group);slots=max((r['axis_slots'] for r in group),default=0)
        experts.append(dict(expert_index=expert,expert=name,fields=len(group),max_axis_slots=slots,
            dynamic_slots=slots+1,attention_bank_slots=20+slots+1+1,
            field_axis_bank_rows=sum(r['field_axis_bank_rows'] for r in group),
            field_local_bank_rows=sum(r['field_local_bank_rows'] for r in group),
            **padded_rows(size,128)))
    gpp=[r for r in fields if r['uses_carbon_bank']]
    if len(gpp)!=1 or gpp[0]['field']!='daily.fast_gpp':raise ValueError('GPP routing differs')
    gpp=dict(gpp[0],**{'execution':padded_rows(gpp[0]['eligible_coordinates'],64)})
    axis_rows=sum(r['field_axis_bank_rows'] for r in fields);local_rows=sum(r['field_local_bank_rows'] for r in fields)
    if (axis_rows,local_rows)!=(packing['shape_evidence']['field_local_axis_bank_entries'],packing['shape_evidence']['field_local_state_bank_entries']):raise ValueError('Dynamic field-bank counts differ')
    counts=dict(fields=len(fields),full_coordinates=sum(r['full_coordinates'] for r in fields),
        static_eligible_coordinates=sum(r['eligible_coordinates'] for r in fields),
        padded_coordinate_rows=sum(e['padded_rows'] for e in experts)+gpp['execution']['padded_rows'],
        total_chunk_iterations=sum(e['chunks'] for e in experts)+gpp['execution']['chunks'],
        non_GPP_fields=len(macro['fields']),non_GPP_rows=macro['eligible_rows'],
        field_axis_bank_rows=axis_rows,field_local_bank_rows=local_rows,
        global_axis_identities=int(plan.axis_expert_index.size),global_local_state_identities=int(plan.local_state_indices.size))
    counts['padding_rows']=counts['padded_coordinate_rows']-counts['static_eligible_coordinates']
    if (counts['fields'],counts['full_coordinates'],counts['static_eligible_coordinates'],counts['padded_coordinate_rows'],counts['total_chunk_iterations'])!=(161,39006,8550,8716,72):raise ValueError('Current expert128 layout changed')
    return dict(counts=counts,experts=experts,GPP=gpp,fields=fields,
        axis_names=list(static['layout'].axis_names),axis_extents=list(map(int,static['layout'].axis_extents)),
        axis_offsets=list(map(int,static['layout'].axis_offsets)),active_PFT_indices=list(static['layout'].active_pft_indices)),old,packing

def forward_mac_formula(layout,*,backbone_width,backbone_ffn,backbone_blocks,decoder_width,decoder_ffn,decoder_blocks,
                        input_width,readout_width,source_rows,axis_rows_per_expert,family_rows,carbon_rows,PFT_rows):
    """Pure algebra for declared shapes; changing arguments implements no model.

    This keeps the existing expert128/current field-local projection identities.
    A future structure with changed banks/packing must pass its own verified layout.
    """
    E=len(layout['experts']);R=layout['counts']['padded_coordinate_rows'];C=layout['counts']['static_eligible_coordinates']
    W=backbone_width;D=decoder_width;Lb=backbone_blocks;Ld=decoder_blocks
    backbone=dict(input_projection=(source_rows+E*axis_rows_per_expert+PFT_rows)*input_width*W,
        source_KV=Lb*2*source_rows*W*W,expert_axis_KV=Lb*2*E*axis_rows_per_expert*W*W,
        process_query_output=Lb*2*E*W*W,process_FFN=Lb*2*E*W*backbone_ffn,
        cross_attention_score_value=Lb*2*E*(source_rows+axis_rows_per_expert)*W)
    bridge=dict(context_bridge=(E*(family_rows+axis_rows_per_expert+1)+PFT_rows)*W*D)
    grouped_key_pairs=sum(e['padded_rows']*(family_rows+e['max_axis_slots']+1+1) for e in layout['experts'])
    gpp=layout['GPP'];gpp_pairs=gpp['execution']['padded_rows']*(family_rows+gpp['axis_slots']+1+1+carbon_rows)
    decoder=dict(family_KV=Ld*2*E*family_rows*D*D,process_KV=Ld*2*E*D*D,
        raw_carbon_KV=Ld*2*carbon_rows*D*D,
        field_axis_KV=Ld*2*layout['counts']['field_axis_bank_rows']*D*D,
        field_local_KV=Ld*2*layout['counts']['field_local_bank_rows']*D*D,
        coordinate_query_output=Ld*2*R*D*D,coordinate_FFN=Ld*2*R*D*decoder_ffn,
        cross_attention_score_value=Ld*2*(grouped_key_pairs+gpp_pairs)*D,
        readout=R*(D*readout_width+readout_width),
        coordinate_feature_local_affine=C*2*D,
        field_dynamic_local_affine=layout['counts']['field_local_bank_rows']*2*D,
        raw_carbon_local_affine=carbon_rows*2*D)
    sections={'shared_input_and_backbone':backbone,'context_bridge':bridge,'coordinate_decoder':decoder}
    for section in sections.values():section['total']=sum(section.values())
    return dict(sections=sections,total=sum(v['total'] for v in sections.values()),
        grouped_attention_query_key_pairs=grouped_key_pairs,GPP_attention_query_key_pairs=gpp_pairs,
        definition='one multiply-add is one MAC; forward source-visible dense/dot work only')

def parameter_formula(metadata,configuration,*,reader_width=None,latent_slots=32):
    """Exact initializer algebra, including biases and each LayerNorm scale/bias."""
    m=metadata;c=configuration;U=c['input_width'];V=256;W=c['backbone_width'];D=c['decoder_width'];R=c['readout_width'];E=m['experts']
    tokenizer=dict(forcing_input=linear_parameters(5,V),forcing_blocks=2*block_parameters(V,4*V),
        axial_blocks=block_parameters(U,4*U),forcing_to_model=linear_parameters(V,U),summary_input=linear_parameters(4,U),
        pft_input=linear_parameters(m['pft_input_width'],U),process_static_input=linear_parameters(1,U),calendar_input=linear_parameters(4,U),
        source_family_embedding=m['source_families']*U,forcing_field_embedding=m['forcing_fields']*V,
        forcing_position_embedding=m['forcing_positions']*V,state_field_embedding=m['state_fields']*U,
        discrete_state_field_embedding=m['discrete_fields']*U,static_field_embedding=m['static_fields']*U,annual_field_embedding=m['annual_fields']*U,
        process_static_field_embedding=m['process_static_fields']*U,process_static_position_embedding=m['process_positions']*U,
        expert_embedding=E*U,axis_role_embedding=len(m['axis_names'])*U,axis_position_embedding=m['nonpft_axis_extent_sum']*U)
    shared=dict(input_projection=linear_parameters(U,W),process_query=E*W,process_embedding=E*W,
        backbone_blocks=c['backbone_blocks']*block_parameters(W,c['backbone_ffn']),
        field_embedding=m['fields']*D,label_embedding=m['labels']*D,output_axis_role_embedding=len(m['axis_names'])*D,
        output_axis_position_embedding=m['nonpft_axis_extent_sum']*D,axis_slot_embedding=m['max_output_rank']*D,
        local_value_input=linear_parameters(2,D),
        carbon=(m['carbon_fields']+len(m['carbon_axis_names'])+sum(m['carbon_axis_extents']))*D,
        decoder_bridge=linear_parameters(W,D) if W!=D else 0)
    if reader_width is None:
        shared.update(coordinate_blocks=c['decoder_blocks']*block_parameters(D,c['decoder_ffn']),
            readout_first=linear_parameters(D,R),readout_second=linear_parameters(R,1))
    else:
        u=reader_width
        # Two attentions (cross + latent self), FFN, and THREE normalizations.
        latent_block=8*D*D+2*D*c['decoder_ffn']+c['decoder_ffn']+15*D
        shared.update(shared_latent_slot_embedding=latent_slots*D,latent_blocks=2*latent_block,
            reader_input_affine=linear_parameters(D,u),reader_blocks=2*block_parameters(u,4*u),
            readout_first=linear_parameters(u,R),readout_second=linear_parameters(R,1))
    return dict(groups={'tokenizer':tokenizer,'shared_transformer':shared},
        group_totals={'tokenizer':sum(tokenizer.values()),'shared_transformer':sum(shared.values())},
        total=sum(tokenizer.values())+sum(shared.values()))

def tokenizer_mac_formula(metadata,input_width=384,forcing_width=256):
    """Include tokenizer dense/self-attention dots; pooling arithmetic excluded."""
    m=metadata;E=m['experts'];F=m['forcing_positions']*m['forcing_fields'];A=len(m['axis_names']);K=max(m['axis_extents']);U=input_width;V=forcing_width
    axial_rows=E*A*K
    terms=dict(forcing_input=F*5*V,forcing_self_QKVO=2*4*F*V*V,forcing_FFN=2*2*F*V*(4*V),
        forcing_self_score_value=2*2*F*F*V,forcing_to_input_projection=F*V*U,
        source_summary_input=(m['state_fields']+m['discrete_fields']+m['static_fields']+m['annual_fields'])*4*U,
        expert_axis_four_summary_inputs=4*E*sum(m['axis_extents'])*4*U,
        process_static_input=sum(m['process_static_counts'].values())*U,
        PFT_input=2*m['pft_input_width']*U,calendar_input=4*U,
        axial_self_QKVO=4*axial_rows*U*U,axial_FFN=2*axial_rows*U*(4*U),
        axial_self_score_value=2*E*A*K*K*U)
    terms['total']=sum(terms.values())
    return dict(terms=terms,forcing_tokens=F,forcing_self_score_slots=F*F,
        axial_tokens_before_padding=E*sum(m['axis_extents']),axial_tokens_after_padding=axial_rows,
        axial_padding_rows=axial_rows-E*sum(m['axis_extents']),axial_self_score_slots=E*A*K*K,
        mask_note='Source-visible dense scores include masked family/invalid/padding slots; no runtime mask population measured.')

def candidate_A_mac_formula(layout,configuration,*,latent_slots=32,reader_width=64):
    """Only the parent-selected architecture A algebra; no executable model."""
    c=configuration;D=c['decoder_width'];u=reader_width;M=latent_slots;E=len(layout['experts']);L=2;R=layout['counts']['padded_coordinate_rows'];C=layout['counts']['static_eligible_coordinates']
    F=c['family_rows'];raw=c['carbon_rows'];axis=layout['counts']['field_axis_bank_rows'];local=layout['counts']['field_local_bank_rows']
    existing=forward_mac_formula(layout,**c)['sections']
    retained=dict(coordinate_feature_local_affine=C*2*D,field_dynamic_local_affine=local*2*D,raw_carbon_local_affine=raw*2*D)
    latent=dict(common_family_process_KV=L*2*E*(F+1)*D*D,cross_query_output=L*2*E*M*D*D,
        cross_score_value=L*2*E*M*(F+1)*D,self_QKVO=L*4*E*M*D*D,
        self_score_value=L*2*E*M*M*D,latent_FFN=L*2*E*M*D*c['decoder_ffn'])
    compression=dict(query=C*D*u,latent=E*M*D*u,process=E*D*u,field_axis=axis*D*u,field_local=local*D*u,raw_carbon=raw*D*u)
    grouped_pairs=sum(e['padded_rows']*(M+1+e['max_axis_slots']+1) for e in layout['experts'])
    gpp=layout['GPP'];gpp_pairs=gpp['execution']['padded_rows']*(M+1+gpp['axis_slots']+1+raw)
    reader=dict(latent_KV=L*2*E*M*u*u,process_KV=L*2*E*u*u,field_axis_KV=L*2*axis*u*u,
        field_local_KV=L*2*local*u*u,raw_carbon_GPP_only_KV=L*2*raw*u*u,
        coordinate_query_output=L*2*R*u*u,coordinate_FFN=L*2*R*u*(4*u),
        cross_score_value=L*2*(grouped_pairs+gpp_pairs)*u,readout=R*(u*c['readout_width']+c['readout_width']))
    sections=dict(shared_input_and_backbone=dict(existing['shared_input_and_backbone']),context_bridge=dict(existing['context_bridge']),
        retained_coordinate_bank_encoding=retained,latent=latent,shared_reader_input_compression=compression,reader=reader)
    for section in sections.values():section['total']=sum(value for key,value in section.items() if key!='total')
    return dict(sections=sections,total=sum(section['total'] for section in sections.values()),
        assumptions=dict(latent_slots=M,latent_width=D,latent_blocks=2,latent_self_attention=True,reader_width=u,reader_heads=8,reader_blocks=2,reader_ffn_width=4*u,
            latent_bank='expert-common lawful family/process ONLY; no query-private axis/local or raw carbon',
            reader_bank='same-expert latent + process + original query-private axis/local; GPP alone also sees masked raw carbon',
            compression='one shared affine; once per query or retained field-bank identity within each invocation; no cross-day/expert reuse',
            parameters_shared_across_experts=True,slot_embedding_shared_across_experts=True))

def no_padding_layout(layout):
    result=json.loads(json.dumps(layout))
    for expert in result['experts']:expert['padded_rows']=expert['rows'];expert['padding_rows']=0
    result['GPP']['execution']['padded_rows']=result['GPP']['execution']['rows']
    result['counts']['padded_coordinate_rows']=result['counts']['static_eligible_coordinates'];result['counts']['padding_rows']=0
    return result

def logical_activations(layout,c,*,batch=4,latent_slots=32,reader_width=64):
    """Single forward layer logical volumes, NEVER peak/live AD estimates."""
    B=batch;R=layout['counts']['padded_coordinate_rows'];E=len(layout['experts']);D=c['decoder_width'];u=reader_width;M=latent_slots;K=128;raw=c['carbon_rows']
    def volume(shape):return dict(shape=shape,float32_bytes=math.prod(shape)*4)
    return dict(note='Hypothetical all-row volumes per layer; actual expert chunks map sequentially. Multiple layers/days are not assumed simultaneously live.',
        current_query_all_rows=volume([B,R,1,D]),current_FFN_all_rows_one_layer=volume([B,R,c['decoder_ffn']]),
        current_FFN_one_expert5_chunk=volume([B,K,c['decoder_ffn']]),
        retained_raw_carbon_encoding=volume([B,raw,D]),retained_coordinate_features=volume([B,layout['counts']['static_eligible_coordinates'],D]),
        candidate_reader_query_all_rows=volume([B,R,1,u]),candidate_reader_FFN_all_rows_one_layer=volume([B,R,4*u]),
        candidate_reader_FFN_one_expert5_chunk=volume([B,K,4*u]),candidate_latents=volume([B,E,M,D]),
        candidate_latent_FFN_one_layer=volume([B,E,M,c['decoder_ffn']]),
        candidate_latent_cross_scores_one_layer=volume([B,E,8,M,c['family_rows']+1]),
        candidate_latent_self_scores_one_layer=volume([B,E,8,M,M]),
        candidate_raw_carbon_reader_encoding=volume([B,raw,u]),
        scaling='All listed volumes linear in B. Query/reader volumes linear in R; latent dense terms linear in E*M, latent self scores quadratic in M. Saving H uncheckpointed steps would multiply each per-day logical volume by H; actual checkpoint/AD liveness is unmeasured.')

def write_markdown(output,result):
    r=result;c=r['configuration'];layout=r['current_layout'];totals=r['counted_neural_dense_dot_totals'];meta=r['source_token_metadata']
    lines=['# Current decoder and architecture A: source-shape accounting','',
        'Completed local metadata/formula audit only. Current architecture is 64,884,481 parameters; architecture A is a specified proposal with 66,123,713 parameters. No model implementation, neural weight allocation, forward, training, GPU or remote compute occurred. A is not admitted or adopted.','',
        'One multiply-add is one MAC. Counts include source-visible dense masked score/value dots and tokenizer dense/self-attention. They exclude pooling/reductions, biases, embedding additions, LayerNorm/L2 normalization, GELU/softmax, masks/gathers/scatters, physical processes, losses, AD/recompute, optimizer, compilation and launch effects. The counted ratio is not a GPU speed or full-update prediction.','',
        '## Current layout and axes','',
        'Metadata is regenerated from the existing synthetic static fixture with both neural initializers stubbed. Current fields/dynamic identities match the saved layout assets. This is not a new prepared real-input or runtime-mask check. PFT [0,13] is the declared layout only; dynamic activity/defined masks can remove values, without removing the dense slots from this source-shape count.','',
        '| Boundary | Shape per sample (outer sample batch B omitted) | Attention/role |',
        '|---|---|---|',
        '| Native forcing | [5 records,9 fields] → [45,256] | One predecessor at -6h plus four native records at 0/6/12/18h; 2 blocks, eight heads of 32; 45×45 self-attention dense scores masked by same source family, not nine isolated 5×5 sequences |',
        '| Tokenizer axial | [7 experts,18 axis roles,32 padded positions,384] | One shared self-attention/FFN block separately within each expert/role; 980 pre-padding vs 4032 padded rows; no cross-role/expert mixing |',
        '| Source stream | [393,768] | 45 forcing+257 state+8 discrete+30 static+1 annual+49 process-static+2 PFT+1 calendar |',
        '| Backbone process query | [7,1,768] → [7,768] | 8 cross-attention blocks; query length1; source K/V 393 reused once per block, axis K/V 7×140 retained separately; eight heads of 96 |',
        '| Expert axis bank | [7,140,768] → [7,140,384] | 18 roles with explicit offsets; bridge is token local, retains masks; not a 140-token joint axial self-attention sequence |',
        '| Family/process bank | [7,20,384] and [7,1,384] | Family pooling after expert ownership masking; 20 padded slots, fewer lawful/valid families by expert |',
        '| Compact PFT | [2,384] | Full indices0/13; PFT role has no learned positional embedding |',
        '| Raw-carbon bank | [2386,384] |107 state fields; only GPP directly reads global or query-owner-PFT defined entries; non-GPP has no raw-carbon segment |',
        '| Field-local dynamic banks | Σfield axis 1380 and local 3249 rows, width 384 | Existing per-field distinct identity encoding/KV reuse; global identity counts 339/1798 are not the current projected-row count |',
        '| Coordinate features | [8550,384] before chunk padding | Process+field+label+physical axis role/slot/position+owner PFT+local value/defined. Equal shapes do not make different inputs identical |',
        '| Coordinate chunk query | [chunk,1,384], chunk≤128 |2 cross-attention blocks, eight heads of 48, FFN 1536; no coordinate×coordinate self-attention |',
        '| Readout/scatter |384→256→1, then full 39006 registry | Shared weights; compute 8716 padded rows and restore 8550 static eligible values to full shapes |','',
        '| Expert (non-GPP) | Fields | Eligible rows | Computed rows | Padding | Chunks | Max private axis slots | Current bank slots |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for e in layout['experts']:lines.append('| '+' | '.join(str(e[k]) for k in ('expert','fields','rows','padded_rows','padding_rows','chunks','max_axis_slots','attention_bank_slots'))+' |')
    lines.extend(['', 'GPP is a separate field with 2 eligible/computed rows and one chunk (max64), plus 2386 raw-carbon slots. Non-GPP totals 8548→8714 rows; all fields 8550→8716, 166 padding, 72 mapped iterations. Expert 5 alone 7542→59×128=7552, 10 padding. The old 9046 field-local chunk 64 count is not this path. Axis-slot padding inside each expert also enlarges score/value dots: original per-field private axes remain masked.','',
        'Axis extents: '+', '.join(f'{name}={extent}' for name,extent in zip(layout['axis_names'],layout['axis_extents']))+'.','',
        'Lawful family slot counts by expert (runtime validity can be smaller): '+', '.join(f'{name}={len(allowed)}/20' for name,allowed in meta['static_allowed_families'].items())+'. Projection/dot MACs count the full source-visible slots.','',
        '## Exact parameter arithmetic','',
        'Linear(a,b)=(a+1)b. One attention/FFN block with two LayerNorms has 4w²+2wh+h+9w; h=4w gives12w²+13w. A latent block has cross+ self attention, FFN and three norms: 8D²+2Dh+h+15D. Each shared latent slot table is 32×384 total, shared across experts. Head count changes score geometry, not these matrix parameter counts.','',
        '| Group | Current parameters | Architecture A parameters |',
        '|---|---:|---:|'] )
    for section in ('tokenizer','shared_transformer'):
        current=r['parameter_count']['groups'][section];candidate=r['selected_architecture_A_parameter_count']['groups'][section]
        for name in sorted(set(current)|set(candidate)):lines.append(f'| {section}.{name} | {current.get(name,0):,} | {candidate.get(name,0):,} |')
    lines.extend([f'| **Total** | **{r["parameter_count"]["total"]:,}** | **{r["selected_architecture_A_parameter_count"]["total"]:,}** |','',
        'The current algebra matches every group from the actual source initializer under symbolic shape stubs: 300 numeric leaves and 2 static None. Full shape/path evidence is `parameter-shapes.json`. Architecture A adds 1,239,232 parameters (~1.91%); it preserves wide encoding but replaces coordinate-conditioned nonlinear width 384 with reader 64, so neither parameter count nor preserved input paths proves unchanged expressivity.','',
        '## Forward MAC ledger','',
        '| Source-visible term | Current MACs/sample-day | Architecture A MACs/sample-day |',
        '|---|---:|---:|'] )
    current_sections=r['forward_macs_excluding_tokenizer']['sections'];candidate_sections=r['selected_architecture_A_MACs_excluding_tokenizer']['sections']
    for name,terms in current_sections.items():
        for term,value in terms.items():
            if term=='total':continue
            candidate=candidate_sections.get(name,{}).get(term)
            if candidate is None:
                candidate=candidate_sections.get('retained_coordinate_bank_encoding',{}).get(term)
            lines.append(f'| current: {name}.{term} | {value:,} | '+('same' if candidate==value else 'replaced; see A terms')+' |')
    for name,terms in candidate_sections.items():
        if name in ('shared_input_and_backbone','context_bridge'):continue
        for term,value in terms.items():
            if term!='total':lines.append(f'| A: {name}.{term} | — | {value:,} |')
    for term,value in r['tokenizer_dense_and_self_attention_MACs']['terms'].items():
        if term!='total':lines.append(f'| unchanged tokenizer: {term} | {value:,} | {value:,} |')
    lines.extend([f'| **Counted neural dense/dot total** | **{totals["current"]:,}** | **{totals["selected_architecture_A"]:,}** |',
        f'| No coordinate-padding bound, R8550 | {totals["current_no_coordinate_padding"]:,} | {totals["selected_architecture_A_no_coordinate_padding"]:,} |','',
        'No-padding bound removes only coordinate-row padding. It retains expert-max private-axis slot padding, tokenizer axial padding, all current input/bank construction and their source-shape dots. It is not an implemented execution or a global operation lower bound.','',
        'Current dominant coordinate terms: Q/output=2LdRD²; FFN=2LdRDh. Dynamic K/V=2Ld(Afield+Lfield)D², already reused. Source-reuse backbone K/V=2Lb(S+EA)W²; its query/output=2LbEW² and process FFN=2LbEWfb. Bridge=(E(F+A+1)+P)WD. Head score/value MACs sum heads×head-width=D, with masked dense bank slots included. Readout=R(D×256+256).','',
        'Architecture A uses 32 expert-isolated latent queries of 384. Each of two blocks performs cross-attention to that expert-common 20-family+1-process bank, self-attention only among those 32 latents, FFN 1536, and 3 norms. No query-private axis/local/raw-carbon bank is routed into latents. There remains indirect lawful axis/PFT/state information in retained process/family tokens.','',
        'A shared384→64 input affine acts once on 8550 query+224 latent+7 process+1380 field-axis+3249 field-local+2386 raw-carbon rows =15796 rows. Two 64 reader blocks (8 heads, FFN 256, 2 norms each) read 32 same-expert latents+1 process+original query-private axis/local. GPP alone additionally reads the same original direct raw-carbon bank/masks. K/V is cached only within the current differentiable invocation, never across experts or days. Field/coordinate identity and applicability masks remain required.','',
        f'Counted neural ratio current/A={totals["current_over_A_counted_MAC_ratio"]:.6f}. Unchanged tokenizer+shared input/backbone+bridge={totals["source_input_backbone_tokenizer_bridge_unchanged"]:,} MACs/sample-day. Even algebraically deleting every remaining decoder-related MAC would give only {totals["current"]/totals["source_input_backbone_tokenizer_bridge_unchanged"]:.6f} in this counted-forward scope. This is not a full-AD upper bound or measured speed. Amdahl sensitivity for a measured affected runtime fraction f and measured factor r is 1/((1−f)+f/r); neither runtime f nor r was measured here.','',
        '| Workload | Forward sample-days | Current counted MACs | A counted MACs |',
        '|---|---:|---:|---:|'] )
    for name,row in r['workload_counted_neural_MACs'].items():lines.append(f'| {name} | {row["sample_days"]} | {row["current"]:,} | {row["selected_architecture_A"]:,} |')
    lines.extend(['', 'Daily32 is eight microbatches of 4. T7 and T30 are sequential scans with one independent window each: 37 recurrent transitions, 69 forward sample-days total. B scales sample-local MACs linearly; H repeats complete state-dependent work and does not permit cross-day token reuse. Loss weights, gradients, rematerialization and optimizer work are not replaced by this forward count. Latent dense terms scale with E×M; latent self dots with E×M²; reader terms with B×R and reader bank slots, including M and GPP raw-carbon C.','',
        '## Logical activation volumes, micro4','',
        'These are single-layer source-shape volumes, not measured peak/live memory or AD tape sizes. All-row FFNs need not be live together: current/A expert chunks are sequentially mapped. Retained wide coordinate features/raw-carbon encoding still exist in A. H saved uncheckpointed days would multiply logical volumes; actual native checkpoint/recompute liveness is unknown.','',
        '| Tensor | FP32 logical shape | Bytes | MiB |','|---|---|---:|---:|'] )
    for name,item in r['logical_activations_micro4'].items():
        if isinstance(item,dict) and 'shape' in item:lines.append(f'| {name} | {item["shape"]} | {item["float32_bytes"]:,} | {item["float32_bytes"]/2**20:.6f} |')
    lines.extend(['', '## Source and execution record','',
        'Source references below are generated from the current AST. Parent graph Verify metadata changed, so exact source fallback was read. Worker graph transport was unavailable; no exhaustive call/branch claim is made. Source line positions and saved-layout matches establish this bounded bookkeeping only.','',
        '| Source | Symbol | Lines |','|---|---|---|'] )
    for ref in r['source_references']:lines.append(f'| {ref["file"]} | {ref["symbol"]} | {ref["start_line"]}–{ref["end_line"]} |')
    lines.extend(['',f'Actual local command: `{r["actual_command"]}`. Exit0; elapsed {r["elapsed_seconds"]:.6f}s, CPU process {r["CPU_process_seconds"]:.6f}s. See `audit.json` for exact integers, formula configuration, scaled workloads and file evidence. Separate symbolic initializer metadata took {r["parameter_shape_CPU_seconds"]["first_symbolic_elapsed"]:.6f}s and its metadata extension {r["parameter_shape_CPU_seconds"]["metadata_extension_elapsed"]:.6f}s; no neural arrays were allocated.','',
        'No architecture has been implemented or admitted. No GPU, remote compute, training, precision/loss/sampling/threshold changes or paid resource use. T30-vmap remains paused and the unselected 384→128 alternative is not evaluated or selected by this audit.'])
    (output/'audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def audit(output):
    began=time.perf_counter();cpu_started=time.process_time();output.mkdir(parents=True,exist_ok=True)
    layout,old,packing=current_layout()
    architecture=json.loads((ROOT/'manifests/coarse_graining/shared_transformer_split_width_v1.json').read_text())['architecture']
    configuration=dict(backbone_width=architecture['width'],backbone_ffn=architecture['ffn_width'],backbone_blocks=architecture['backbone_blocks'],
        decoder_width=architecture['decoder_width'],decoder_ffn=architecture['decoder_ffn_width'],decoder_blocks=architecture['decoder_blocks'],
        input_width=384,readout_width=256,source_rows=old['configuration']['source_tokens'],
        axis_rows_per_expert=sum(layout['axis_extents']),family_rows=20,carbon_rows=old['configuration']['raw_carbon_tokens'],PFT_rows=2)
    symbolic_path=output/'parameter-shapes.json'
    symbolic=json.loads(symbolic_path.read_text());metadata=symbolic['metadata']
    if metadata['source_token_total']!=configuration['source_rows'] or metadata['axis_total_extent']!=configuration['axis_rows_per_expert']:raise ValueError('Symbolic/current source token geometry differs')
    parameters=parameter_formula(metadata,configuration)
    if parameters['groups']!=symbolic['groups'] or parameters['total']!=64884481 or symbolic['numeric_leaves']!=300 or symbolic['static_none_count']!=2:raise ValueError('Parameter formula does not match actual symbolic initializer')
    parameters_A=parameter_formula(metadata,configuration,reader_width=64,latent_slots=32)
    if parameters_A['total']!=66123713:raise ValueError('Selected A parameter draft differs')
    tokenizer=tokenizer_mac_formula(metadata)
    macs=forward_mac_formula(layout,**configuration)
    no_pad=no_padding_layout(layout);current_lower=forward_mac_formula(no_pad,**configuration)
    candidate_A=candidate_A_mac_formula(layout,configuration);candidate_lower=candidate_A_mac_formula(no_pad,configuration)
    full_current=macs['total']+tokenizer['terms']['total'];full_A=candidate_A['total']+tokenizer['terms']['total']
    result=dict(status='current_source_layout_verified_MAC_and_parameter_metadata_audit',configuration=configuration,
        current_layout=layout,forward_macs_excluding_tokenizer=macs,
        source_token_metadata=metadata,parameter_count=parameters,selected_architecture_A_parameter_count=parameters_A,
        tokenizer_dense_and_self_attention_MACs=tokenizer,selected_architecture_A_MACs_excluding_tokenizer=candidate_A,
        no_coordinate_padding_lower_bounds=dict(current=current_lower,selected_architecture_A=candidate_lower,
            note='R8550, no coordinate padding ONLY; preserves current expert max private-axis slots, tokenizer axial padding and unchanged banks. It is not an implemented padding-free execution.'),
        counted_neural_dense_dot_totals=dict(current=full_current,selected_architecture_A=full_A,
            current_no_coordinate_padding=current_lower['total']+tokenizer['terms']['total'],
            selected_architecture_A_no_coordinate_padding=candidate_lower['total']+tokenizer['terms']['total'],
            current_over_A_counted_MAC_ratio=full_current/full_A,
            source_input_backbone_tokenizer_bridge_unchanged=macs['sections']['shared_input_and_backbone']['total']+macs['sections']['context_bridge']['total']+tokenizer['terms']['total']),
        logical_activations_micro4=logical_activations(layout,configuration),
        workload_forward_scaling=dict(micro4=4*macs['total'],daily32_eight_micro4=32*macs['total'],T7=7*macs['total'],T30=30*macs['total'],daily32_plus_T7_plus_T30=69*macs['total']),
        graph_evidence=dict(tier='Verify_parent_supplied',project='E-ORCHIDEE-MAN-JAX',generation='2026-10-04T11:28:29Z',
            coverage='no_recorded_issue_with_metadata_changed; exact sources read',worker_MCP='index_status Transport closed; source fallback'),
        scientific_or_execution_changes=0,neural_initializations=0,allocated_parameter_coordinates=0,model_calls=0,GPU_calls=0,training_updates=0,
        limits=['Runtime applicability masks depend on each actual input; static eligible is potentially active, not measured runtime nonzero.',
            'Counted neural total includes tokenizer dense/self-attention dots. It excludes tokenizer summary pooling/reductions, bias, embeddings/addition, GELU, normalization, softmax, masks, gather/scatter and physical processes.',
            'No AD/loss/optimizer/recompute/JIT cost multiplier, peak-memory or wall-time estimate is asserted.',
            'Shared source and field-local K/V reuse already exist; they are the current baseline, not a new route.',
            'Different field/coordinate feature inputs are not assumed identical; chunking does not create coordinate self attention.',
            'Forward69 sample-days is daily32 + recurrent7 + recurrent30. Recurrent trajectories have full state dependence, not69 interchangeable independent examples.'])
    references={
        'research/daily_coarse_graining/daily_operator/shared_transformer.py':['_initialize_tokenizer_parameters','initialize_shared_transformer_parameters','_source_family_bank','_project_decoder_context','_decoder_projection_cache','_field_dynamic_projection_cache','_attention_from_projected_segments','_run_coordinate_blocks_reusing_projections','_decoder_dynamic_plan','_decoder_dynamic_tokens','_field_coordinate_inputs','_carbon_tokens','_decode_field'],
        'research/daily_coarse_graining/daily_operator/shared_backbone_projection_reuse.py':['run_backbone_with_shared_source_projection'],
        'research/daily_coarse_graining/daily_operator/shared_decoder_macro_packed.py':['_decode_mapped_rows','expert_decode_fields'],
        'research/daily_coarse_graining/daily_operator/shared_decoder_expert_execution.py':['shared_transformer_raw_outputs_expert128'],
        'research/daily_coarse_graining/daily_operator/strong_baseline_architecture.py':['_linear_parameters','_residual_block_parameters','_forcing_tokens','_factorized_axial_tokens','assemble_strong_baseline_context'],
        'research/daily_coarse_graining/daily_operator/shared_recurrent_training.py':['make_shared_recurrent_objective'],
        'research/daily_coarse_graining/daily_operator/rollout_loss.py':['_scan']}
    result['source_references']=[source_ref(file,symbol) for file,symbols in references.items() for symbol in symbols]
    result['asset_evidence']=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (
        ROOT/'outputs/shared-decoder-cost-audit-20260923.json',ROOT/'outputs/shared-decoder-packing-audit-20260923.json',
        ROOT/'outputs/shared-decoder-macro-layout-20260925.json',ROOT/'manifests/coarse_graining/shared_transformer_split_width_v1.json',symbolic_path)]
    result['actual_command']='JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 conda run -n ORCJAX python -B scripts/hpc/audit_shared_decoder_structure.py'
    result['parameter_shape_CPU_seconds']=dict(first_symbolic_elapsed=symbolic['elapsed_seconds'],metadata_extension_elapsed=symbolic.get('metadata_extension_seconds'))
    result['workload_counted_neural_MACs']={name:dict(sample_days=scale,current=scale*full_current,selected_architecture_A=scale*full_A)
        for name,scale in (('micro4',4),('daily32_eight_micro4',32),('T7',7),('T30',30),('daily32_plus_T7_plus_T30',69))}
    result['elapsed_seconds']=time.perf_counter()-began;result['CPU_process_seconds']=time.process_time()-cpu_started
    (output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_markdown(output,result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args()
    result=audit(args.output)
    print(json.dumps(dict(status=result['status'],counts=result['current_layout']['counts'],forward_MACs_excluding_tokenizer=result['forward_macs_excluding_tokenizer']['total'],elapsed_seconds=result['elapsed_seconds'],CPU_process_seconds=result['CPU_process_seconds']),indent=2))

if __name__=='__main__':main()
