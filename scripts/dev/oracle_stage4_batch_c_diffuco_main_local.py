"""Bind the real active DIFFUCO main local-chain trace to its owner arms."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/reference_mode/micro_oracles/diffuco_batch_c/main_local'
OUT.mkdir(parents=True,exist_ok=True)
command=[sys.executable,'-m','pytest','-q','tests/unit/test_diffuco.py','-k','server_1961_diffuco_trans_co2_pft14_first_active_record_matches_fortran_trace or server_1961_diffuco_pft14_active_main_beta_closure_matches_after_main_trace']
run=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
trace=ROOT/'outputs/server_1961_diffuco_active_trace_20260624_181143/traces/orchjax_diffuco_trans_co2_trace.txt'
run_def=ROOT/'outputs/server_1961_diffuco_active_trace_20260624_181143/run/used_run.def'
assert 'after_diffuco_trans_co2_pft14' in trace.read_text(encoding='utf-8')
assert 'CHEMISTRY_BVOC = FALSE' in run_def.read_text(encoding='utf-8')
result={'schema_version':2,'family':'diffuco_batch_c_main_local','status':'passed' if run.returncode==0 else 'failed','comparisons':[{'name':'original_fortran_active_trans_co2_trace_to_jax','comparison':'pytest','passed':run.returncode==0}],'trace_asset':str(trace.relative_to(ROOT)).replace('\\','/'),'run_def_asset':str(run_def.relative_to(ROOT)).replace('\\','/'),'gcov_mapping':{'fortran_source/ORCHIDEE/src_sechiba/diffuco.f90:665:if:true':{'kind':'real_fortran_trace','tag':'after_diffuco_trans_co2_pft14','kjit':49},'fortran_source/ORCHIDEE/src_sechiba/diffuco.f90:686:if:false':{'kind':'exact_source_config_proof','setting':'CHEMISTRY_BVOC = n'}},'stdout':run.stdout,'stderr':run.stderr}
(OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
raise SystemExit(run.returncode)
