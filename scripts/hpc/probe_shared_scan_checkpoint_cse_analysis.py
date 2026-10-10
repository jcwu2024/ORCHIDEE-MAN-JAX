"""CPU-only bounded analysis of preserved CSE pairs; never compiles/replays GPU."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import re
import numpy as np


def leaves(tree):
    if isinstance(tree,dict):
        return [v for k in sorted(tree) for v in leaves(tree[k])]
    if isinstance(tree,(tuple,list)):
        return [v for item in tree for v in leaves(item)]
    return [] if tree is None else [np.asarray(tree)]


def scopes(value):
    return {"loss_and_aux":value[0],"decoder_parameter_vjp":value[1][0][0],
            "captured_dynamic_raw_bank_vjp":value[1][0][1:],"initial_query_vjp":value[1][1]}


def compare(a,b):
    failed=0;maximum=0.0;identical=True
    for x,y in zip(leaves(a),leaves(b),strict=True):
        if x.shape!=y.shape or x.dtype!=y.dtype:
            raise ValueError("shape/dtype differs")
        d=np.abs(x.astype(np.float64)-y)
        failed+=int(np.count_nonzero(~(np.isfinite(x)&np.isfinite(y)&(d<=1e-6+1e-5*np.abs(y)))))
        maximum=max(maximum,float(d.max(initial=0)))
        identical &= np.array_equal(x,y)
    return dict(passed=failed==0,failed_coordinates=failed,max_abs_error=maximum,bitwise_identical=bool(identical))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    args=parser.parse_args()
    stem=args.directory/"component"
    def read(name):
        with Path(str(stem)+name).open("rb") as stream:
            return pickle.load(stream)
    initial={name:read(f".{name}.initial.pkl") for name in ("true","false")}
    report=dict(status="completed",gpu_computation=False,new_compiles=0,self_drift_not_tolerance_waiver=True,
                self_comparisons={"true":[],"false":[]},initial_finite_and_nonzero={},order_strata={})
    for name,value in initial.items():
        report["initial_finite_and_nonzero"][name]={key:dict(finite=all(np.isfinite(x).all() for x in leaves(tree)),
          coordinates=sum(x.size for x in leaves(tree)),nonzero_coordinates=sum(int(np.count_nonzero(x)) for x in leaves(tree)))
          for key,tree in scopes(value).items()}
    persistent=None
    for repeat in range(7):
        pair=read(f".pair{repeat}.pkl")
        for name in pair:
            report["self_comparisons"][name].append({k:compare(scopes(pair[name])[k],scopes(initial[name])[k]) for k in scopes(pair[name])})
        masks={}
        for key in scopes(pair["true"]):
            masks[key]=[(np.isfinite(x)&np.isfinite(y)&(np.abs(x.astype(np.float64)-y)>1e-6+1e-5*np.abs(y)))
                for x,y in zip(leaves(scopes(pair["false"])[key]),leaves(scopes(pair["true"])[key]),strict=True)]
        if persistent is None:
            persistent=masks
        else:
            persistent={k:[a&b for a,b in zip(persistent[k],masks[k],strict=True)] for k in masks}
    report["coordinates_failing_all_seven_pairs"]={k:sum(int(np.count_nonzero(v)) for v in masks) for k,masks in persistent.items()}
    timed=json.loads(Path(str(stem)+".json").read_text())
    for label,select in (("true_then_false",(0,2,4,6)),("false_then_true",(1,3,5))):
        median={name:float(np.median([timed["variants"][name]["warm_seconds"][i] for i in select])) for name in ("true","false")}
        report["order_strata"][label]={**median,"speedup":median["true"]/median["false"]}
    report["optimized_hlo_counts"]={}
    for name in ("true","false"):
        hlo=Path(str(stem)+f".{name}.compiledhlo.txt").read_text()
        report["optimized_hlo_counts"][name]={k:len(re.findall(pattern,hlo)) for k,pattern in {
            "fusion":r"\bfusion\(","custom_call":r"\bcustom-call\(","while":r"\bwhile\(",
            "scatter":r"\bscatter\(","gather":r"\bgather\(","optimization_barrier":r"\bopt-barrier\("}.items()}
    (args.directory/"analysis.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
