"""Current prepared contract/output-expert mapping; metadata only, no forward."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-context", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with args.prepared_context.open("rb") as stream:
        context = pickle.load(stream)
    head = context["head"]
    fields = [{"field": field.name, "shape": field.shape,
               "expert_index": int(head.layout.field_expert_index[index]),
               "expert_name": head.layout.expert_names[int(head.layout.field_expert_index[index])]}
              for index, field in enumerate(head.learned_schema.fields)]
    state = [{"key": leaf.key, "shape": leaf.shape, "start": leaf.start, "stop": leaf.stop, "source_ref": leaf.source_ref}
             for leaf in head.contract.state_leaves if leaf.start is not None]
    report = {"scope": "prepared_metadata_only_no_forward_or_compilation", "identity": context["identity"],
              "expert_names": head.layout.expert_names, "output_fields": fields, "continuous_state_leaves": state,
              "interpretation_limit": "Output ownership does not identify which decoder caused a shared-input state derivative discrepancy."}
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
