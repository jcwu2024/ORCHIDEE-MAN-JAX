"""Collect full native-AD dot type/shape/algorithm signatures without full IR text."""
from collections import Counter
import json


def collect(lowered):
    module = lowered.compiler_ir(dialect="stablehlo")
    counters, samples = Counter(), {}
    operations = dots = 0
    def walk(operation):
        yield operation
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from walk(child.operation)
    for operation in walk(module.operation):
        operations += 1
        if operation.name != "stablehlo.dot_general":
            continue
        dots += 1
        attributes = {}
        for attribute in operation.attributes:
            name = attribute if isinstance(attribute, str) else attribute.name
            attributes[str(name)] = str(operation.attributes[str(name)])
        row = {"operands": [str(value.type) for value in operation.operands],
               "results": [str(value.type) for value in operation.results],
               "attributes": attributes}
        key = json.dumps(row, sort_keys=True)
        counters[key] += 1
        if key not in samples:
            samples[key] = str(operation.location)[:600]
    return {"scope": "complete_four_sample_public_loss_native_parameter_and_state_AD_StableHLO",
            "IR_operations": operations, "dot_general_count": dots,
            "distinct_dot_signatures": len(counters),
            "signatures": [{**json.loads(key), "count": count, "location_sample": samples[key]} for key, count in counters.items()],
            "source_metadata_full_text_exported": False, "model_backend_compilations": 0,
            "model_values_gradients_or_speed_measured": False}
