"""Diagnostic-only zero-delta observability at eight FP32 summary inputs.

Source: strong_baseline_architecture.assemble_strong_baseline_context,
lines 1100-1193 (field and expert-axis state summaries); original masked
summary reducers at lines 957-973. No process or precision replacement.
"""
from __future__ import annotations
import ast
from contextlib import contextmanager
import inspect
from pathlib import Path

DELTA_SHAPE = (7, 8, 3854)
CONSUMERS = ('state_field_pool',) + tuple(f'state_axis_pool[{i}]' for i in range(7))


def instrument_tree(source, filename='<summary-source>'):
    tree = ast.parse(source, filename=filename)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'assemble_strong_baseline_context']
    if len(functions) != 1:
        raise ValueError('expected exactly one original strong context assembler')
    function = functions[0]
    sites = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != '_masked_summary':
            continue
        if len(node.args) != 3 or not isinstance(node.args[1], ast.Name) or node.args[1].id != 'state':
            continue
        if not isinstance(node.args[2], ast.Name) or node.args[2].id != 'state_defined':
            raise ValueError('original source-defined mask binding differs')
        pool = ast.unparse(node.args[0])
        if pool == 'layout.state_field_pool':
            index = ast.Constant(0)
        elif pool == 'layout.state_axis_pool[expert_index]':
            index = ast.BinOp(ast.Constant(1), ast.Add(), ast.Name('expert_index', ast.Load()))
        else:
            raise ValueError('unexpected normalized-state consumer: ' + pool)
        sites.append(dict(pool=pool, line=node.lineno, end_line=node.end_lineno))
        node.args[1] = ast.BinOp(node.args[1], ast.Add(),
                               ast.Subscript(ast.Name('_public_summary_delta', ast.Load()), index, ast.Load()))
    if sorted(row['pool'] for row in sites) != ['layout.state_axis_pool[expert_index]', 'layout.state_field_pool']:
        raise ValueError('expected one field and one seven-expert axis callsite')
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    return module, dict(source_file=filename, subroutine=function.name,
                        line_span=[function.lineno, function.end_lineno], sites=sites,
                        delta_shape=list(DELTA_SHAPE), delta_dtype='float32', consumers=list(CONSUMERS),
                        masks='original state_defined passed unchanged', custom_VJP=False,
                        day_binding='dynamic delta carried as second member of scan days')


def source_check(path):
    return instrument_tree(Path(path).read_text(encoding='utf-8'), str(path))[1]


class SummarySeam:
    """One private assembler namespace; bind a tracer only during a step trace."""
    def __init__(self, original):
        path = inspect.getsourcefile(original)
        tree, self.evidence = instrument_tree(Path(path).read_text(encoding='utf-8'), path)
        self.namespace = dict(original.__globals__)
        self.namespace['_public_summary_delta'] = None
        exec(compile(tree, path, 'exec'), self.namespace)
        self.assembler = self.namespace[original.__name__]

    @contextmanager
    def bind_day(self, delta):
        if tuple(delta.shape) != DELTA_SHAPE[1:] or str(delta.dtype) != 'float32':
            raise ValueError('per-day summary delta ABI differs')
        prior = self.namespace['_public_summary_delta']
        self.namespace['_public_summary_delta'] = delta
        try:
            yield
        finally:
            self.namespace['_public_summary_delta'] = prior


def bind_rollout(rollout, objective, seam):
    """Use the exact public objective and its checkpointed scan with extra days."""
    original_step = rollout.step
    def step(parameters, state, discrete, day):
        original_day, delta_day = day
        with seam.bind_day(delta_day):
            return original_step(parameters, state, discrete, original_day)
    rollout.step = step
    def captured(parameters, state, discrete, days, targets, context, delta):
        if tuple(delta.shape) != DELTA_SHAPE or str(delta.dtype) != 'float32':
            raise ValueError('public T7 summary delta ABI differs')
        return objective(parameters, state, discrete, (days, delta), targets, context)
    return captured


@contextmanager
def install(backbone, seam):
    namespace = backbone.__globals__
    original = namespace.get('assemble_strong_baseline_context')
    if original is None or original.__name__ != 'assemble_strong_baseline_context':
        raise ValueError('actual staged backbone strong-assembler alias differs')
    namespace['assemble_strong_baseline_context'] = seam.assembler
    try:
        yield seam.evidence
    finally:
        namespace['assemble_strong_baseline_context'] = original
