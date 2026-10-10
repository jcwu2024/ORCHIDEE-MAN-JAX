"""Source-pinned public T7 overlay of the admitted component native casts.

The only inserted numerical statement widens the original normalized F32
state to F64. Eight original summary consumers still cast/reduce F32. Native
AD accumulates their F64 cotangents before the one original F32 boundary.
"""
from __future__ import annotations
import ast
import copy
from contextlib import contextmanager
from pathlib import Path
import normalized_state_fanout_component as component

MARKER = 'normalized_state_fanout_fp64'
CONSUMER_MARKER = 'normalized_state_consumer_'
CONSUMERS = ('state_field_pool',) + tuple(f'state_axis_pool[{i}]' for i in range(7))


def transformed_function(root):
    tree, source = component._source_pin(Path(root), component.SOURCE,
        ('_masked_summary', '_mask_inactive_pft_elements', 'assemble_strong_baseline_context'))
    _, normalizer = component._source_pin(Path(root), component.CONDITIONING, ('standardize_continuous',))
    original = component._function(tree, 'assemble_strong_baseline_context')
    function = copy.deepcopy(original)
    assignments = [n for n in function.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'state' for t in n.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Call) or ast.unparse(assignments[0].value.func) != 'standardize_continuous':
        raise ValueError('single source normalized-state assignment differs')
    consumers = [n for n in ast.walk(function) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == '_masked_summary'
        and len(n.args) == 3 and isinstance(n.args[1], ast.Name) and n.args[1].id == 'state']
    if sorted(ast.unparse(n.args[0]) for n in consumers) != ['layout.state_axis_pool[expert_index]', 'layout.state_field_pool']:
        raise ValueError('one field/seven-axis original state consumer sites differ')
    class ConsumerScopes(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            if node.value not in consumers: return node
            pool=ast.unparse(node.value.args[0])
            scope=(ast.Constant(CONSUMER_MARKER+'field') if pool=='layout.state_field_pool' else
                ast.JoinedStr(values=[ast.Constant(CONSUMER_MARKER+'axis_'),
                    ast.FormattedValue(value=ast.Name('expert_index',ast.Load()),conversion=-1)]))
            with_node=ast.With(items=[ast.withitem(context_expr=ast.Call(
                func=ast.Attribute(ast.Name('jax',ast.Load()),'named_scope',ast.Load()),args=[scope],keywords=[]))],body=[node])
            return ast.copy_location(with_node,node)
    function=ConsumerScopes().visit(function)
    statement = ast.parse("with jax.named_scope('normalized_state_fanout_fp64'):\n    state = jnp.asarray(state, dtype=jnp.float64)").body[0]
    ast.increment_lineno(statement,assignments[0].lineno-statement.lineno)
    insertion = function.body.index(assignments[0])+1
    function.body.insert(insertion, statement)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    compile(module, str(Path(root)/component.SOURCE), 'exec')
    # Every original statement remains byte-for-byte equivalent as an AST.
    restored = copy.deepcopy(function)
    restored.body.pop(insertion)
    class StripScopes(ast.NodeTransformer):
        def visit_With(self,node):
            self.generic_visit(node)
            expression=node.items[0].context_expr
            if isinstance(expression,ast.Call) and ast.unparse(expression.func)=='jax.named_scope':
                scope=expression.args[0]
                prefix=scope.value if isinstance(scope,ast.Constant) else scope.values[0].value if isinstance(scope,ast.JoinedStr) else ''
                if prefix.startswith(CONSUMER_MARKER): return node.body
            return node
    restored=StripScopes().visit(restored)
    if ast.dump(restored, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise ValueError('overlay changed original statements beyond native widen')
    return module, dict(status='prepared_native_cast_only', source=source+normalizer,
        component_policy='normalized_state_fanout_component.make_component(candidate=True)',
        mechanism='original F32 normalize -> one F64 shared state -> eight original F32 summary consumers',
        native_reverse='eight F32 partials widen -> native F64 fan-in -> one F32 normalization cotangent',
        consumers=list(CONSUMERS), marker=MARKER, summary_reducers='original F32',
        consumer_markers=[CONSUMER_MARKER+'field']+[CONSUMER_MARKER+f'axis_{i}' for i in range(7)],
        consumer_wrappers='jax.named_scope metadata only; original numeric calls unchanged',
        inserted_cast_source_line=statement.body[0].lineno,
        masks='all original masks unchanged', original_statement_AST_equal=True,
        delta_inputs=False, custom_VJP=False, changed_decoder_banks=False,
        actual_GPU_IR_proof=False, reference_adoption=False)


def source_check(root):
    return transformed_function(root)[1]


@contextmanager
def install(backbone, root):
    """Replace only the actual staged backbone's strong-assembler alias."""
    namespace = backbone.__globals__
    original = namespace.get('assemble_strong_baseline_context')
    if original is None or Path(original.__code__.co_filename).resolve() != (Path(root)/component.SOURCE).resolve():
        raise ValueError('actual strong assembler does not use declared source checkout')
    if not original.__globals__['jax'].config.x64_enabled:
        raise ValueError('native fan-in overlay requires JAX x64')
    tree, evidence = transformed_function(root)
    cloned_globals = dict(original.__globals__)
    exec(compile(tree, str(Path(root)/component.SOURCE), 'exec'), cloned_globals)
    candidate = cloned_globals[original.__name__]
    namespace['assemble_strong_baseline_context'] = candidate
    try:
        yield evidence
    finally:
        namespace['assemble_strong_baseline_context'] = original
        if namespace['assemble_strong_baseline_context'] is not original:
            raise RuntimeError('original staged backbone alias restoration failed')
