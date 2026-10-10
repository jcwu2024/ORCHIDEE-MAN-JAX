"""Scoped native-AD overlay for float64 accumulation of mapped decoder banks.

The overlay is source-pinned to the current shared transformer and packed
macro decoder. It widens only axis/local projected key/value banks before the
original macro lax.map, and restores selected gathers to float32 before the
original concatenate, masking, attention, normalization, and readout code.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib
import os
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ST_RELATIVE_PATH = Path("research/daily_coarse_graining/daily_operator/shared_transformer.py")
MACRO_RELATIVE_PATH = Path("research/daily_coarse_graining/daily_operator/shared_decoder_macro_packed.py")
ST_SHA256 = "8AF78FBDF969C97DEB553DA1AD2EC90FA4D817D2CDB79E3CFFF0A9AD4DD89B9D"
MACRO_SHA256 = "689BAA917AB518E87EAE809AF22900979D04EBED06825F8FF12F83F200E7158D"
DEPLOYED_SOURCE_PATHS = {
    ST_RELATIVE_PATH: "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1/research/daily_coarse_graining/daily_operator/shared_transformer.py",
    MACRO_RELATIVE_PATH: "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_macro_packed.py",
}

BANK_FIELDS = ("axis_key", "axis_value", "local_key", "local_value")
GATHER_EXPRESSIONS = {
    "projected.axis_key[axis_indices]": "axis_key",
    "projected.axis_value[axis_indices]": "axis_value",
    "projected.local_key[local_indices, None, :]": "local_key",
    "projected.local_value[local_indices, None, :]": "local_value",
}
MACRO_SCOPE = "bank_adjoint/macro_map_closure"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_key(path: str | Path) -> str:
    rendered = str(path).replace("\\", "/")
    if rendered.startswith("/WORK/"):
        return rendered.casefold()
    return os.path.normcase(str(Path(rendered).resolve()))


def _pinned_source(module: Any, relative_path: Path, expected_hash: str) -> Path:
    path = Path(module.__file__).resolve()
    local_allowed = PROJECT_ROOT / relative_path
    deployed_allowed = DEPLOYED_SOURCE_PATHS.get(relative_path)
    allowed = [_path_key(local_allowed)]
    if deployed_allowed is not None:
        allowed.append(_path_key(deployed_allowed))
    if _path_key(path) not in allowed:
        raise ValueError(
            f"source path mismatch for {module.__name__}: {path}; expected "
            f"{local_allowed} or {deployed_allowed}"
        )
    observed_hash = _sha256(path)
    if observed_hash.lower() != expected_hash.lower():
        raise ValueError(f"source SHA256 mismatch for {relative_path}: {observed_hash}")
    return path


def _find_function(module_ast: ast.Module, name: str) -> ast.FunctionDef:
    found = [node for node in module_ast.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(found) != 1:
        raise ValueError(f"expected one top-level function {name}, found {len(found)}")
    return found[0]


class _GatherRewrite(ast.NodeTransformer):
    def __init__(self) -> None:
        self.counts = {expression: 0 for expression in GATHER_EXPRESSIONS}

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        rendered = ast.unparse(node)
        field = GATHER_EXPRESSIONS.get(rendered)
        if field is None:
            return self.generic_visit(node)
        self.counts[rendered] += 1
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id="_bank64_gather", ctx=ast.Load()),
                args=[copy.deepcopy(node.value), copy.deepcopy(node.slice), ast.Constant(value=field)],
                keywords=[],
            ),
            node,
        )


def build_gather_helper(st: Any, *, transform: bool = True) -> tuple[Any, dict[str, Any]]:
    """Compile the exact pinned source helper, optionally wrapping four gathers."""
    source_path = _pinned_source(st, ST_RELATIVE_PATH, ST_SHA256)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    function = copy.deepcopy(_find_function(tree, "_run_coordinate_blocks_reusing_projections"))
    rewrite = _GatherRewrite()
    if transform:
        function = rewrite.visit(function)
        function = ast.fix_missing_locations(function)
        if any(count != 1 for count in rewrite.counts.values()):
            raise ValueError(f"expected exactly four source gathers, got {rewrite.counts}")

        jax_module, jnp_module = st.jax, st.jnp

        def bank64_gather(bank: Any, indices: Any, field: str) -> Any:
            expected_name = f"bank_adjoint/gather_{field}"
            if field not in BANK_FIELDS or getattr(bank.dtype, "name", None) != "float64":
                raise TypeError(f"{expected_name} requires a widened float64 bank")
            with jax_module.named_scope(expected_name):
                selected = bank[indices]
                return selected.astype(jnp_module.float32)

        namespace = dict(st.__dict__)
        namespace["_bank64_gather"] = bank64_gather
    else:
        namespace = dict(st.__dict__)

    module_ast = ast.Module(body=[function], type_ignores=[])
    code = compile(ast.fix_missing_locations(module_ast), str(source_path), "exec")
    exec(code, namespace)
    helper = namespace["_run_coordinate_blocks_reusing_projections"]
    return helper, {
        "source_path": str(source_path),
        "source_sha256": ST_SHA256,
        "transformed": bool(transform),
        "gather_rewrite_counts": dict(rewrite.counts) if transform else {},
        "gather_scope_markers": [f"bank_adjoint/gather_{field}" for field in BANK_FIELDS]
        if transform else [],
    }


def _baseline_helpers(st: Any, macro: Any, sba: Any | None) -> tuple[Any, Any, Any]:
    if sba is None:
        package = getattr(st, "__package__", "")
        if not package:
            raise ValueError("shared_transformer module has no package for baseline helper identity")
        sba = importlib.import_module(package + ".strong_baseline_architecture")
    helpers = ("_linear", "_ffn", "_layer_norm")
    for name in helpers:
        expected = getattr(sba, name, None)
        if expected is None or getattr(st, name, None) is not expected or getattr(macro, name, None) is not expected:
            raise ValueError(f"original strong-baseline helper identity failed for {name}")
    return tuple(getattr(sba, name) for name in helpers)


def _validate_expert_alias(expert: Any, macro: Any) -> bool | None:
    if expert is None:
        return None
    decoder = expert if callable(expert) else getattr(expert, "expert_decode_fields", None)
    if decoder is None or getattr(decoder, "__globals__", None) is not macro.__dict__:
        raise ValueError("expert_decode_fields globals are not the supplied macro module dictionary")
    return True


def _widen_projections(st: Any, projections: Any) -> tuple[Any, list[dict[str, Any]]]:
    if not isinstance(projections, tuple) or not projections:
        raise TypeError("projected banks must be a non-empty tuple")
    if not bool(st.jax.config.x64_enabled):
        raise RuntimeError("jax_enable_x64 must be enabled for bank-adjoint accumulation")

    widened = []
    bank_rows = []
    for block_index, projected in enumerate(projections):
        replace = getattr(projected, "_replace", None)
        if not callable(replace):
            raise TypeError("projection entries must preserve the original NamedTuple _replace API")
        updates = {}
        for field in BANK_FIELDS:
            bank = getattr(projected, field, None)
            if bank is None or getattr(bank.dtype, "name", None) != "float32":
                raise TypeError(f"block {block_index} {field} must be an original float32 projection bank")
            with st.jax.named_scope(f"bank_adjoint/widen_{field}"):
                wide = bank.astype(st.jnp.float64)
            updates[field] = wide
            bank_rows.append({
                "block_index": block_index,
                "field": field,
                "input_dtype": "float32",
                "closure_dtype": "float64",
                "shape": list(bank.shape),
                "elements": int(bank.size),
            })
        widened_item = replace(**updates)
        if type(widened_item) is not type(projected):
            raise TypeError("NamedTuple _replace changed the projection record type")
        widened.append(widened_item)
    return tuple(widened), bank_rows


@contextmanager
def patches(st: Any, macro: Any, expert: Any | None = None, sba: Any | None = None):
    """Temporarily install the source-cloned helper and F64 bank closure overlay.

    ``expert`` may be the adapter module or its ``expert_decode_fields`` callable.
    Metadata yielded to the caller accumulates one row per macro invocation.
    """
    st_path = _pinned_source(st, ST_RELATIVE_PATH, ST_SHA256)
    macro_path = _pinned_source(macro, MACRO_RELATIVE_PATH, MACRO_SHA256)
    _baseline_helpers(st, macro, sba)
    expert_alias_checked = _validate_expert_alias(expert, macro)
    if not bool(st.jax.config.x64_enabled):
        raise RuntimeError("jax_enable_x64 must be enabled for bank-adjoint accumulation")

    original_run = macro._run_coordinate_blocks_reusing_projections
    if original_run is not st._run_coordinate_blocks_reusing_projections:
        raise ValueError("macro's imported coordinate-block helper alias differs from shared_transformer")
    original_decode = macro._decode_mapped_rows
    cloned_run, transform_evidence = build_gather_helper(st, transform=True)
    metadata = {
        "status": "active",
        "overlay": "native_ad_projected_bank64_outer_map_closure",
        "source_files": {
            "shared_transformer": {"path": str(st_path), "sha256": ST_SHA256},
            "shared_decoder_macro_packed": {"path": str(macro_path), "sha256": MACRO_SHA256},
        },
        "source_clone": transform_evidence,
        "baseline_helper_identity_passed": True,
        "macro_imported_helper_alias_passed": True,
        "expert_decode_fields_globals_is_macro_dict": expert_alias_checked,
        "x64_enabled": True,
        "scope_markers": {
            "macro_map": MACRO_SCOPE,
            "bank_widen": [f"bank_adjoint/widen_{field}" for field in BANK_FIELDS],
            "bank_gather": transform_evidence["gather_scope_markers"],
        },
        "applications": [],
    }

    def decode_with_wide_closure(shared: Any, *, projections: Any, query: Any,
                                 chunk_size: int, **kwargs: Any) -> Any:
        widened, bank_rows = _widen_projections(st, projections)
        rows = int(query.shape[0])
        chunk = min(int(chunk_size), rows)
        chunk_count = (rows + chunk - 1) // chunk
        application = {
            "status": "in_progress",
            "rows": rows,
            "chunk_size": int(chunk_size),
            "effective_chunk_size": chunk,
            "chunk_count": chunk_count,
            "padded_rows": chunk_count * chunk,
            "padding_rows": chunk_count * chunk - rows,
            "projection_count": len(widened),
            "bank_count": len(bank_rows),
            "banks": bank_rows,
            "float32_narrowing_before_concat": True,
        }
        metadata["applications"].append(application)
        try:
            with st.jax.named_scope(MACRO_SCOPE):
                result = original_decode(shared, projections=widened, query=query,
                                         chunk_size=chunk_size, **kwargs)
            application["status"] = "complete"
            return result
        except BaseException:
            application["status"] = "failed"
            raise

    macro._run_coordinate_blocks_reusing_projections = cloned_run
    macro._decode_mapped_rows = decode_with_wide_closure
    try:
        yield metadata
        metadata["status"] = "completed"
    except BaseException:
        metadata["status"] = "failed"
        raise
    finally:
        macro._decode_mapped_rows = original_decode
        macro._run_coordinate_blocks_reusing_projections = original_run
        if (macro._decode_mapped_rows is not original_decode
                or macro._run_coordinate_blocks_reusing_projections is not original_run
                or st._run_coordinate_blocks_reusing_projections is not original_run):
            metadata["status"] = "restoration_failed"
            raise RuntimeError("macro/shared_transformer imported helper alias restoration failed")
        if (_sha256(st_path).lower() != ST_SHA256.lower()
                or _sha256(macro_path).lower() != MACRO_SHA256.lower()):
            metadata["status"] = "source_changed_during_overlay"
            raise RuntimeError("pinned source changed while bank overlay was active")


def _instructions(hlo: str) -> list[str]:
    blocks: list[list[str]] = []
    current: list[str] | None = None
    instruction_start = re.compile(r"^\s*(?:ROOT\s+)?%[\w.$-]+\s*=")
    for line in hlo.splitlines():
        if instruction_start.search(line):
            if current is not None:
                blocks.append(current)
            current = [line]
        elif current is not None:
            if line.lstrip().startswith(("ENTRY ", "}", "//")):
                blocks.append(current)
                current = None
            else:
                current.append(line)
    if current is not None:
        blocks.append(current)
    return ["\n".join(block) for block in blocks]


def _computation_sections(hlo: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    header = re.compile(r"^\s*(?:ENTRY\s+)?%?([\w.$-]+)(?:\s*\(.*)?\s*\{\s*$")
    for line in hlo.splitlines():
        match = header.match(line)
        if match:
            current = match.group(1)
            sections[current] = []
        elif current is not None and line.strip() == "}":
            current = None
        elif current is not None:
            sections[current].append(line)
    return sections


def _reachable_computation_lines(sections: dict[str, list[str]], root: str) -> list[str]:
    lines: list[str] = []
    pending = [root]
    seen: set[str] = set()
    reference = re.compile(r"\b(?:to_apply|body|condition|calls)\s*=\s*%?([\w.$-]+)")
    called_computations = re.compile(r"\bcalled_computations\s*=\s*\{([^}]*)\}")
    while pending:
        name = pending.pop()
        if name in seen or name not in sections:
            continue
        seen.add(name)
        body = sections[name]
        lines.extend(body)
        for line in body:
            targets = reference.findall(line)
            for group in called_computations.findall(line):
                targets.extend(re.findall(r"%?([\w.$-]+)", group))
            pending.extend(target for target in targets if target not in seen)
    return lines


def _reachable_computation_names(sections: dict[str, list[str]], root: str) -> set[str]:
    names=set();pending=[root]
    reference=re.compile(r"\b(?:to_apply|body|condition|calls)\s*=\s*%?([\w.$-]+)")
    called_computations=re.compile(r"\bcalled_computations\s*=\s*\{([^}]*)\}")
    while pending:
        name=pending.pop()
        if name in names or name not in sections:continue
        names.add(name)
        for line in sections[name]:
            targets=reference.findall(line)
            for group in called_computations.findall(line):
                targets.extend(re.findall(r"%?([\w.$-]+)",group))
            pending.extend(target for target in targets if target not in names)
    return names


def _f64_shape(instruction: str, operation: str) -> tuple[int, ...] | None:
    match = re.search(r"=\s*f64\[([^]]*)\](?:\{[^}]*\})?\s+" + re.escape(operation) + r"\s*\(", instruction)
    if match is None:
        return None
    dims = match.group(1).strip()
    if not dims:
        return ()
    try:
        return tuple(int(part.strip()) for part in dims.split(","))
    except ValueError:
        return None


def _tuple_shape(instruction: str, operation: str) -> list[tuple[int, ...] | None] | None:
    match = re.search(r"=\s*\((.*?)\)\s+" + re.escape(operation) + r"\s*\(", instruction)
    if match is None:
        return None
    shapes: list[tuple[int, ...] | None] = []
    for item in re.finditer(r"f64\[([^]]*)\]|(?:f32|s32|pred)\[([^]]*)\]", match.group(1)):
        dims = item.group(1) if item.group(1) is not None else item.group(2)
        try:
            shapes.append(tuple(int(part.strip()) for part in dims.split(",") if part.strip()))
        except ValueError:
            shapes.append(None)
    return shapes


def _marked_bank_shapes(instructions: list[str], marker_prefix: str, operation: str) -> dict[tuple[int, ...], int]:
    result: dict[tuple[int, ...], int] = {}
    for instruction in instructions:
        if marker_prefix not in instruction:
            continue
        shape = _f64_shape(instruction, operation)
        if shape is not None:
            result[shape] = result.get(shape, 0) + 1
    return result


def _split_arguments(text: str) -> list[str]:
    """Split HLO tuple types/operands without splitting shape/layout commas."""
    parts=[];start=0;depth=0
    for index,char in enumerate(text):
        if char in '([{':depth+=1
        elif char in ')]}':depth-=1
        elif char==',' and depth==0:
            parts.append(text[start:index].strip());start=index+1
    parts.append(text[start:].strip())
    return parts if text.strip() else []


def _hlo_type(text: str):
    text=re.sub(r'/\*.*?\*/','',text).strip()
    if text.startswith('(') and text.endswith(')'):
        children=[_hlo_type(part) for part in _split_arguments(text[1:-1])]
        return ('tuple',tuple(children)) if all(child is not None for child in children) else None
    match=re.fullmatch(r'(\w+)\[([\d,]*)\](?:\{[^}]*\})?',text)
    if match is None:return None
    return (match.group(1),tuple(int(value) for value in match.group(2).split(',') if value))


def _parsed_computation(lines):
    parsed={};root=None
    pattern=re.compile(r'^\s*(ROOT\s+)?%([\w.$-]+)\s*=\s*(.*?)\s+([\w-]+)\((.*?)\)(.*)$',re.S)
    for line in _instructions('\n'.join(lines)):
        match=pattern.match(line)
        if match is None:continue
        kind=_hlo_type(match.group(3))
        if kind is None:continue
        name=match.group(2)
        parsed[name]=(kind,match.group(4),_split_arguments(match.group(5)),match.group(6))
        if match.group(1):root=name
    return parsed,root


def _carry_add_linked(sections, body_name: str, field: str, bank_shape: tuple[int, ...], *, parsed_sections=None) -> bool:
    """Resolve fusion parameters in their own computation and exact call context.

    Proof binds outer parameter tuple index -> F64 carry -> native F64 add
    with the field-marked F64 scatter -> the same ROOT tuple index. Names
    are never flattened across computations. Unknown operations fail closed.
    """
    cache=parsed_sections if parsed_sections is not None else {name:_parsed_computation(lines) for name,lines in sections.items()}
    if body_name not in cache:return False
    body,root=cache[body_name]
    parameters=[row for row in body.values() if row[1]=='parameter' and row[2]==['0']]
    if len(parameters)!=1 or root is None or parameters[0][0][0]!='tuple':return False
    parameter_type=parameters[0][0]
    bound=('tuple',tuple(('carry',index,kind) for index,kind in enumerate(parameter_type[1])),parameter_type)

    def resolve(computation,name,bindings,stack):
        key=(computation,name)
        if key in stack or computation not in cache:return None
        definitions,_=cache[computation]
        if name not in definitions:return None
        kind,operation,args,attributes=definitions[name]
        stack=stack|{key}
        if operation=='parameter':
            if len(args)!=1 or not args[0].isdigit():return None
            value=bindings.get(int(args[0]))
            return value if value is not None and value[-1]==kind else None
        def operand(argument):
            names=re.findall(r'%([\w.$-]+)',argument)
            return resolve(computation,names[-1],bindings,stack) if len(names)==1 else None
        if operation=='scatter':
            marker=re.search(r'bank_adjoint/gather_(axis_key|axis_value|local_key|local_value)(?:[/"\s]|$)',attributes)
            target=re.search(r'\bto_apply\s*=\s*%?([\w.$-]+)',attributes)
            if not marker or not target or target.group(1) not in cache or not args:return None
            update,update_root=cache[target.group(1)]
            if update_root is None:return None
            reducer=update[update_root]
            if reducer[0]!=('f64',()) or reducer[1]!='add' or len(reducer[2])!=2:return None
            indices=[]
            for argument in reducer[2]:
                names=re.findall(r'%([\w.$-]+)',argument)
                node=update.get(names[0]) if len(names)==1 else None
                if node is None or node[0]!=('f64',()) or node[1]!='parameter' or len(node[2])!=1:return None
                indices.append(node[2][0])
            if sorted(indices)!=['0','1']:return None
            return ('scatter',marker.group(1),operand(args[0]),kind)
        if operation=='tuple':
            children=tuple(operand(argument) for argument in args)
            return ('tuple',children,kind)
        if operation=='get-tuple-element' and len(args)==1:
            value=operand(args[0]);index=re.search(r'\bindex=(\d+)',attributes)
            if value and value[0]=='tuple' and index and int(index.group(1))<len(value[1]):
                child=value[1][int(index.group(1))]
                return child if child is not None and child[-1]==kind else None
            return None
        if operation in ('copy','bitcast','reshape') and len(args)==1:
            value=operand(args[0])
            return value if value and value[-1]==kind else None
        if operation=='add' and len(args)==2:
            children=tuple(operand(argument) for argument in args)
            return ('add',children,kind) if all(child is not None and child[-1]==kind for child in children) else None
        if operation=='while':
            body_match=re.search(r'\bbody\s*=\s*%?([\w.$-]+)',attributes)
            if len(args)!=1 or body_match is None or body_match.group(1) not in cache:return None
            carried=operand(args[0])
            if carried is None or carried[0]!='tuple' or carried[-1]!=kind:return None
            _body,body_root=cache[body_match.group(1)]
            if body_root is None:return None
            body_parameters=[row for row in _body.values() if row[1]=='parameter' and row[2]==['0']]
            if len(body_parameters)!=1 or body_parameters[0][0]!=carried[-1]:return None
            value=resolve(body_match.group(1),body_root,{0:carried},stack)
            return value if value and value[-1]==kind else None
        if operation in ('fusion','call'):
            targets=re.findall(r'\b(?:calls|to_apply)\s*=\s*%?([\w.$-]+)',attributes)
            if not targets:
                groups=re.findall(r'\bcalled_computations\s*=\s*\{([^}]*)\}',attributes)
                targets=[name for group in groups for name in re.findall(r'%?([\w.$-]+)',group)]
            if len(targets)!=1 or targets[0] not in cache:return None
            _callee,callee_root=cache[targets[0]]
            values={index:operand(argument) for index,argument in enumerate(args)}
            if callee_root is None:return None
            value=resolve(targets[0],callee_root,values,stack)
            return value if value and value[-1]==kind else None
        return None

    # Resolve members independently: unrelated ROOT tuple members may use
    # operations outside this deliberately bounded proof vocabulary.
    kind,operation,args,_=body[root]
    if operation!='tuple' or kind!=parameter_type:return False
    for index,argument in enumerate(args):
        if index>=len(parameter_type[1]) or parameter_type[1][index]!=('f64',bank_shape):continue
        names=re.findall(r'%([\w.$-]+)',argument)
        if len(names)!=1:continue
        value=resolve(body_name,names[0],{0:bound},set())
        if not value or value[-1]!=('f64',bank_shape):continue
        carry=('carry',index,('f64',bank_shape))
        if value[0]=='scatter' and value[1]==field and value[2]==carry:return True
        if value[0]=='add':
            left,right=value[1]
            for carried,scattered in ((left,right),(right,left)):
                if carried==carry and scattered[0]=='scatter' and scattered[1]==field and scattered[-1]==('f64',bank_shape):return True
    return False


def _outer_while_signature_matches(parsed_sections, row, body_name: str) -> bool:
    name_match=re.match(r"\s*%([\w.$-]+)\s*=",row)
    if name_match is None:return False
    loop_name=name_match.group(1)
    owners=[definitions for definitions,_ in parsed_sections.values()
        if loop_name in definitions and definitions[loop_name][1]=='while'
        and re.search(r'\bbody\s*=\s*%?'+re.escape(body_name)+r'(?:\b|$)',definitions[loop_name][3])]
    if len(owners)!=1:return False
    owner=owners[0];kind,operation,args,attributes=owner[loop_name]
    if operation!='while' or len(args)!=1:return False
    body,root=parsed_sections.get(body_name,({},None))
    if root is None:return False
    parameters=[value for value in body.values() if value[1]=='parameter' and value[2]==['0']]
    if len(parameters)!=1 or parameters[0][0]!=kind or body[root][0]!=kind:return False
    input_names=re.findall(r'%([\w.$-]+)',args[0])
    if len(input_names)!=1 or input_names[0] not in owner or owner[input_names[0]][0]!=kind:return False
    return True


def _marked_scatter_reaches_root(parsed_sections, field: str, bank_shape: tuple[int, ...]) -> bool:
    marker=re.compile(r'bank_adjoint/gather_'+re.escape(field)+r'(?:[ /"\s]|$)')
    for definitions,root in parsed_sections.values():
        if root is None:continue
        users={name:[] for name in definitions}
        for user,(_,_,args,_) in definitions.items():
            for argument in args:
                for operand_name in re.findall(r'%([\w.$-]+)',argument):
                    if operand_name in users:users[operand_name].append(user)
        pending=[]
        for name,(kind,operation,args,attributes) in definitions.items():
            if operation!='scatter' or kind!=('f64',bank_shape) or marker.search(attributes) is None:continue
            target=re.search(r'\bto_apply\s*=\s*%?([\w.$-]+)',attributes)
            if target is None or target.group(1) not in parsed_sections:continue
            reducer,reducer_root=parsed_sections[target.group(1)]
            if reducer_root is None:continue
            root_row=reducer[reducer_root]
            if root_row[0]!=('f64',()) or root_row[1]!='add' or len(root_row[2])!=2:continue
            parameters=[]
            for argument in root_row[2]:
                names=re.findall(r'%([\w.$-]+)',argument)
                node=reducer.get(names[0]) if len(names)==1 else None
                if node is None or node[0]!=('f64',()) or node[1]!='parameter' or len(node[2])!=1:
                    parameters=[];break
                parameters.append(node[2][0])
            if sorted(parameters)==['0','1']:pending.append(name)
        seen=set()
        while pending:
            name=pending.pop()
            if name==root:return True
            if name in seen:continue
            seen.add(name);pending.extend(users.get(name,()))
    return False


def _required_carries_covered(required_carries, linked_carries) -> bool:
    return all(required_carries[field].issubset(linked_carries[field]) for field in BANK_FIELDS)


def audit_gpu_hlo(hlo: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless marked reverse F64 scatter and outer carry are linked.

    HLO formats vary by JAX/XLA version. Unknown or optimized-away markers yield
    ``incomplete``; they are never inferred from an unrelated float64 loop.
    """
    instructions = _instructions(hlo)
    applications = evidence.get("applications", [])
    expected_shapes: dict[str, set[tuple[int, ...]]] = {field: set() for field in BANK_FIELDS}
    required_carry_shapes: dict[str, set[tuple[int, ...]]] = {field: set() for field in BANK_FIELDS}
    application_metadata_valid=True
    for application in applications:
        if application.get("status") != "complete":
            continue
        rows=application.get("rows")
        chunk_size=application.get("chunk_size")
        effective_chunk_size=application.get("effective_chunk_size")
        chunk_count=application.get("chunk_count")
        padded_rows=application.get("padded_rows")
        if not all(isinstance(value,int) and value>0 for value in (rows,chunk_size,effective_chunk_size,chunk_count,padded_rows)):
            application_metadata_valid=False
        elif ((rows+effective_chunk_size-1)//effective_chunk_size!=chunk_count
                or effective_chunk_size!=min(rows,chunk_size)
                or padded_rows!=chunk_count*effective_chunk_size):
            application_metadata_valid=False
        for bank in application.get("banks", []):
            field = bank.get("field")
            shape = tuple(bank.get("shape", ()))
            if field in expected_shapes and all(isinstance(dim, int) for dim in shape):
                expected_shapes[field].add(shape)
                if isinstance(chunk_count,int) and chunk_count>1:
                    required_carry_shapes[field].add(shape)
    expected_bank_ops = sum(len(shapes) for shapes in expected_shapes.values())
    sections = _computation_sections(hlo)
    entry_match=re.search(r"^\s*ENTRY\s+%?([\w.$-]+)",hlo,re.M)
    reachable_names=_reachable_computation_names(sections,entry_match.group(1)) if entry_match else set()
    parsed_sections={name:_parsed_computation(lines) for name,lines in sections.items()}
    reachable_parsed_sections={name:parsed_sections[name] for name in reachable_names if name in parsed_sections}
    fields = {}
    for field in BANK_FIELDS:
        marker = f"bank_adjoint/gather_{field}"
        gather_shapes = _marked_bank_shapes(instructions, marker, "gather")
        scatter_shapes = _marked_bank_shapes(instructions, marker, "scatter")
        consumed_shapes={shape for shape in expected_shapes[field]
            if _marked_scatter_reaches_root(reachable_parsed_sections,field,shape)}
        fields[field] = {
            "expected_deduplicated_bank_shapes": [list(shape) for shape in sorted(expected_shapes[field])],
            "marked_f64_gather_shapes": {str(shape): count for shape, count in sorted(gather_shapes.items())},
            "marked_f64_scatter_shapes": {str(shape): count for shape, count in sorted(scatter_shapes.items())},
            "marked_scatter_shapes_reaching_computation_root": [list(shape) for shape in sorted(consumed_shapes)],
            "required_shapes_present": bool(expected_shapes[field]) and expected_shapes[field].issubset(scatter_shapes.keys()) and expected_shapes[field].issubset(consumed_shapes),
            "forward_gather_precision_required": False,
            "marker": marker,
        }
    macro_loops = []
    linked={field:set() for field in BANK_FIELDS}
    for row in instructions:
        if MACRO_SCOPE not in row or not re.search(r"\bwhile\s*\(", row) or "f64[" not in row:
            continue
        body_match = re.search(r"\bbody\s*=\s*%?([\w.$-]+)", row)
        if body_match is None:
            continue
        if not _outer_while_signature_matches(reachable_parsed_sections,row,body_match.group(1)):
            continue
        found=False
        for field,shapes in expected_shapes.items():
            for shape in shapes:
                if _carry_add_linked(sections,body_match.group(1),field,shape,parsed_sections=parsed_sections):
                    linked[field].add(shape);found=True
        if found:
            macro_loops.append(row)
    enough_fields = all(row["required_shapes_present"] for row in fields.values())
    carry_shapes_verified=_required_carries_covered(required_carry_shapes,linked)
    verified = bool(expected_bank_ops and enough_fields and application_metadata_valid and carry_shapes_verified)
    return {
        "passed": verified,
        "status": "verified" if verified else "incomplete",
        "expected_deduplicated_bank_shape_count": expected_bank_ops,
        "field_operations": fields,
        "marked_f64_macro_while_count": len(macro_loops),
        "validated_application_chunk_metadata": application_metadata_valid,
        "required_outer_carry_shapes":{field:[list(shape) for shape in sorted(shapes)] for field,shapes in required_carry_shapes.items()},
        "macro_scope_marker": MACRO_SCOPE,
        "linked_outer_carry_shapes":{field:[list(shape) for shape in sorted(shapes)] for field,shapes in linked.items()},
        "verified_by": "field-marked reverse F64 scatter with a native scalar F64 add reducer and SSA path to computation ROOT; multi-chunk banks also require exact outer carry ROOT tuple linkage",
        "limitations": [
            "This line-oriented analyzer is intentionally fail-closed and may return incomplete for valid alternate XLA text formats.",
            "It does not claim that an arbitrary float64 physics loop is a bank-adjoint carry; both the unique macro scope and field gather markers are required.",
            "CPU prototype evidence does not establish GPU lowering or memory behavior.",
            "Forward gather output shape is the selected shape, not the full bank shape; F64 primal gathers may legally optimize to F32.",
            "Unknown fusion/custom-call semantics, tuple-root fusions, reshaped carry types or lowered scatter implementations require targeted saved-executable/source review before numerical evaluation; incomplete is not proof of boundary removal.",
        ],
    }
