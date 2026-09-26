"""Inventory tracked source without importing research code or running a model.

Print compact summaries by default; detailed reports go only beneath outputs/.
This is conservative dependency evidence, never automatic deletion authority.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "manifests/coarse_graining/research_workspace.json"
TEXT_SUFFIXES = {".py", ".sh", ".json", ".md", ".toml", ".yaml", ".yml", ".txt", ".ps1"}
DYNAMIC_CALLS = {"import_module", "__import__", "spec_from_file_location", "run_path", "exec", "eval"}


def module_name(path):
    parts = Path(path).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def analyze_sources(sources):
    """Resolve local AST imports, package initialization and literal paths.

    ``sources`` maps repository-relative names to text, allowing fixture tests.
    Unknown computed paths are reported separately, not guessed into closure.
    """
    modules = {module_name(path): path for path in sources if path.endswith(".py")}
    basenames = defaultdict(set)
    for path in sources:
        basenames[Path(path).name].add(path)
    edges = defaultdict(set)
    dynamic = defaultdict(list)
    errors = {}
    lazy_exports = {}
    lazy_star_exports = {}
    export_maps = {"_PUBLIC_EXPORTS", "_COMPAT_EXPORTS", "_COMPAT_MODULES"}
    for path, source in sources.items():
        if not path.endswith("/__init__.py") or "_PUBLIC_EXPORTS" not in source:
            continue
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            # The main parse pass records errors uniformly for every file.
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id in export_maps for target in node.targets
            ):
                continue
            exports = ast.literal_eval(node.value)
            package = module_name(path)
            targets = {name: value if isinstance(value, str) else value[0]
                       for name, value in exports.items()}
            resolved = {
                name: package + value if value.startswith(".") else value
                for name, value in targets.items()
            }
            lazy_exports.setdefault(package, {}).update(resolved)
            if any(isinstance(target, ast.Name) and target.id == "_PUBLIC_EXPORTS" for target in node.targets):
                lazy_star_exports[package] = resolved

    def add_module(owner, name):
        target = modules.get(name)
        if target:
            edges[owner].add(target)
        parts = name.split(".")
        for count in range(1, len(parts) + 1):
            package = modules.get(".".join(parts[:count]))
            if package and package.endswith("/__init__.py"):
                edges[owner].add(package)

    def add_literal(owner, value):
        normalized = value.replace("\\", "/").removeprefix("./")
        if normalized in sources:
            edges[owner].add(normalized)
        # Filenames are conservative evidence for ROOT / directory / filename.
        if "/" not in normalized and normalized in basenames:
            edges[owner].update(basenames[normalized])

    for path, source in sources.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as error:
            errors[path] = {"line": error.lineno, "message": error.msg}
            continue
        owner_module = module_name(path)
        package = owner_module if path.endswith("/__init__.py") else owner_module.rpartition(".")[0]
        # Executing/importing a module also initializes its containing packages.
        add_module(path, owner_module)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(path, alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".")
                    base = ".".join(parts[:len(parts) - node.level + 1])
                    name = ".".join(filter(None, (base, node.module)))
                else:
                    name = node.module or ""
                add_module(path, name)
                for alias in node.names:
                    add_module(path, ".".join(filter(None, (name, alias.name))))
                    exports = lazy_exports.get(name, {})
                    if alias.name == "*":
                        for target_module in lazy_star_exports.get(name, {}).values():
                            add_module(path, target_module)
                    elif alias.name in exports:
                        add_module(path, exports[alias.name])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                add_literal(path, node.value)
            elif isinstance(node, ast.Call):
                called = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                if called.lstrip("_") in DYNAMIC_CALLS:
                    dynamic[path].append({"line": node.lineno, "call": called})
                    if called.lstrip("_") in {"import_module", "import"} and node.args:
                        value = node.args[0]
                        if isinstance(value, ast.Constant) and isinstance(value.value, str):
                            add_module(path, value.value)
        edges[path].discard(path)

    # Shell -m invocations and exact tracked path/filename references are kept
    # as independent inbound evidence. Documentation is not a runtime edge.
    references = defaultdict(list)
    for path, source in sources.items():
        for line_number, line in enumerate(source.splitlines(), 1):
            for token in set(re.findall(r"[\w./\\-]+\.(?:py|sh|json)", line)):
                normalized = token.replace("\\", "/").removeprefix("./")
                targets = {normalized} if normalized in sources else basenames.get(Path(normalized).name, set())
                for target in targets:
                    if target != path:
                        references[target].append({"path": path, "line": line_number})
                        if path.endswith((".sh", ".json")):
                            edges[path].add(target)
            if path.endswith(".sh"):
                for name in re.findall(r"\s-m\s+([\w.]+)", line):
                    add_module(path, name)
    return edges, references, dynamic, errors


def closure(seeds, edges):
    reached = set(seeds)
    pending = list(seeds)
    while pending:
        for target in edges.get(pending.pop(), ()):
            if target not in reached:
                reached.add(target)
                pending.append(target)
    return sorted(reached)


def build_report(root, manifest):
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    sources = {}
    for path in tracked:
        if Path(path).suffix in TEXT_SUFFIXES and (root / path).is_file():
            sources[path] = (root / path).read_text(encoding="utf-8-sig", errors="replace")
    edges, references, dynamic, errors = analyze_sources(sources)
    groups = {}
    for name, config in manifest["groups"].items():
        reached = closure(config["seeds"], edges)
        groups[name] = {
            "status": config["status"], "seeds": config["seeds"],
            "missing_seeds": sorted(set(config["seeds"]) - sources.keys()),
            "files": reached, "file_count": len(reached),
            "python_count": sum(path.endswith(".py") for path in reached),
            "lines": sum(len(sources.get(path, "").splitlines()) for path in reached),
            "dynamic_loading": {path: dynamic[path] for path in reached if path in dynamic},
        }
    candidates = {}
    for target in manifest.get("retirement_candidates", []):
        candidates[target] = {
            "retained_by": [name for name, group in groups.items() if target in group["files"]],
            "inbound_dependencies": sorted(path for path, targets in edges.items() if target in targets),
            "literal_references": references.get(target, []),
        }
    active_files = {path for group in groups.values() for path in group["files"]}
    compatibility = {}
    for path, source in sources.items():
        if path.endswith("/__init__.py") and "_PUBLIC_EXPORTS" in source and path not in errors:
            tree = ast.parse(source, filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in {"_PUBLIC_EXPORTS", "_COMPAT_EXPORTS", "_COMPAT_MODULES"}
                    for target in node.targets
                ):
                    exports = ast.literal_eval(node.value)
                    package_path = Path(path).parent
                    modules = [value if isinstance(value, str) else value[0] for value in exports.values()]
                    targets = [str(package_path / (value.lstrip(".").replace(".", "/") + ".py")).replace("\\", "/")
                               for value in modules if value.startswith(".")]
                    previous = compatibility.get(path, {"files": [], "export_count": 0})
                    compatibility[path] = {"files": closure(targets + previous["files"], edges),
                                           "export_count": previous["export_count"] + len(exports)}
    return {"schema_version": "research_workspace_inventory_v1", "recovery_tag": manifest["recovery_tag"],
            "groups": groups, "retirement_candidates": candidates, "parse_errors": errors,
            "retired_entrypoints": manifest.get("retired_entrypoints", []),
            "active_parse_errors": {path: error for path, error in errors.items() if path in active_files},
            "lazy_public_api_compatibility": compatibility,
            "limitations": manifest["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / MANIFEST)
    parser.add_argument("--output", type=Path, help="Optional detailed JSON, beneath repository outputs/")
    args = parser.parse_args()
    report = build_report(ROOT, json.loads(args.manifest.read_text(encoding="utf-8")))
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to((ROOT / "outputs").resolve()):
            parser.error("--output must be beneath repository outputs/")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {"groups": {name: {key: group[key] for key in ("file_count", "python_count", "lines", "missing_seeds")}
                          for name, group in report["groups"].items()},
               "candidate_retained_by": {path: row["retained_by"] for path, row in report["retirement_candidates"].items()},
               "parse_error_count": len(report["parse_errors"]),
               "active_parse_error_count": len(report["active_parse_errors"])}
    print(json.dumps(summary, indent=2))
    return int(bool(report["active_parse_errors"] or any(group["missing_seeds"] for group in report["groups"].values())))


if __name__ == "__main__":
    raise SystemExit(main())
