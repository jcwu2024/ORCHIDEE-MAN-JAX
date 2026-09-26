"""Source-only inventory checks; no JAX imports or runtime data required."""
from scripts.dev.research_workspace_inventory import analyze_sources, closure


def test_import_closure_keeps_package_exports_and_from_modules():
    sources = {
        "entry.py": "from pkg import current\n",
        "pkg/__init__.py": "from .old_helper import useful\n",
        "pkg/current.py": "from .nested import live\n",
        "pkg/old_helper.py": "def useful(): pass\n",
        "pkg/nested/__init__.py": "from .. import other\n",
        "pkg/nested/live.py": "import pkg.other\n",
        "pkg/other.py": "",
        "unused.py": "",
    }
    edges, _, _, errors = analyze_sources(sources)
    assert not errors
    assert closure(["entry.py"], edges) == sorted(set(sources) - {"unused.py"})


def test_dynamic_calls_literal_config_and_shell_entrypoints_are_visible():
    sources = {
        "entry.py": "import importlib\nimportlib.import_module('pkg.live')\n"
                    "importlib.util.spec_from_file_location('candidate', args.path)\n"
                    "config = ROOT / 'manifests' / 'active.json'\n",
        "pkg/__init__.py": "",
        "pkg/live.py": "",
        "manifests/active.json": '{"overlay": "overlay.py"}',
        "overlay.py": "",
        "run.sh": "python -m pkg.live\npython entry.py\n",
        "history.md": "Historical command uses overlay.py.\n",
    }
    edges, references, dynamic, _ = analyze_sources(sources)
    assert set(closure(["entry.py"], edges)) == {
        "entry.py", "pkg/__init__.py", "pkg/live.py", "manifests/active.json", "overlay.py"
    }
    assert set(closure(["run.sh"], edges)) == set(sources) - {"history.md"}
    assert dynamic["entry.py"] == [{"line": 2, "call": "import_module"},
                                 {"line": 3, "call": "spec_from_file_location"}]
    assert {"path": "history.md", "line": 1} in references["overlay.py"]


def test_parse_error_is_not_silent_dead_code():
    _, _, _, errors = analyze_sources({"broken.py": "def broken(:\n"})
    assert errors["broken.py"]["line"] == 1


def test_malformed_lazy_initializer_is_reported_as_parse_error():
    edges, _, _, errors = analyze_sources({
        "entry.py": "import pkg\n",
        "pkg/__init__.py": "_PUBLIC_EXPORTS = {'live': ('.current', 'live')\n",
    })
    assert errors["pkg/__init__.py"]["line"] == 1
    assert "pkg/__init__.py" in closure(["entry.py"], edges)


def test_lazy_public_exports_follow_only_requested_names():
    sources = {
        "entry.py": "from pkg import live\n",
        "pkg/__init__.py": "_PUBLIC_EXPORTS = {'live': ('.current', 'live'), 'old': ('.historical', 'old')}\n"
                           "_COMPAT_EXPORTS = {'hidden': ('.hidden', 'hidden')}\n"
                           "_COMPAT_MODULES = {'alias': '.hidden'}\n",
        "pkg/current.py": "live = 1\n",
        "pkg/historical.py": "old = 1\n",
        "pkg/hidden.py": "hidden = 1\n",
    }
    edges, _, _, _ = analyze_sources(sources)
    assert set(closure(["entry.py"], edges)) == {"entry.py", "pkg/__init__.py", "pkg/current.py"}
    sources["entry.py"] = "from pkg import alias\n"
    edges, _, _, _ = analyze_sources(sources)
    assert set(closure(["entry.py"], edges)) == {"entry.py", "pkg/__init__.py", "pkg/hidden.py"}
    sources["entry.py"] = "from pkg import *\n"
    edges, _, _, _ = analyze_sources(sources)
    assert "pkg/hidden.py" not in closure(["entry.py"], edges)
