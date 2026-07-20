from __future__ import annotations

from pathlib import Path

from jax_orchidee.cli.main import main
from jax_orchidee.runtime import configure_jax_compilation_cache


def test_public_cli_help_and_version(capsys):
    assert main(["--help"]) == 0
    assert "validate-landpoints" in capsys.readouterr().out

    assert main(["--version"]) == 0
    assert "orchidee-man-jax" in capsys.readouterr().out


def test_compilation_cache_uses_external_output_root(monkeypatch, tmp_path: Path):
    output_root = tmp_path / "external-output"
    monkeypatch.setenv("ORCHIDEE_OUTPUT_ROOT", str(output_root))

    cache = configure_jax_compilation_cache(tmp_path / "checkout")

    assert cache == output_root / "xla_cache"
    assert cache.is_dir()
