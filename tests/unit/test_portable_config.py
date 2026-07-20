from __future__ import annotations

from pathlib import Path

from jax_orchidee.driver.domain import load_case_config


def test_config_root_tokens_follow_environment(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    config_dir = repo / "configs"
    config_dir.mkdir(parents=True)
    config = config_dir / "case.yaml"
    config.write_text(
        "paths:\n"
        "  repo: ${ORCHIDEE_REPO_ROOT}\n"
        "  data: ${ORCHIDEE_DATA_ROOT}/forcing\n"
        "  reference: ${ORCHIDEE_REFERENCE_ROOT}/paper\n"
        "  output: ${ORCHIDEE_OUTPUT_ROOT}/run\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ORCHIDEE_DATA_ROOT", str(tmp_path / "external-data"))
    monkeypatch.setenv("ORCHIDEE_REFERENCE_ROOT", str(tmp_path / "external-reference"))
    monkeypatch.setenv("ORCHIDEE_OUTPUT_ROOT", str(tmp_path / "scratch"))

    loaded = load_case_config(config)

    assert loaded["paths"]["repo"] == str(repo)
    assert loaded["paths"]["data"] == str(tmp_path / "external-data" / "forcing")
    assert loaded["paths"]["reference"] == str(tmp_path / "external-reference" / "paper")
    assert loaded["paths"]["output"] == str(tmp_path / "scratch" / "run")


def test_config_root_tokens_fall_back_to_checkout(monkeypatch, tmp_path: Path):
    for name in (
        "ORCHIDEE_REPO_ROOT",
        "ORCHIDEE_DATA_ROOT",
        "ORCHIDEE_REFERENCE_ROOT",
        "ORCHIDEE_OUTPUT_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    repo = tmp_path / "repo"
    config_dir = repo / "configs"
    config_dir.mkdir(parents=True)
    config = config_dir / "case.yaml"
    config.write_text(
        "paths:\n"
        "  data: ${ORCHIDEE_DATA_ROOT}\n"
        "  reference: ${ORCHIDEE_REFERENCE_ROOT}\n"
        "  output: ${ORCHIDEE_OUTPUT_ROOT}\n",
        encoding="utf-8",
    )

    loaded = load_case_config(config)

    assert loaded["paths"]["data"] == str(repo / "data")
    assert loaded["paths"]["reference"] == str(repo / "reference")
    assert loaded["paths"]["output"] == str(repo / "outputs")
