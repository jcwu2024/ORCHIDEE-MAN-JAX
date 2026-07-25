from scripts.hpc.run_counterfactual_teacher_gpu import main


def test_gpu_launcher_initializes_backend_before_import(monkeypatch):
    events = []

    monkeypatch.setattr(
        "scripts.hpc.run_counterfactual_teacher_gpu.initialize_gpu_backend",
        lambda: events.append("gpu"),
    )
    monkeypatch.setattr(
        "research.daily_coarse_graining.counterfactual_teacher_diagnostic.main",
        lambda argv: events.append(("diagnostic", argv)) or 0,
    )

    assert main(["--example"]) == 0
    assert events == ["gpu", ("diagnostic", ["--example"])]
