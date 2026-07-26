from scripts.hpc.run_spatial_conditioning_gpu import main


def test_gpu_launcher_initializes_backend_before_diagnostic_import(monkeypatch):
    events = []
    monkeypatch.setattr(
        "scripts.hpc.run_spatial_conditioning_gpu.initialize_gpu_backend",
        lambda: events.append("gpu"),
    )
    monkeypatch.setattr(
        "research.daily_coarse_graining.spatial_conditioning_diagnostic.main",
        lambda argv: events.append(("diagnostic", argv)) or 0,
    )

    assert main(["--example"]) == 0
    assert events == ["gpu", ("diagnostic", ["--example"])]
