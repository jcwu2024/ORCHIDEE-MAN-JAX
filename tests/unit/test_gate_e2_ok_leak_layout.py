from pathlib import Path

from scripts.dev.benchmark_gate_e2_ok_leak_layout import benchmark

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = (
    ROOT
    / "outputs/research/daily_coarse_graining/gate_c2_daily_flux_capture"
    / "daily_flux_labels.npz"
)


def test_bounded_ok_leak_layouts_are_lossless_and_cover_required_fields():
    report = benchmark(CAPTURE)

    assert report["sample"]["field_count"] == 24
    assert report["sample"]["raw_value_bytes"] == 279_872
    assert report["sample"]["bit_nonzero_fraction"] < 0.02
    assert report["hybrid_layout_counts"]["constant_zero"] > 0
    assert all(
        result["bit_exact_roundtrip"]
        for result in report["layouts"].values()
    )
    assert (
        report["layouts"]["dense_deflate"]["stored_bytes"]
        < report["layouts"]["hybrid_auto_deflate"]["stored_bytes"]
    )
    assert report["decision"]["accepted_local_layout"] == "dense_deflate"
    assert report["decision"]["production_mask_semantics_validated"] is False
    assert report["decision"]["full_generation_authorized"] is False
