from __future__ import annotations

from jax_orchidee.driver.paper_validation import (
    PaperValidationPoint,
    annual_delta_passes,
    select_representative_paper_points,
)


def _point(index: int) -> PaperValidationPoint:
    value = float(index)
    return PaperValidationPoint(
        landpoint_id=f"{index:03d}.0-071.0",
        iteration_id="I1",
        param_set=f"S1_{value}_{value}_{value}_{value}",
        target_year=2010,
        grid_x=value,
        grid_y=71.0,
        vcmax25=value,
        maint_resp_slope_c=value,
        alloc_min=value,
        residence_time=value,
        AGB_model=value,
        BGB_model=value,
        GPP_model=value,
        NPP_model=value,
    )


def test_representative_selection_is_deterministic_and_keeps_anchor():
    points = tuple(_point(index) for index in range(1, 8))
    first = select_representative_paper_points(points, count=4, include_ids=("003.0-071.0",))
    second = select_representative_paper_points(points, count=4, include_ids=("003.0-071.0",))

    assert first == second
    assert first[0].landpoint_id == "003.0-071.0"
    assert len({point.landpoint_id for point in first}) == 4


def test_annual_delta_tolerance_uses_absolute_and_relative_terms():
    delta = {
        "GPP_model": {"reference": 10.0, "abs_error": 2.5e-6},
        "NPP_model": {"reference": 1.0, "abs_error": 1.0e-6},
    }

    assert annual_delta_passes(delta, atol=2.0e-6, rtol=1.0e-7)
    assert not annual_delta_passes(delta, atol=1.0e-6, rtol=1.0e-8)
    assert not annual_delta_passes(None, atol=2.0e-6, rtol=1.0e-7)
