import numpy as np

from research.daily_coarse_graining.daily_operator.feedback_loss_geometry import (
    calibration_readout,
    joint_gradient_geometry,
)


def test_conflicting_gradient_can_worsen_base_despite_positive_weights():
    # g_base=(1,0), g_canopy=(-2,0), g_litter=(0,3).
    batch = {'groups': [
        {'name': name, 'gradient': {'l2_norm': norm}}
        for name, norm in [('base', 1.), ('canopy_stock_change', 2.), ('litterpart_feedback', 3.)]],
        'gradient_cross_geometry': {
            'base__canopy_stock_change': {'dot': -2.},
            'base__litterpart_feedback': {'dot': 0.},
            'canopy_stock_change__litterpart_feedback': {'dot': 0.}}}
    result = joint_gradient_geometry(batch, {'base': 1., 'canopy_stock_change': 1., 'litterpart_feedback': .5})
    np.testing.assert_allclose(result['raw_combined_gradient_norm'], np.sqrt(3.25))
    derivatives = result['component_derivative_along_negative_combined_gradient']
    np.testing.assert_allclose([derivatives[k] for k in ('base', 'canopy_stock_change', 'litterpart_feedback')],
                               [1., -2., -4.5])


def test_shape_only_completion_cannot_admit_weight_selection():
    result = calibration_readout({'status': 'completed', 'stage': 'completed'})
    assert not result['ready_for_weight_review']
    assert result['coefficient_selected'] is None
