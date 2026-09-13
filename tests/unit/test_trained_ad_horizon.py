"""Keep the intended T30 case distinct from the accepted T7 durable gradient."""
import pytest

from scripts.hpc.probe_trained_checkpointed_grad import selected_window


def test_horizon_uses_first_frozen_window_without_result_selection():
    t7 = {'length':7,'artifact_id':1887,'window_index':5}
    t30 = {'length':30,'artifact_id':1887,'window_index':2}
    panel = {'windows':[t7,t30,{'length':30,'artifact_id':2546,'window_index':3}]}
    assert selected_window(panel,7) is t7
    assert selected_window(panel,30) is t30
    with pytest.raises(ValueError):
        selected_window(panel,365)
