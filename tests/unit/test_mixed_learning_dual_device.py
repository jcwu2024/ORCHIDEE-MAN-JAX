from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from scripts.hpc.mixed_learning_dual_device import cost_comparison, split_components


def test_split_branches_overlap_and_keep_daily_recurrent_order():
    started = Event()
    release = Event()

    def recurrent():
        started.set()
        assert release.wait(5), 'daily branch did not run concurrently'
        return 'recurrent'

    def daily():
        assert started.wait(5), 'recurrent branch was not dispatched'
        release.set()
        return 'daily'

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert split_components(daily, recurrent, executor=executor) == ('daily', 'recurrent')


def test_failed_recurrent_branch_cannot_yield_an_update():
    def failed():
        raise RuntimeError('physical failure')

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match='physical failure'):
            split_components(lambda: 1, failed, executor=executor)


def test_serial_execution_keeps_original_branch_order():
    order = []
    split_components(lambda: order.append('daily'), lambda: order.append('recurrent'))
    assert order == ['daily', 'recurrent']


def test_cost_separates_wall_time_from_gpu_hours_and_cpu_charges():
    result = cost_comparison(12., 10.)
    assert result['speedup'] == pytest.approx(1.2)
    assert result['gpu_hour_ratio'] == pytest.approx(2/1.2)
    assert result['estimated_total_charge_ratio'] == pytest.approx(4.96/2.76/1.2)
    with pytest.raises(ValueError):
        cost_comparison(0., 10.)
