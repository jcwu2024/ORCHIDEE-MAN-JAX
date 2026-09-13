"""Host-only pool accounting for diagnostics; never alters model states."""
import numpy as np

from research.daily_coarse_graining.constrained_daily_replay import CARBON_INVENTORY_FIELDS

from .successor_training import _state_leaf


def reduce_pool(values, fraction, active, leaf, field):
    """Independently sum named pool/layer axes once, in land-area units."""
    values = np.asarray(values, dtype=np.float64)
    fraction = np.asarray(fraction, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    days = values.shape[0]
    fraction = fraction.reshape(days, -1)
    active = active.reshape(days, -1)
    if field == 'interception_storage':
        # Already per land area: soilcarbon_leak pool_start/end add directly
        # (stomate_soilcarbon.f90:1245-1251,2326-2332).
        fraction = np.ones_like(fraction)
    axis = 1 + leaf.axis_names.index('nvm')
    shape = [1] * values.ndim
    shape[0], shape[axis] = days, fraction.shape[-1]
    weight = np.where(active, fraction, 0.).reshape(shape)
    if field != 'carbon_32l':
        # Fortran carbon element is index1; Python index0, last array axis.
        values = values[..., :1]
    weighted = np.where(weight != 0., values * weight, 0.)
    return np.sum(weighted, axis=tuple(range(1, weighted.ndim)))


def state_pools(states, fraction, active, contract):
    states = np.asarray(states)
    result = {}
    for field in CARBON_INVENTORY_FIELDS:
        leaf = _state_leaf(contract, field)
        values = states[:, leaf.start:leaf.stop].reshape((len(states),) + tuple(leaf.shape))
        result[field] = reduce_pool(values, fraction, active, leaf, field)
    return result


def inventory_pools(vectors, fraction, active, contract):
    vectors = np.asarray(vectors)
    result, cursor = {}, 0
    for field in CARBON_INVENTORY_FIELDS:
        leaf = _state_leaf(contract, field)
        size = int(np.prod(leaf.shape))
        values = vectors[:, cursor:cursor + size].reshape((len(vectors),) + tuple(leaf.shape))
        result[field] = reduce_pool(values, fraction, active, leaf, field)
        cursor += size
    if cursor != vectors.shape[1]:
        raise ValueError('inventory and compact state pool layout differ')
    return result


def split_pool_changes(start, fast, final, incoming, outgoing):
    """Separate predicted fast endpoint change from retained-tail change."""
    return {key: {
        'start': start[key], 'fast_endpoint': fast[key], 'final': final[key],
        'incoming': incoming[key], 'outgoing': outgoing[key],
        'fast_change': fast[key] - start[key],
        'tail_change': final[key] - fast[key],
        'endpoint_identity_residual': fast[key] - start[key] - incoming[key] + outgoing[key],
    } for key in start}
