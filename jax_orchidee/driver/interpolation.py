"""Pure array owners extracted from ORCHIDEE ``interpweight``.

File IO, polygon aggregation, and MPI remain separate boundaries.  These
helpers preserve the source loop order because the 2-D and 4-D ``msumrange``
branches mutate values that later loop iterations read again.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple


INTERPWEIGHT_INPUT_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_modifying_input1D lines 2759-2779",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_modifying_input2D lines 2797-2817",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_modifying_input3D lines 2835-2855",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_modifying_input4D lines 2873-2893",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_masking_input1D lines 2911-2961",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_masking_input2D lines 2979-3048",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_masking_input3D lines 3066-3132",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_masking_input4D lines 3150-3221",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_provide_fractions1D lines 3239-3346",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_provide_fractions2D lines 3364-3473",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_provide_fractions4D lines 3661-3778",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_calc_resolution_in lines 2688-2741",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_ValVecR lines 4902-4960",
    "fortran_source/ORCHIDEE/src_global/interpweight.f90::interpweight_provide_interpolation2D lines 3796-3936",
)


class InterpweightFractionsResult(NamedTuple):
    fractions: np.ndarray
    availability: np.ndarray


class InterpweightValVecRResult(NamedTuple):
    """Defined prefix of the Fortran result; its unused tail is uninitialized."""

    count: int
    stored_positions: np.ndarray


def _ranked_array(value, rank: int, *, name: str, dtype=None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got shape {array.shape}")
    return array.copy()


def _modify(value, zeroval, rank: int) -> np.ndarray:
    array = _ranked_array(value, rank, name="ivar")
    return np.where(array < zeroval, zeroval, array)


def interpweight_modifying_input1d(ivar, zeroval) -> np.ndarray:
    return _modify(ivar, zeroval, 1)


def interpweight_modifying_input2d(ivar, zeroval) -> np.ndarray:
    return _modify(ivar, zeroval, 2)


def interpweight_modifying_input3d(ivar, zeroval) -> np.ndarray:
    return _modify(ivar, zeroval, 3)


def interpweight_modifying_input4d(ivar, zeroval) -> np.ndarray:
    return _modify(ivar, zeroval, 4)


def _mask_inputs(var, msk, mvalues, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _ranked_array(var, rank, name="var")
    mask = _ranked_array(msk, 1 if rank == 1 else 2, name="msk")
    expected = values.shape if rank == 1 else values.shape[:2]
    if mask.shape != expected:
        raise ValueError(f"msk must have shape {expected}, got {mask.shape}")
    thresholds = np.asarray(mvalues, dtype=values.dtype)
    if thresholds.shape != (3,):
        raise ValueError("mvalues must have shape (3,)")
    return values, mask, thresholds


def interpweight_masking_input1d(var, msk, mtype: str, mvalues, *, un: int = 1):
    values, mask, thresholds = _mask_inputs(var, msk, mvalues, 1)
    if mtype == "nomask":
        mask[...] = un
    elif mtype == "mbelow":
        mask[values < thresholds[0]] = un
    elif mtype == "mabove":
        mask[values > thresholds[0]] = un
    elif mtype == "msumrange":
        raise RuntimeError("interpweight_masking_input1D: 'msumrange' no sens on a 1D variable")
    else:
        raise ValueError(f"interpweight_masking_input1D: mask with {mtype!r} not ready")
    return values, mask


def interpweight_masking_input2d(var, msk, mtype: str, mvalues, *, un: int = 1):
    values, mask, thresholds = _mask_inputs(var, msk, mvalues, 2)
    if mtype == "nomask":
        mask[...] = un
    elif mtype == "mbelow":
        mask[values < thresholds[0]] = un
    elif mtype == "mabove":
        mask[values > thresholds[0]] = un
    elif mtype == "msumrange":
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                sumval = np.sum(values[i, :])
                if thresholds[1] <= sumval <= thresholds[0]:
                    mask[i, j] = un
                elif thresholds[0] < sumval <= thresholds[2]:
                    values[i, j] = values[i, j] / sumval
                    mask[i, j] = un
    else:
        raise ValueError(f"interpweight_masking_input2D: mask with {mtype!r} not ready")
    return values, mask


def interpweight_masking_input3d(var, msk, mtype: str, mvalues, *, un: int = 1):
    values, mask, thresholds = _mask_inputs(var, msk, mvalues, 3)
    if mtype == "nomask":
        mask[...] = un
    elif mtype == "mbelow":
        mask[np.any(values < thresholds[0], axis=2)] = un
    elif mtype == "mabove":
        mask[np.any(values > thresholds[0], axis=2)] = un
    elif mtype == "msumrange":
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                sumval = np.sum(values[i, j, :])
                if thresholds[1] <= sumval <= thresholds[0]:
                    mask[i, j] = un
                elif thresholds[0] < sumval <= thresholds[2]:
                    values[i, j, :] = values[i, j, :] / sumval
                    mask[i, j] = un
    else:
        raise ValueError(f"interpweight_masking_input3D: mask with {mtype!r} not ready")
    return values, mask


def interpweight_masking_input4d(var, msk, mtype: str, mvalues, *, un: int = 1):
    values, mask, thresholds = _mask_inputs(var, msk, mvalues, 4)
    if mtype == "nomask":
        mask[...] = un
    elif mtype == "mbelow":
        mask[np.any(values < thresholds[0], axis=(2, 3))] = un
    elif mtype == "mabove":
        mask[np.any(values > thresholds[0], axis=(2, 3))] = un
    elif mtype == "msumrange":
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                for layer in range(values.shape[3]):
                    sumval = np.sum(values[i, j, :, layer])
                    if thresholds[1] <= sumval <= thresholds[0] and mask[i, j] != un:
                        mask[i, j] = un
                    elif thresholds[0] < sumval <= thresholds[2]:
                        values[i, j, :, layer] = values[i, j, :, layer] / sumval
                        mask[i, j] = un
    else:
        raise ValueError(f"interpweight_masking_input4D: mask with {mtype!r} not ready")
    return values, mask


def _fraction_geometry(sarea, sindex, *, source_rank: int):
    areas = _ranked_array(sarea, 2, name="sarea")
    indices = _ranked_array(sindex, 2 if source_rank == 1 else 3, name="sindex", dtype=np.int64)
    expected = areas.shape if source_rank == 1 else areas.shape + (2,)
    if indices.shape != expected:
        raise ValueError(f"sindex must have shape {expected}, got {indices.shape}")
    return areas, indices


def _default_fraction_index(vmin, vmax, two, ntypes: int) -> int:
    index = int((vmax + vmin) / two) - 1
    if index < 0 or index >= ntypes:
        raise IndexError("Fortran fallback fraction index is outside Ntypes")
    return index


def interpweight_provide_fractions1d(
    ivar,
    sarea,
    sindex,
    valstypes,
    *,
    tint="default",
    zeroval=0.0,
    vmin=1.0,
    vmax=1.0,
    two=2.0,
) -> InterpweightFractionsResult:
    """Exact ``interpweight_provide_fractions1D`` lines 3239-3346."""

    values = _ranked_array(ivar, 1, name="ivar1D")
    areas, indices = _fraction_geometry(sarea, sindex, source_rank=1)
    types = _ranked_array(valstypes, 1, name="valstypes")
    if tint != "default":
        raise ValueError(f"interpweight_provide_fractions1D: interpolation type {tint!r} not ready")
    if np.all(values == zeroval):
        raise ValueError("interpweight_provide_fractions1D: wrong 1D input variable")
    out = np.full((areas.shape[0], types.size), zeroval, dtype=np.result_type(values, areas))
    availability = np.empty(areas.shape[0], dtype=out.dtype)
    for point in range(areas.shape[0]):
        availability[point] = zeroval
        count = int(np.count_nonzero(areas[point, :] > zeroval))
        if count > 0:
            for overlap in range(count):
                source = int(indices[point, overlap]) - 1
                if source < 0 or source >= values.size:
                    raise IndexError("sindex contains an invalid Fortran 1-based source index")
                for kind in range(types.size):
                    if values[source] == types[kind]:
                        out[point, kind] += areas[point, overlap]
                availability[point] += areas[point, overlap]
            out[point, :] /= availability[point]
        else:
            availability[point] = -1.0
            out[point, _default_fraction_index(vmin, vmax, two, types.size)] = 1.0
        if np.sum(out[point, :]) > 1.0000001:
            raise ValueError("interpweight_provide_fractions1D: total of fractions above 1")
    return InterpweightFractionsResult(out, availability)


def interpweight_provide_fractions2d(
    ivar,
    sarea,
    sindex,
    valstypes,
    *,
    tint="default",
    zeroval=0.0,
    vmin=1.0,
    vmax=1.0,
    two=2.0,
) -> InterpweightFractionsResult:
    """Exact ``interpweight_provide_fractions2D`` lines 3364-3473."""

    values = _ranked_array(ivar, 2, name="ivar2D")
    areas, indices = _fraction_geometry(sarea, sindex, source_rank=2)
    types = _ranked_array(valstypes, 1, name="valstypes")
    if tint != "default":
        raise ValueError(f"interpweight_provide_fractions2D: interpolation type {tint!r} not ready")
    if np.all(values == zeroval):
        raise ValueError("interpweight_provide_fractions2D: wrong 2D input variable")
    out = np.full((areas.shape[0], types.size), zeroval, dtype=np.result_type(values, areas))
    availability = np.empty(areas.shape[0], dtype=out.dtype)
    for point in range(areas.shape[0]):
        availability[point] = zeroval
        count = int(np.count_nonzero(areas[point, :] > zeroval))
        if count > 0:
            for overlap in range(count):
                source = indices[point, overlap, :] - 1
                i, j = int(source[0]), int(source[1])
                if i < 0 or i >= values.shape[0] or j < 0 or j >= values.shape[1]:
                    raise IndexError("sindex contains an invalid Fortran 1-based source index")
                for kind in range(types.size):
                    if values[i, j] == types[kind]:
                        out[point, kind] += areas[point, overlap]
                availability[point] += areas[point, overlap]
            out[point, :] /= availability[point]
        else:
            availability[point] = -1.0
            out[point, _default_fraction_index(vmin, vmax, two, types.size)] = 1.0
        if np.sum(out[point, :]) > 1.0000001:
            raise ValueError("interpweight_provide_fractions2D: total of fractions above 1")
    return InterpweightFractionsResult(out, availability)


def interpweight_provide_fractions4d(
    ivar,
    sarea,
    sindex,
    valstypes,
    *,
    tint="default",
    zeroval=0.0,
    vmin=None,
    vmax=None,
    two=2.0,
) -> InterpweightFractionsResult:
    """Exact ``interpweight_provide_fractions4D`` lines 3661-3778."""

    values = _ranked_array(ivar, 4, name="ivar4D")
    areas, indices = _fraction_geometry(sarea, sindex, source_rank=2)
    types = _ranked_array(valstypes, 1, name="valstypes")
    if types.size != values.shape[2]:
        raise ValueError("Different number of types than third dimension in the input data")
    if tint != "default":
        raise ValueError(f"interpweight_provide_fractions4D: interpolation type {tint!r} not ready")
    if np.all(values == zeroval):
        raise ValueError("interpweight_provide_fractions4D: wrong 4D input variable")
    vmin_arr = np.ones(types.size) if vmin is None else _ranked_array(vmin, 1, name="vmin")
    vmax_arr = np.ones(types.size) if vmax is None else _ranked_array(vmax, 1, name="vmax")
    if vmin_arr.shape != types.shape or vmax_arr.shape != types.shape:
        raise ValueError("vmin and vmax must have shape (Ntypes,)")
    out = np.full(
        (areas.shape[0], types.size, values.shape[3]),
        zeroval,
        dtype=np.result_type(values, areas),
    )
    availability = np.empty(areas.shape[0], dtype=out.dtype)
    for point in range(areas.shape[0]):
        availability[point] = zeroval
        count = int(np.count_nonzero(areas[point, :] > zeroval))
        if count > 0:
            for overlap in range(count):
                source = indices[point, overlap, :] - 1
                i, j = int(source[0]), int(source[1])
                if i < 0 or i >= values.shape[0] or j < 0 or j >= values.shape[1]:
                    raise IndexError("sindex contains an invalid Fortran 1-based source index")
                for kind in range(types.size):
                    for time in range(values.shape[3]):
                        value = values[i, j, kind, time]
                        if value > zeroval and value < 20.0:
                            out[point, kind, time] += value * areas[point, overlap]
                availability[point] += areas[point, overlap]
            out[point, :, :] /= availability[point]
        else:
            availability[point] = -1.0
            for kind in range(types.size):
                fallback = _default_fraction_index(vmin_arr[kind], vmax_arr[kind], two, types.size)
                for time in range(values.shape[3]):
                    out[point, fallback, time] = 1.0
    return InterpweightFractionsResult(out, availability)


def interpweight_calc_resolution_in(
    lons,
    lats,
    *,
    mcos=0.001,
    r_earth=6_378_000.0,
    pi_value=np.pi,
) -> np.ndarray:
    """Exact regular-grid resolution loop at ``interpweight.f90`` 2688-2741."""

    lon = _ranked_array(lons, 2, name="lons", dtype=np.float64)
    lat = _ranked_array(lats, 2, name="lats", dtype=np.float64)
    if lon.shape != lat.shape:
        raise ValueError("lons and lats must have the same shape")
    dx, dy = lon.shape
    if dx < 2 or dy < 2:
        raise ValueError("Fortran resolution calculation requires dx>=2 and dy>=2")
    result = np.empty((dx, dy, 2), dtype=np.result_type(lon, lat))
    for ip in range(dx):
        for jp in range(dy):
            coslat = max(np.cos(lat[ip, jp] * pi_value / 180.0), mcos)
            if ip == 0:
                delta_lon = abs(lon[ip + 1, jp] - lon[ip, jp])
            elif ip == dx - 1:
                delta_lon = abs(lon[ip, jp] - lon[ip - 1, jp])
            else:
                delta_lon = abs(lon[ip + 1, jp] - lon[ip - 1, jp]) / 2.0
            result[ip, jp, 0] = delta_lon * pi_value / 180.0 * r_earth * coslat
            if jp == 0:
                delta_lat = abs(lat[ip, jp] - lat[ip, jp + 1])
            elif jp == dy - 1:
                delta_lat = abs(lat[ip, jp - 1] - lat[ip, jp])
            else:
                delta_lat = abs(lat[ip, jp - 1] - lat[ip, jp + 1]) / 2.0
            result[ip, jp, 1] = delta_lat * pi_value / 180.0 * r_earth
    return result


def interpweight_valvecr(vec, val, oper: str) -> InterpweightValVecRResult:
    """Preserve the assigned portion of ``interpweight_ValVecR`` 4902-4960."""

    values = _ranked_array(vec, 1, name="vec")
    predicates = {
        "eq": lambda item: item == val,
        "ge": lambda item: item >= val,
        "le": lambda item: item <= val,
        "neq": lambda item: item != val,
    }
    if oper not in predicates:
        raise ValueError("interpweight_ValVecR supports only 'eq', 'ge', 'le', and 'neq'")
    count = 0
    stored: list[int] = []
    for index, item in enumerate(values, start=1):
        if predicates[oper](item):
            count += 1
            if count < values.size:
                stored.append(index)
    return InterpweightValVecRResult(count, np.asarray(stored, dtype=np.int32))


def interpweight_provide_interpolation2d(
    ivar,
    sarea,
    sindex,
    *,
    tint="default",
    zeroval=0.0,
    defaultval=0.0,
    default_no_value=1.0,
) -> InterpweightFractionsResult:
    """Exact source-ordered interpolation at ``interpweight.f90`` 3796-3936."""

    values = _ranked_array(ivar, 2, name="ivar2D")
    areas, indices = _fraction_geometry(sarea, sindex, source_rank=2)
    if tint not in {"default", "slopecalc"}:
        raise ValueError(f"interpweight_provide_interpolation2D: interpolation type {tint!r} not ready")
    if np.all(values == zeroval):
        raise ValueError(f"interpweight_provide_interpolation2D: {tint!r} requires a nonzero 2D input variable")
    out = np.full(areas.shape[0], zeroval, dtype=np.result_type(values, areas))
    availability = np.empty(areas.shape[0], dtype=out.dtype)
    for point in range(areas.shape[0]):
        weighted = zeroval
        last_index = areas.shape[1]
        availability[point] = zeroval
        for overlap in range(areas.shape[1]):
            if areas[point, overlap] <= zeroval:
                last_index = overlap
                break
            source = indices[point, overlap, :] - 1
            i, j = int(source[0]), int(source[1])
            if i < 0 or i >= values.shape[0] or j < 0 or j >= values.shape[1]:
                raise IndexError("sindex contains an invalid Fortran 1-based source index")
            value = values[i, j]
            if tint == "default":
                weighted += value * areas[point, overlap]
            else:
                weighted += min(value / default_no_value, 1.0) * areas[point, overlap]
            availability[point] += areas[point, overlap]
        if last_index >= 1:
            mean = weighted / availability[point]
            out[point] = mean if tint == "default" else 1.0 - mean
        else:
            out[point] = defaultval
            availability[point] = -1.0
    return InterpweightFractionsResult(out, availability)
