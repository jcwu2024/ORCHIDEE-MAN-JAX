"""Typed ``setvar`` contracts routed from SECHIBA's Fortran IO helpers.

These routines are host-side configuration/parallel-IO boundaries.  They
return new values instead of mutating JAX arrays, while retaining Fortran's
generic rank/type dispatch and whole-array exceptional-value tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


SERIAL_SOURCE = "fortran_source/ORCHIDEE/src_sechiba/sechiba_io.f90"
PARALLEL_SOURCE = "fortran_source/ORCHIDEE/src_sechiba/sechiba_io_p.f90"


class SetvarContractError(ValueError):
    """Raised when a value cannot bind to the selected Fortran overload."""


class ConfigurationBoundaryError(RuntimeError):
    """Raised when a source ``getin`` call has no explicit reader."""


class ParallelIOBoundaryError(RuntimeError):
    """Raised when the grid gather/read/scatter path has no backend."""


@runtime_checkable
class ConfigurationReader(Protocol):
    """Typed equivalent of IOIPSL ``getin``/``getin_p``."""

    def read(self, key: str, default: Any) -> Any:
        """Read ``key``, retaining ``default`` when the key is absent."""


@runtime_checkable
class GridParallelIO(Protocol):
    """Explicit transport boundary used by ``i11setvar_p`` grid values."""

    def gather(self, local_value: np.ndarray) -> np.ndarray:
        """Gather a local vector into Fortran global one-based index order."""

    def root_read(self, key: str, global_default: np.ndarray) -> np.ndarray:
        """Run serial ``getin`` on the root-side global vector."""

    def scatter(self, global_value: np.ndarray) -> np.ndarray:
        """Scatter the global vector back into local one-based index order."""


@dataclass(frozen=True)
class MappingConfigurationReader:
    """Deterministic key-routed configuration source for tests and runtimes."""

    values: Mapping[str, Any]

    def read(self, key: str, default: Any) -> Any:
        return self.values.get(key, default)


@dataclass(frozen=True)
class SerialGridParallelIO:
    """The exact single-process gather/getin/scatter specialization.

    ``global_indices_one_based`` is checked even though serial gather/scatter
    are identity operations.  This keeps the distributed index convention at
    the boundary instead of silently converting it to Python indexing.
    """

    reader: ConfigurationReader
    global_indices_one_based: np.ndarray

    def _indices(self, size: int) -> np.ndarray:
        indices = np.asarray(self.global_indices_one_based)
        expected = np.arange(1, size + 1, dtype=indices.dtype)
        if indices.ndim != 1 or indices.shape != (size,) or not np.array_equal(indices, expected):
            raise ParallelIOBoundaryError(
                "serial gather/scatter indices must be contiguous Fortran one-based 1..N"
            )
        return indices

    def gather(self, local_value: np.ndarray) -> np.ndarray:
        self._indices(local_value.size)
        return np.array(local_value, copy=True)

    def root_read(self, key: str, global_default: np.ndarray) -> np.ndarray:
        return np.asarray(self.reader.read(key, np.array(global_default, copy=True)))

    def scatter(self, global_value: np.ndarray) -> np.ndarray:
        self._indices(global_value.size)
        return np.array(global_value, copy=True)


_ABSENT = object()


def _uses_keyword(key_wd: str) -> bool:
    if not isinstance(key_wd, str):
        raise SetvarContractError("key_wd must be a character string")
    return "NO_KEYWORD" not in key_wd and "NOKEYWORD" not in key_wd


def _array(value: Any, *, rank: int, kind: str, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != rank:
        raise SetvarContractError(f"{name} must have rank {rank}, got rank {result.ndim}")
    accepted = result.dtype.kind in ({"i", "u"} if kind == "integer" else {"f"})
    if not accepted:
        raise SetvarContractError(f"{name} must have Fortran {kind} type")
    return result


def _read(reader: ConfigurationReader | None, key: str, default: np.ndarray) -> np.ndarray:
    if reader is None:
        raise ConfigurationBoundaryError(f"setvar key {key!r} requires an explicit configuration reader")
    result = np.asarray(reader.read(key, np.array(default, copy=True)))
    if result.dtype.kind != default.dtype.kind:
        raise SetvarContractError(f"configuration value for {key!r} has incompatible type")
    if result.shape != default.shape:
        raise SetvarContractError(
            f"configuration value for {key!r} has shape {result.shape}, expected {default.shape}"
        )
    return result


def _spread_21(var: np.ndarray, value: np.ndarray) -> np.ndarray:
    # Exact SPREAD calls at sechiba_io.f90:192/202 and _p.f90:208/220.
    if value.size == var.shape[0]:
        spread = np.repeat(value[:, None], var.shape[0], axis=1)
    elif value.size == var.shape[1]:
        spread = np.repeat(value[None, :], var.shape[0], axis=0)
    else:
        raise SetvarContractError(
            f"rank-1 val_put length {value.size} matches neither var dimension {var.shape}"
        )
    if spread.shape != var.shape:
        raise SetvarContractError(
            f"Fortran SPREAD result {spread.shape} is not conformable with var shape {var.shape}"
        )
    return spread


def _setvar_impl(
    var: Any,
    val_exp: Any,
    key_wd: str,
    val_put: Any,
    *,
    kind: str,
    var_rank: int,
    put_rank: int,
    reader: ConfigurationReader | None,
    parallel_r22: bool = False,
    grid_optional: Any = _ABSENT,
    grid_io: GridParallelIO | None = None,
) -> Any:
    target = _array(var, rank=var_rank, kind=kind, name="var")
    exceptional = _array(val_exp, rank=0, kind=kind, name="val_exp")
    put = _array(val_put, rank=put_rank, kind=kind, name="val_put")

    if put_rank == var_rank and put_rank > 0 and put.shape != target.shape:
        raise SetvarContractError(f"val_put shape {put.shape} must equal var shape {target.shape}")
    if var_rank == 2 and put_rank == 1:
        # The source checks this even when var is not wholly exceptional.
        _spread_21(target, put)

    if not bool(np.all(target == exceptional)):
        return target.item() if var_rank == 0 else np.array(target, copy=True)

    value = np.array(put, copy=True)
    if _uses_keyword(key_wd):
        if grid_optional is not _ABSENT:
            if grid_io is None:
                raise ParallelIOBoundaryError(
                    "PRESENT(is_grid) requires an explicit gather/root-read/scatter backend"
                )
            gathered = np.asarray(grid_io.gather(value))
            if gathered.ndim != 1 or gathered.dtype.kind != value.dtype.kind:
                raise SetvarContractError("grid gather must preserve rank-1 Fortran type")
            root_value = np.asarray(grid_io.root_read(key_wd, gathered))
            if root_value.shape != gathered.shape or root_value.dtype.kind != gathered.dtype.kind:
                raise SetvarContractError("root getin must preserve gathered shape and type")
            value = np.asarray(grid_io.scatter(root_value))
            if value.shape != put.shape or value.dtype.kind != put.dtype.kind:
                raise SetvarContractError("grid scatter must restore local shape and type")
        elif parallel_r22:
            scalar_default = np.asarray(exceptional)
            scalar_value = _read(reader, key_wd, scalar_default)
            if bool(scalar_value != exceptional):
                value = np.full(put.shape, scalar_value.item(), dtype=put.dtype)
        else:
            value = _read(reader, key_wd, value)

    if var_rank == 2 and put_rank == 1:
        value = _spread_21(target, value)
    elif put_rank == 0 and var_rank > 0:
        value = np.full(target.shape, value.item(), dtype=target.dtype)
    elif value.shape != target.shape:
        raise SetvarContractError(f"assignment shape {value.shape} must equal var shape {target.shape}")

    cast = np.asarray(value, dtype=target.dtype)
    return cast.item() if var_rank == 0 else np.array(cast, copy=True)


def _serial(var: Any, val_exp: Any, key_wd: str, val_put: Any, *, kind: str, vr: int, pr: int,
            config: ConfigurationReader | None = None) -> Any:
    return _setvar_impl(var, val_exp, key_wd, val_put, kind=kind, var_rank=vr, put_rank=pr, reader=config)


def _parallel(var: Any, val_exp: Any, key_wd: str, val_put: Any, *, kind: str, vr: int, pr: int,
              config: ConfigurationReader | None = None, is_grid: Any = _ABSENT,
              grid_io: GridParallelIO | None = None, r22: bool = False) -> Any:
    return _setvar_impl(var, val_exp, key_wd, val_put, kind=kind, var_rank=vr, put_rank=pr,
                        reader=config, parallel_r22=r22, grid_optional=is_grid, grid_io=grid_io)


def i0setvar(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io.f90::i0setvar``, lines 48-69."""
    return _serial(var, val_exp, key_wd, val_put, kind="integer", vr=0, pr=0, config=config)


def i10setvar(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io.f90::i10setvar``, lines 73-94."""
    return _serial(var, val_exp, key_wd, val_put, kind="integer", vr=1, pr=0, config=config)


def i11setvar(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io.f90::i11setvar``, lines 98-122."""
    return _serial(var, val_exp, key_wd, val_put, kind="integer", vr=1, pr=1, config=config)


def i20setvar(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io.f90::i20setvar``, lines 126-151."""
    return _serial(var, val_exp, key_wd, val_put, kind="integer", vr=2, pr=0, config=config)


def i21setvar(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io.f90::i21setvar``, lines 165-213."""
    return _serial(var, val_exp, key_wd, val_put, kind="integer", vr=2, pr=1, config=config)


def i22setvar(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io.f90::i22setvar``, lines 216-240."""
    return _serial(var, val_exp, key_wd, val_put, kind="integer", vr=2, pr=2, config=config)


def r0setvar(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io.f90::r0setvar``, lines 244-265."""
    return _serial(var, val_exp, key_wd, val_put, kind="real", vr=0, pr=0, config=config)


def r10setvar(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io.f90::r10setvar``, lines 269-290."""
    return _serial(var, val_exp, key_wd, val_put, kind="real", vr=1, pr=0, config=config)


def r11setvar(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io.f90::r11setvar``, lines 294-318."""
    return _serial(var, val_exp, key_wd, val_put, kind="real", vr=1, pr=1, config=config)


def r20setvar(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io.f90::r20setvar``, lines 322-345."""
    return _serial(var, val_exp, key_wd, val_put, kind="real", vr=2, pr=0, config=config)


def r21setvar(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io.f90::r21setvar``, lines 359-409."""
    return _serial(var, val_exp, key_wd, val_put, kind="real", vr=2, pr=1, config=config)


def r22setvar(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io.f90::r22setvar``, lines 413-439."""
    return _serial(var, val_exp, key_wd, val_put, kind="real", vr=2, pr=2, config=config)


def r30setvar(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io.f90::r30setvar``, lines 442-465."""
    return _serial(var, val_exp, key_wd, val_put, kind="real", vr=3, pr=0, config=config)


def i0setvar_p(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io_p.f90::i0setvar_p``, lines 42-65."""
    return _parallel(var, val_exp, key_wd, val_put, kind="integer", vr=0, pr=0, config=config)


def i10setvar_p(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io_p.f90::i10setvar_p``, lines 69-92."""
    return _parallel(var, val_exp, key_wd, val_put, kind="integer", vr=1, pr=0, config=config)


def i11setvar_p(var, val_exp, key_wd, val_put, *, config=None, is_grid=_ABSENT, grid_io=None):
    """``sechiba_io_p.f90::i11setvar_p``, lines 96-137."""
    return _parallel(var, val_exp, key_wd, val_put, kind="integer", vr=1, pr=1, config=config,
                     is_grid=is_grid, grid_io=grid_io)


def i20setvar_p(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io_p.f90::i20setvar_p``, lines 141-164."""
    return _parallel(var, val_exp, key_wd, val_put, kind="integer", vr=2, pr=0, config=config)


def i21setvar_p(var, val_exp, key_wd, val_put, *, config=None, is_grid=_ABSENT):
    """``sechiba_io_p.f90::i21setvar_p``, lines 178-231; ``is_grid`` is unused."""
    return _parallel(var, val_exp, key_wd, val_put, kind="integer", vr=2, pr=1, config=config)


def i22setvar_p(var, val_exp, key_wd, val_put, *, config=None, is_grid=_ABSENT):
    """``sechiba_io_p.f90::i22setvar_p``, lines 234-261; ``is_grid`` is unused."""
    return _parallel(var, val_exp, key_wd, val_put, kind="integer", vr=2, pr=2, config=config)


def r0setvar_p(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io_p.f90::r0setvar_p``, lines 265-288."""
    return _parallel(var, val_exp, key_wd, val_put, kind="real", vr=0, pr=0, config=config)


def r10setvar_p(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io_p.f90::r10setvar_p``, lines 292-315."""
    return _parallel(var, val_exp, key_wd, val_put, kind="real", vr=1, pr=0, config=config)


def r11setvar_p(var, val_exp, key_wd, val_put, *, config=None, is_grid=_ABSENT):
    """``sechiba_io_p.f90::r11setvar_p``, lines 319-346; ``is_grid`` is unused."""
    return _parallel(var, val_exp, key_wd, val_put, kind="real", vr=1, pr=1, config=config)


def r20setvar_p(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io_p.f90::r20setvar_p``, lines 350-375."""
    return _parallel(var, val_exp, key_wd, val_put, kind="real", vr=2, pr=0, config=config)


def r21setvar_p(var, val_exp, key_wd, val_put, *, config=None, is_grid=_ABSENT):
    """``sechiba_io_p.f90::r21setvar_p``, lines 389-444; ``is_grid`` is unused."""
    return _parallel(var, val_exp, key_wd, val_put, kind="real", vr=2, pr=1, config=config)


def r22setvar_p(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io_p.f90::r22setvar_p``, lines 448-482."""
    return _parallel(var, val_exp, key_wd, val_put, kind="real", vr=2, pr=2, config=config, r22=True)


def r30setvar_p(var, val_exp, key_wd, val_put, *, config=None):
    """``sechiba_io_p.f90::r30setvar_p``, lines 485-508."""
    return _parallel(var, val_exp, key_wd, val_put, kind="real", vr=3, pr=0, config=config)


_SERIAL_DISPATCH = {
    ("integer", 0, 0): i0setvar, ("integer", 1, 0): i10setvar,
    ("integer", 1, 1): i11setvar, ("integer", 2, 0): i20setvar,
    ("integer", 2, 1): i21setvar, ("integer", 2, 2): i22setvar,
    ("real", 0, 0): r0setvar, ("real", 1, 0): r10setvar,
    ("real", 1, 1): r11setvar, ("real", 2, 0): r20setvar,
    ("real", 2, 1): r21setvar, ("real", 2, 2): r22setvar,
    ("real", 3, 0): r30setvar,
}
_PARALLEL_DISPATCH = {
    key: globals()[function.__name__ + "_p"] for key, function in _SERIAL_DISPATCH.items()
}


def _dispatch(table, var, val_put):
    target = np.asarray(var)
    put = np.asarray(val_put)
    kind = "integer" if target.dtype.kind in {"i", "u"} else "real" if target.dtype.kind == "f" else None
    function = table.get((kind, target.ndim, put.ndim))
    if function is None:
        raise SetvarContractError(
            f"no Fortran setvar overload for type={target.dtype}, ranks=({target.ndim},{put.ndim})"
        )
    return function


def setvar(var, val_exp, key_wd, val_put, *, config=None):
    """Dispatch the 13-arm generic interface at ``sechiba_io.f90:30-33``."""
    return _dispatch(_SERIAL_DISPATCH, var, val_put)(var, val_exp, key_wd, val_put, config=config)


def setvar_p(var, val_exp, key_wd, val_put, *, config=None, is_grid=_ABSENT, grid_io=None):
    """Dispatch the 13-arm generic interface at ``sechiba_io_p.f90:31-34``."""
    function = _dispatch(_PARALLEL_DISPATCH, var, val_put)
    kwargs = {"config": config}
    if function is i11setvar_p:
        kwargs.update(is_grid=is_grid, grid_io=grid_io)
    elif function in {i21setvar_p, i22setvar_p, r11setvar_p, r21setvar_p}:
        kwargs.update(is_grid=is_grid)
    elif is_grid is not _ABSENT:
        raise SetvarContractError("is_grid is not an optional argument of the selected Fortran overload")
    return function(var, val_exp, key_wd, val_put, **kwargs)
