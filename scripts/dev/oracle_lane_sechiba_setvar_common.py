from __future__ import annotations

import csv
from typing import Any

import numpy as np

from jax_orchidee.sechiba import io_contracts as io


IX = np.int32(-9)
RX = np.float64(-9.0)
NO_KEY = "NO_KEYWORD"


class OracleReader:
    def read(self, key: str, default: Any) -> np.ndarray:
        value = np.asarray(default)
        if value.ndim == 0:
            result = 101 if value.dtype.kind in {"i", "u"} else 101.5
            return np.asarray(result, dtype=value.dtype)
        indices = np.indices(value.shape)
        if value.ndim == 1:
            result = 101.0 + indices[0]
        elif value.ndim == 2:
            result = 100.0 + 10.0 * (indices[0] + 1) + (indices[1] + 1)
        else:
            raise ValueError(f"unsupported OracleReader rank {value.ndim}")
        if value.dtype.kind == "f":
            result = result + 0.5
        return np.asarray(result, dtype=value.dtype)


class MissingReader:
    def read(self, key: str, default: Any) -> np.ndarray:
        return np.array(default, copy=True)


def read_csv(path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _keep(shape: tuple[int, ...], exceptional, dtype) -> np.ndarray:
    value = np.full(shape, exceptional, dtype=dtype)
    value.flat[0] = 5
    return value


def jax_outputs(*, parallel: bool) -> dict[str, np.ndarray]:
    suffix = "_p" if parallel else ""
    reader = OracleReader()
    outputs: dict[str, np.ndarray] = {}

    def call(name: str, field: str, var, val_exp, key, val_put, *, config=None, **kwargs) -> None:
        function = getattr(io, name + suffix)
        result = function(var, val_exp, key, val_put, config=config, **kwargs)
        outputs[field] = np.asarray(result).ravel(order="C")

    for kind, exceptional, dtype, prefix in (
        ("i", IX, np.int32, "i"),
        ("r", RX, np.float64, "r"),
    ):
        scalar_put = dtype(7)
        call(f"{prefix}0setvar", f"{prefix}0.no", dtype(exceptional), exceptional, NO_KEY, scalar_put)
        call(f"{prefix}0setvar", f"{prefix}0.cfg", dtype(exceptional), exceptional, f"CFG_{kind}0", scalar_put, config=reader)
        call(f"{prefix}0setvar", f"{prefix}0.keep", dtype(5), exceptional, f"CFG_{kind}0", scalar_put, config=reader)

        vec_put = np.asarray([7, 8, 9], dtype=dtype)
        sentinel_vec = np.full(3, exceptional, dtype=dtype)
        call(f"{prefix}10setvar", f"{prefix}10.no", sentinel_vec, exceptional, NO_KEY, scalar_put)
        call(f"{prefix}10setvar", f"{prefix}10.cfg", sentinel_vec, exceptional, f"CFG_{kind}0", scalar_put, config=reader)
        call(f"{prefix}10setvar", f"{prefix}10.keep", _keep((3,), exceptional, dtype), exceptional, f"CFG_{kind}0", scalar_put, config=reader)
        call(f"{prefix}11setvar", f"{prefix}11.no", sentinel_vec, exceptional, NO_KEY, vec_put)
        call(f"{prefix}11setvar", f"{prefix}11.cfg", sentinel_vec, exceptional, f"CFG_{kind}1", vec_put, config=reader)
        call(f"{prefix}11setvar", f"{prefix}11.keep", _keep((3,), exceptional, dtype), exceptional, f"CFG_{kind}1", vec_put, config=reader)

        sentinel_23 = np.full((2, 3), exceptional, dtype=dtype)
        call(f"{prefix}20setvar", f"{prefix}20.no", sentinel_23, exceptional, NO_KEY, scalar_put)
        call(f"{prefix}20setvar", f"{prefix}20.cfg", sentinel_23, exceptional, f"CFG_{kind}0", scalar_put, config=reader)
        call(f"{prefix}20setvar", f"{prefix}20.keep", _keep((2, 3), exceptional, dtype), exceptional, f"CFG_{kind}0", scalar_put, config=reader)

        put2 = np.asarray([7, 8], dtype=dtype)
        sentinel_22 = np.full((2, 2), exceptional, dtype=dtype)
        call(f"{prefix}21setvar", f"{prefix}21.first.no", sentinel_22, exceptional, NO_KEY, put2)
        call(f"{prefix}21setvar", f"{prefix}21.first.cfg", sentinel_22, exceptional, f"CFG_{kind}1", put2, config=reader)
        call(f"{prefix}21setvar", f"{prefix}21.first.keep", _keep((2, 2), exceptional, dtype), exceptional, f"CFG_{kind}1", put2, config=reader)
        call(f"{prefix}21setvar", f"{prefix}21.second.no", sentinel_23, exceptional, NO_KEY, vec_put)
        call(f"{prefix}21setvar", f"{prefix}21.second.cfg", sentinel_23, exceptional, f"CFG_{kind}1", vec_put, config=reader)
        call(f"{prefix}21setvar", f"{prefix}21.second.keep", _keep((2, 3), exceptional, dtype), exceptional, f"CFG_{kind}1", vec_put, config=reader)

        matrix_put = np.asarray([[1, 3, 5], [2, 4, 6]], dtype=dtype)
        call(f"{prefix}22setvar", f"{prefix}22.no", sentinel_23, exceptional, NO_KEY, matrix_put)
        call(f"{prefix}22setvar", f"{prefix}22.cfg", sentinel_23, exceptional, f"CFG_{kind}2", matrix_put, config=reader)
        call(f"{prefix}22setvar", f"{prefix}22.keep", _keep((2, 3), exceptional, dtype), exceptional, f"CFG_{kind}2", matrix_put, config=reader)

        if prefix == "r":
            cube = np.full((2, 2, 2), exceptional, dtype=dtype)
            call("r30setvar", "r30.no", cube, exceptional, NO_KEY, scalar_put)
            call("r30setvar", "r30.cfg", cube, exceptional, "CFG_r0", scalar_put, config=reader)
            call("r30setvar", "r30.keep", _keep((2, 2, 2), exceptional, dtype), exceptional, "CFG_r0", scalar_put, config=reader)
    return outputs
