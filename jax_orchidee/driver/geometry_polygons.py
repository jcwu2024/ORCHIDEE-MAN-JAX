"""Source-faithful polygon helpers from ORCHIDEE ``polygones.f90``.

The routines intentionally keep the source loop order and exact comparisons.
They do not normalize closed polygons, remove repeated crossings, or replace
the source algorithms with a geometry library.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


POLYGONES_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_global/polygones.f90::polygones_pointinside lines 20-59",
    "fortran_source/ORCHIDEE/src_global/polygones.f90::polygones_lineintersect lines 63-100",
    "fortran_source/ORCHIDEE/src_global/polygones.f90::polygones_extend lines 104-170",
    "fortran_source/ORCHIDEE/src_global/polygones.f90::polygones_intersection lines 174-256",
    "fortran_source/ORCHIDEE/src_global/polygones.f90::polygones_cleanup lines 260-339",
    "fortran_source/ORCHIDEE/src_global/polygones.f90::polygones_area lines 343-377",
    "fortran_source/ORCHIDEE/src_global/polygones.f90::polygones_crossing lines 381-465",
    "fortran_source/ORCHIDEE/src_global/polygones.f90::polygones_convexhull lines 469-731",
)


class PolygonResult(NamedTuple):
    nvert: int
    vertices: np.ndarray


class LineIntersectionResult(NamedTuple):
    intersection: bool
    point_x: float
    point_y: float


class FortranUndefinedOutputError(RuntimeError):
    """The Fortran branch returns before assigning an ``INTENT(out)`` value."""


def _polygon(value, *, name: str) -> np.ndarray:
    poly = np.asarray(value, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError(f"{name} cannot be a polygon: expected shape (n, 2), got {poly.shape}")
    return poly


def _check_prefix(poly: np.ndarray, nvert: int, *, name: str) -> None:
    if nvert < 0:
        raise ValueError("nvert must be non-negative")
    if poly.shape[0] < nvert:
        raise ValueError(f"{name} is smaller than nvert={nvert}")


def _capacity(required: int, output_capacity: int | None, routine: str) -> int:
    capacity = required if output_capacity is None else int(output_capacity)
    if capacity < required:
        raise ValueError(f"{routine}: output polygon too small ({capacity} < {required})")
    return capacity


def polygones_pointinside(nvert_in: int, poly, point_x: float, point_y: float) -> bool:
    """Exact ray crossing test from lines 20-59; boundary is source-dependent."""

    vertices = _polygon(poly, name="poly")
    _check_prefix(vertices, nvert_in, name="poly")
    inside = False
    j = nvert_in - 1
    for i in range(nvert_in):
        if (vertices[i, 1] > point_y) != (vertices[j, 1] > point_y):
            xonline = (
                (vertices[j, 0] - vertices[i, 0])
                * (point_y - vertices[i, 1])
                / (vertices[j, 1] - vertices[i, 1])
                + vertices[i, 0]
            )
            if point_x < xonline:
                inside = not inside
        j = i
    return inside


def polygones_lineintersect(la, lb) -> LineIntersectionResult:
    """Intersect two closed segments, retaining the source parallel-line arm."""

    line_a = _polygon(la, name="la")
    line_b = _polygon(lb, name="lb")
    if line_a.shape != (2, 2) or line_b.shape != (2, 2):
        raise ValueError("la and lb must both have shape (2, 2)")
    x1, y1 = line_a[0]
    x2, y2 = line_a[1]
    x3, y3 = line_b[0]
    x4, y4 = line_b[1]
    den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(den) > np.finfo(np.asarray(den).dtype).eps:
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
        if 0.0 <= ua <= 1.0 and 0.0 <= ub <= 1.0:
            return LineIntersectionResult(True, float(x1 + ua * (x2 - x1)), float(y1 + ua * (y2 - y1)))
    # point_x/point_y are undefined in the source whenever intersection is false.
    return LineIntersectionResult(False, np.nan, np.nan)


def polygones_extend(
    nvert_in: int, poly_in, nbdots: int, *, output_capacity: int | None = None
) -> PolygonResult:
    """Add ``nbdots`` samples to every last-to-first source edge."""

    vertices = _polygon(poly_in, name="poly_in")
    _check_prefix(vertices, nvert_in, name="poly_in")
    required = nvert_in * nbdots
    _capacity(required, output_capacity, "polygones_extend")
    if nbdots <= 0:
        if nbdots < 0:
            raise ValueError("polygones_extend: negative output extent")
        return PolygonResult(0, np.empty((0, 2), dtype=vertices.dtype))
    out = np.empty((required, 2), dtype=vertices.dtype)
    j = nvert_in - 1
    ipos = 0
    for i in range(nvert_in):
        xs, ys = vertices[j]
        xe, ye = vertices[i]
        for ident in range(1, nbdots + 1):
            if xs == xe:
                out[ipos] = (xs, ys + (ident - 1) * (ye - ys) / nbdots)
            elif ys == ye:
                out[ipos] = (xs + (ident - 1) * (xe - xs) / nbdots, ys)
            else:
                y = ys + (ident - 1) * (ye - ys) / nbdots
                out[ipos] = ((xs - xe) * (y - ye) / (ys - ye) + xe, y)
            ipos += 1
        j = i
    return PolygonResult(required, out)


def polygones_intersection(
    nvert_a: int,
    poly_a,
    nvert_b: int,
    poly_b,
    *,
    output_capacity: int | None = None,
    poly_b_capacity: int | None = None,
) -> PolygonResult:
    """Collect sampled vertices strictly classified inside the other polygon."""

    a = _polygon(poly_a, name="poly_a")
    b = _polygon(poly_b, name="poly_b")
    _check_prefix(a, nvert_a, name="poly_a")
    # Line 203 compares the allocated poly_b extent against nvert_a, not nvert_b.
    # Compact Python arrays can carry the original workspace extent explicitly.
    allocated_b = b.shape[0] if poly_b_capacity is None else int(poly_b_capacity)
    if allocated_b < b.shape[0]:
        raise ValueError("poly_b_capacity cannot be smaller than the supplied array")
    if allocated_b < nvert_a:
        raise ValueError("polygones_intersection: poly_b is smaller than nvert_a")
    in_a = [i for i in range(nvert_a) if polygones_pointinside(nvert_b, b, a[i, 0], a[i, 1])]
    in_b = [i for i in range(nvert_b) if polygones_pointinside(nvert_a, a, b[i, 0], b[i, 1])]
    required = len(in_a) + len(in_b)
    _capacity(required, output_capacity, "polygones_intersection")
    if required == 0:
        return PolygonResult(0, np.empty((0, 2), dtype=np.result_type(a, b)))
    out = np.concatenate((a[in_a], b[in_b]), axis=0)
    return PolygonResult(required, out)


def polygones_cleanup(
    nvert_in: int, poly_in, *, output_capacity: int | None = None
) -> PolygonResult:
    """Nearest-neighbour ordering with exact duplicate deletion (lines 260-339)."""

    vertices = _polygon(poly_in, name="poly_in")
    _check_prefix(vertices, nvert_in, name="poly_in")
    capacity = nvert_in if output_capacity is None else int(output_capacity)
    if nvert_in == 0:
        raise IndexError("polygones_cleanup: source reads poly_in(1,:) for nvert_in=0")
    remaining = vertices[1:nvert_in].copy()
    ordered = [vertices[0].copy()]
    eps = np.finfo(vertices.dtype).eps
    while remaining.shape[0] > 0:
        distances = np.sqrt(np.sum((remaining - ordered[-1]) ** 2, axis=1))
        nearest = int(np.argmin(distances))
        if distances[nearest] > eps:
            if len(ordered) + 1 > capacity:
                raise ValueError("polygones_cleanup: output polygon too small")
            ordered.append(remaining[nearest].copy())
        remaining = np.delete(remaining, nearest, axis=0)
    out = np.asarray(ordered, dtype=vertices.dtype)
    return PolygonResult(out.shape[0], out)


def polygones_area(nvert_in: int, poly_in, dx: float, dy: float) -> float:
    """Source shoelace loop, including its absolute ``dx*dy`` scaling."""

    vertices = _polygon(poly_in, name="poly_in")
    _check_prefix(vertices, nvert_in, name="poly_in")
    area = np.asarray(0.0, dtype=np.result_type(vertices, dx, dy)).item()
    j = nvert_in - 1
    for i in range(nvert_in):
        area += dy * dx / 2.0 * (vertices[j, 1] + vertices[i, 1]) * (
            vertices[j, 0] - vertices[i, 0]
        )
        j = i
    return float(abs(area))


def polygones_crossing(
    nvert_a: int,
    poly_a,
    nvert_b: int,
    poly_b,
    *,
    output_capacity: int | None = None,
) -> PolygonResult:
    """Return crossings in source edge-loop order, including duplicates."""

    a = _polygon(poly_a, name="poly_a")
    b = _polygon(poly_b, name="poly_b")
    _check_prefix(a, nvert_a, name="poly_a")
    _check_prefix(b, nvert_b, name="poly_b")
    capacity = nvert_a * nvert_b if output_capacity is None else int(output_capacity)
    points: list[tuple[float, float]] = []
    ja = nvert_a - 1
    for ia in range(nvert_a):
        la = a[[ja, ia], :]
        jb = nvert_b - 1
        for ib in range(nvert_b):
            result = polygones_lineintersect(la, b[[jb, ib], :])
            if result.intersection:
                if len(points) + 1 > capacity:
                    raise ValueError("polygones_crossing: output polygon too small")
                points.append((result.point_x, result.point_y))
            jb = ib
        ja = ia
    out = np.asarray(points, dtype=np.result_type(a, b)).reshape((-1, 2))
    return PolygonResult(len(points), out)


def polygones_convexhull(
    nvert_in: int, poly_in, *, output_capacity: int | None = None
) -> PolygonResult:
    """Literal 1-based translation of the Alan Miller scan at lines 469-731."""

    poly = _polygon(poly_in, name="poly_in")
    _check_prefix(poly, nvert_in, name="poly_in")
    if poly.shape[0] <= 2:
        raise ValueError("polygones_convexhull: input polygon is too small")
    capacity = nvert_in if output_capacity is None else int(output_capacity)
    vertex = np.zeros(nvert_in + 1, dtype=np.int64)
    iwk = np.zeros(nvert_in + 1, dtype=np.int64)
    next_point = np.zeros(nvert_in * 20 + 1, dtype=np.int64)

    def x(index: int) -> float:
        return poly[index - 1, 0]

    def y(index: int) -> float:
        return poly[index - 1, 1]

    if x(1) > x(nvert_in):
        vertex[1], vertex[2] = nvert_in, 1
    else:
        vertex[1], vertex[2] = 1, nvert_in
    xmin, xmax = x(vertex[1]), x(vertex[2])
    for i in range(2, nvert_in):
        if x(i) < xmin:
            vertex[1], xmin = i, x(i)
        elif x(i) > xmax:
            vertex[2], xmax = i, x(i)

    if xmax == xmin:
        if y(1) > y(nvert_in):
            vertex[1], vertex[2] = nvert_in, 1
        else:
            vertex[1], vertex[2] = 1, nvert_in
        ymin, ymax = y(vertex[1]), y(vertex[2])
        for i in range(2, nvert_in):
            if y(i) < ymin:
                vertex[1], ymin = i, y(i)
            elif y(i) > ymax:
                vertex[2], ymax = i, y(i)
        # Lines 571-573 assign local nvert then RETURN. INTENT(out) values remain undefined.
        count = 1 if ymax == ymin else 2
        raise FortranUndefinedOutputError(
            f"polygones_convexhull lines 547-573 return with local nvert={count} "
            "before assigning nvert_out or poly_out"
        )

    i1, i2 = int(vertex[1]), int(vertex[2])
    iwk[i1] = iwk[i2] = -1
    dx = xmax - xmin
    y1 = y(i1)
    dy = y(i2) - y1
    dmax = dmin = 0.0
    next_point[1] = next_point[2] = -1
    for i in range(1, nvert_in + 1):
        if i == vertex[1] or i == vertex[2]:
            continue
        dist = (y(i) - y1) * dx - (x(i) - xmin) * dy
        if dist > 0.0:
            iwk[i1], i1 = i, i
            if dist > dmax:
                next_point[1], dmax = i, dist
        elif dist < 0.0:
            iwk[i2], i2 = i, i
            if dist < dmin:
                next_point[2], dmin = i, dist
    iwk[i1] = iwk[i2] = -1
    nvert = 2
    j = 1
    points_todo = True
    while points_todo:
        while points_todo and next_point[j] < 0:
            if j == nvert:
                points_todo = False
            j += 1
        if points_todo:
            jp1 = j + 1
            if jp1 >= nvert_in * 20:
                raise ValueError("polygones_convexhull: please increase nextinc")
            for i in range(nvert, jp1 - 1, -1):
                vertex[i + 1] = vertex[i]
                next_point[i + 1] = next_point[i]
            jp2 = jp1 + 1
            nvert += 1
            if jp2 > nvert:
                jp2 = 1
            i1 = int(vertex[j])
            i2 = int(next_point[j])
            i3 = int(vertex[jp2])
            vertex[jp1] = i2
            x1, x2 = x(i1), x(i2)
            y1, y2 = y(i1), y(i2)
            dx1, dx2 = x2 - x1, x(i3) - x2
            dy1, dy2 = y2 - y1, y(i3) - y2
            dmax1 = dmax2 = 0.0
            next_point[j] = next_point[jp1] = -1
            i2save = i2
            i2next = int(iwk[i2])
            i = int(iwk[i1])
            iwk[i1] = iwk[i2] = -1
            while i > 0:
                if i != i2save:
                    dist = (y(i) - y1) * dx1 - (x(i) - x1) * dy1
                    if dist > 0.0:
                        iwk[i1], i1 = i, i
                        if dist > dmax1:
                            next_point[j], dmax1 = i, dist
                    else:
                        dist = (y(i) - y2) * dx2 - (x(i) - x2) * dy2
                        if dist > 0.0:
                            iwk[i2], i2 = i, i
                            if dist > dmax2:
                                next_point[jp1], dmax2 = i, dist
                    i = int(iwk[i])
                else:
                    i = i2next
            iwk[i1] = iwk[i2] = -1
    if capacity < nvert:
        raise ValueError("polygones_convexhull: output polygon is smaller than nvert_out")
    out = np.asarray([poly[vertex[i] - 1] for i in range(1, nvert + 1)], dtype=poly.dtype)
    return PolygonResult(nvert, out)
