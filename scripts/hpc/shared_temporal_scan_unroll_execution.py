"""Diagnostic-only adapter for static temporal lax.scan partial unrolling."""
from __future__ import annotations

from typing import Any


class _LaxScanUnroll:
    def __init__(self, lax: Any, tree_util: Any, unroll: int):
        if unroll not in (1, 2):
            raise ValueError("the bounded screen supports only unroll 1 or 2")
        self._lax = lax
        self._tree_util = tree_util
        self.unroll = unroll
        self.calls: list[dict[str, int]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lax, name)

    def scan(self, f: Any, init: Any, xs: Any, **kwargs: Any) -> Any:
        if "unroll" in kwargs:
            raise ValueError("the source scan must not already specify unroll")
        leaves = self._tree_util.tree_leaves(xs)
        lengths = {int(leaf.shape[0]) for leaf in leaves if getattr(leaf, "shape", ())}
        if len(lengths) != 1:
            raise ValueError("temporal scan inputs must share one leading length")
        length = lengths.pop()
        self.calls.append({"length": length, "unroll": self.unroll})
        return self._lax.scan(f, init, xs, unroll=self.unroll, **kwargs)


class TemporalUnrollJax:
    """JAX facade with scan unrolling and an optional shared checkpoint setting."""

    def __init__(self, jax: Any, unroll: int, *, checkpoint_prevent_cse: bool | None = None):
        self._jax = jax
        self.lax = _LaxScanUnroll(jax.lax, jax.tree_util, unroll)
        self._checkpoint_prevent_cse = checkpoint_prevent_cse
        self.checkpoint_calls: list[dict[str, object]] = []

    def checkpoint(self, function: Any, **kwargs: Any) -> Any:
        if self._checkpoint_prevent_cse is None:
            self.checkpoint_calls.append({"name": function.__name__, "kwargs": sorted(kwargs)})
            return self._jax.checkpoint(function, **kwargs)
        if function.__name__ != "transition" or kwargs:
            raise ValueError("the screen must preserve RolloutLoss's default checkpoint policy")
        self.checkpoint_calls.append({"name": function.__name__, "prevent_cse": self._checkpoint_prevent_cse,
                                      "policy": None})
        return self._jax.checkpoint(function, prevent_cse=self._checkpoint_prevent_cse)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._jax, name)
