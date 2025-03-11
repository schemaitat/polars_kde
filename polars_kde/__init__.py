from __future__ import annotations
from typing import TYPE_CHECKING

import polars as pl
from pathlib import Path

from polars.plugins import register_plugin_function


LIB = Path(__file__).parent

if TYPE_CHECKING:
    from polars_kde.typing import IntoExprColumn


def kde(expr: IntoExprColumn, *, eval_points: list[float]) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="kde_agg",
        is_elementwise=False,
        returns_scalar=False,
        kwargs={"eval_points": eval_points},
    )


def kde_static_evals(expr: IntoExprColumn, *, eval_points: list[float]) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="kde_static_evals",
        is_elementwise=True,
        kwargs={"eval_points": eval_points},
    )


def kde_dynamic_evals(expr: IntoExprColumn, eval_points: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr, eval_points],
        plugin_path=LIB,
        function_name="kde_dynamic_evals",
        is_elementwise=True,
    )


__all__ = ["__version__", "kde", "kde_static_evals", "kde_dynamic_evals"]
