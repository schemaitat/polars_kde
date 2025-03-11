from __future__ import annotations
from typing import TYPE_CHECKING

import polars as pl
from pathlib import Path

from polars.plugins import register_plugin_function


LIB = Path(__file__).parent

if TYPE_CHECKING:
    from polars_kde.typing import IntoExprColumn


def kde(expr: IntoExprColumn, *, eval_points: list[float]) -> pl.Expr:
    """Kernel Density Estimation (KDE) aggregation.

    Args:
        expr (IntoExprColumn): Which column to aggregate into a population.
        eval_points (list[float]): At which points to evaluate the KDE.

    Returns:
        pl.Expr: The KDE evaluated at the given points.
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="kde_agg",
        is_elementwise=False,
        returns_scalar=False,
        kwargs={"eval_points": eval_points},
    )


def kde_static_evals(expr: IntoExprColumn, *, eval_points: list[float]) -> pl.Expr:
    """
    Kernel Density Estimation (KDE) evaluation on already aggregated data.
    Takes a column of lists of floats and evaluates the KDE at the given points.

    Args:
        expr (IntoExprColumn): Column of lists of floats (pl.List(pl.Float32)).
        eval_points (list[float]): At which points to evaluate the KDE.

    Returns:
        pl.Expr: The KDE evaluated at the given points.
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="kde_static_evals",
        is_elementwise=True,
        kwargs={"eval_points": eval_points},
    )


def kde_dynamic_evals(expr: IntoExprColumn, eval_points: IntoExprColumn) -> pl.Expr:
    """
    Kernel Density Estimation (KDE) evaluation on already aggregated data but with dynamic eval points.
    Takes a column of lists of floats and evaluates the KDE at the given points (which are defined row-wise)
    and can be different for each row.

    Args:
        expr (IntoExprColumn): Column of lists of floats (pl.List(pl.Float32)).
        eval_points (IntoExprColumn): Column of lists of floats (pl.List(pl.Float32)).
    """
    return register_plugin_function(
        args=[expr, eval_points],
        plugin_path=LIB,
        function_name="kde_dynamic_evals",
        is_elementwise=True,
    )


__all__ = ["__version__", "kde", "kde_static_evals", "kde_dynamic_evals"]
